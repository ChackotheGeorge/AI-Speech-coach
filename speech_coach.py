import sys
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer

# --- CONFIGURATION ---
WHISPER_MODEL_SIZE = "base"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096
SILENCE_THRESHOLD = 0.015    
SILENCE_DURATION_AUTO = 2.0  # 2-second pause triggers auto-feedback in continuous mode
MAX_SPEECH_DURATION = 30     

print("Loading Whisper Model on GPU...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")

print("Loading Fine-Tuned Speech Coach Model...")
base_model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_name,
    max_seq_length = 2048,
    load_in_4bit = True,
)

coach_model = PeftModel.from_pretrained(model, "speech_coach_lora")
FastLanguageModel.for_inference(coach_model)

print("\n" + "="*55)
print("🎙️ INTERACTIVE SPEECH COACH ACTIVE!")
print("Modes:")
print(" • Speak continuously: Pauses >= 2s trigger automated feedback.")
print(" • Say 'Now tell me what was wrong with this' (or similar) to evaluate on demand.")
print("="*55 + "\n")

audio_buffer = []
is_speaking = False
silence_counter = 0
speech_counter = 0
speech_start_time = 0

chunks_to_silence = int((SAMPLE_RATE / BLOCK_SIZE) * SILENCE_DURATION_AUTO)
max_chunks = int((SAMPLE_RATE / BLOCK_SIZE) * MAX_SPEECH_DURATION)

def callback(indata, frames, time_info, status):
    global audio_buffer, is_speaking, silence_counter, speech_counter, speech_start_time
    
    if status and not status.input_overflow:
        print(status, file=sys.stderr)
        
    audio_chunk = indata.flatten()
    volume = np.sqrt(np.mean(audio_chunk**2))
    
    if volume > SILENCE_THRESHOLD:
        if not is_speaking:
            is_speaking = True
            audio_buffer = []
            speech_start_time = time.time()
        
        audio_buffer.append(audio_chunk)
        silence_counter = 0
        speech_counter += 1
        
        if speech_counter >= max_chunks:
            process_utterance(triggered_by_command=False)
    else:
        if is_speaking:
            audio_buffer.append(audio_chunk)
            silence_counter += 1
            
            if silence_counter >= chunks_to_silence:
                process_utterance(triggered_by_command=False)

def process_utterance(triggered_by_command=False):
    global audio_buffer, is_speaking, silence_counter, speech_counter, speech_start_time
    if not audio_buffer:
        return
        
    full_audio = np.concatenate(audio_buffer).astype(np.float32)
    duration = time.time() - speech_start_time
    
    audio_buffer = []
    is_speaking = False
    silence_counter = 0
    speech_counter = 0
    
    # 1. Transcribe with Whisper
    segments, _ = whisper_model.transcribe(full_audio, beam_size=1, language="en")
    transcript_text = " ".join([segment.text.strip() for segment in segments if segment.text.strip()])
    
    if not transcript_text or len(transcript_text.split()) < 2:
        return 

    # 2. Check for On-Demand Evaluation Command
    # Looks for phrases indicating the user wants feedback right now
    lower_text = transcript_text.lower()
    command_phrases = ["what was wrong", "tell me what was wrong", "evaluate this", "how did I do", "critique this"]
    
    is_command_match = any(cmd in lower_text for cmd in command_phrases)

    if is_command_match:
        # Strip out the command phrase from the input sentence so the coach only evaluates the speech itself
        for cmd in command_phrases:
            if cmd in lower_text:
                # Find position and slice out the command portion
                idx = lower_text.find(cmd)
                transcript_text = transcript_text[:idx].strip()
                break
        
        if not transcript_text:
            print("\n🤖 Coach: You asked for a review, but no speech content preceded the command!")
            return
        
        print(f"\n[Command Detected] Evaluating requested segment...")
        triggered_by_command = True

    # If not explicitly commanded and under continuous mode, process normally after 2s pause
    if not triggered_by_command:
        # Optional: You can filter whether auto-mode runs every time or only on longer pauses
        pass

    # Calculate metrics
    word_count = len(transcript_text.split())
    wpm = int((word_count / max(duration, 0.5)) * 60)
    
    print(f"\n[Transcript] \"{transcript_text}\"")
    print(f"[Metrics] Duration: {duration:.1f}s | WPM: {wpm}")
    print("-" * 40)
    print("🤖 Coach Feedback:")

    # 3. Generate Local LLM Critique
    prompt = f"""Below is an instruction describing a task, paired with an input context. Write a response that appropriately completes the request.

### Instruction:
Analyze the following transcript segment and pacing metrics. Identify filler words, pacing issues, and offer actionable public speaking corrections.

### Input:
Transcript: '{transcript_text}' | WPM: {wpm} | Duration: {duration:.1f}s

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    _ = coach_model.generate(
        **inputs, 
        streamer=streamer, 
        max_new_tokens=350, 
        temperature=0.7,
        do_sample=True
    )
    print("\n" + "="*50 + "\n")

# Start audio capture stream
try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nStopping Speech Coach App. Great practice session!")