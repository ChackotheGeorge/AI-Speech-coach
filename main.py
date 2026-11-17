import queue
import sys
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# Configuration
MODEL_SIZE = "base"          # Use "base" or "small" for real-time speed on an RTX 2050
SAMPLE_RATE = 16000          # Whisper expects 16kHz audio sampling
BLOCK_DURATION = 4           # Seconds of audio per transcription chunk

# Queue to hold incoming audio chunks
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This function is called for every audio block by sounddevice."""
    if status:
        print(status, file=sys.stderr)
    # Copy the audio data into the queue
    audio_queue.put(indata.copy())

print("Loading Whisper model...")
# Initialize model on your RTX 2050 GPU using float16 for speed
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print(f"Model loaded! Starting live transcription. Press Ctrl+C to stop.\n")

# Calculate block size in samples based on duration
block_size = int(SAMPLE_RATE * BLOCK_DURATION)

try:
    # Open a continuous microphone input stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                        blocksize=block_size, callback=audio_callback):
        while True:
            # Gather audio chunks from the queue
            audio_chunk = audio_queue.get()
            
            # Flatten array and convert from float32 to float expected by whisper
            audio_data = audio_chunk.flatten().astype(np.float32)
            
            # Transcribe the current chunk
            segments, info = model.transcribe(audio_data, beam_size=1, language="en")
            
            # Print the recognized text instantly
            for segment in segments:
                if segment.text.strip():
                    print(f"[Live] {segment.text.strip()}")

except KeyboardInterrupt:
    print("\nStopped live transcription.")