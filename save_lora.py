import torch
from unsloth import FastLanguageModel

max_seq_length = 2048
model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

print("Loading fine-tuned model checkpoint from memory...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# Re-apply LoRA configuration matching your training setup
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# Save the trained adapters and tokenizer locally
print("Saving fine-tuned LoRA adapters to 'speech_coach_lora'...")
model.save_pretrained("speech_coach_lora")
tokenizer.save_pretrained("speech_coach_lora")

print("Done! Your fine-tuned speech coach weights are saved in 'C:\\AI PROJECT\\speech_coach_lora'.")