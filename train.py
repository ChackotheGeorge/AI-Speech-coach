from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048
# Switched from 7B to 3B to fit easily inside RTX 2050's 4GB VRAM
model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

# 1. Load 3B 4-bit quantized model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 2. Add LoRA adapters for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 3. Format prompt template matching Qwen chat format
prompt_format = """Below is an instruction describing a task, paired with an input context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_prompts(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        text = prompt_format.format(inst, inp, out) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts }

# Load dataset from your path
dataset = load_dataset("json", data_files="dataset/dataset_200.json", split="train")
dataset = dataset.map(format_prompts, batched = True)

# 4. Configure SFTTrainer with gradient checkpointing for VRAM efficiency
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = False,                 # Set to False
        bf16 = True,                  # Set to True to match bfloat16 precision
        logging_steps = 1,
        output_dir = "speech_coach_lora",
    ),
)
trainer.train()

# 5. Save the trained LoRA adapters directly
model.save_pretrained("speech_coach_lora")
tokenizer.save_pretrained("speech_coach_lora")
print("Training complete! LoRA adapters saved to 'speech_coach_lora'.")