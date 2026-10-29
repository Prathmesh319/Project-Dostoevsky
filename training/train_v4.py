
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "SmolLM2-360M-Instruct"
DATA_FILE = ROOT / "library" / "personality_v4_philosophical.jsonl"
OUTPUT = ROOT / "models" / "dostoevsky-lora-v4"

print("Loading philosophical dataset...", flush=True)
rows = [
    json.loads(line)
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if len(rows) < 20:
    raise ValueError(f"Expected at least 20 examples; found {len(rows)}")

tokenizer = AutoTokenizer.from_pretrained(
    str(BASE), local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_example(row):
    text = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = Dataset.from_list([format_example(row) for row in rows])

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

print("Loading base model on CPU...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    str(BASE),
    local_files_only=True,
    torch_dtype=torch.float32,
)
model.config.use_cache = False

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=str(OUTPUT / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    use_cpu=True,
    dataloader_num_workers=0,
    fp16=False,
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)

print("Starting v4 CPU training...", flush=True)
trainer.train()

OUTPUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUTPUT))
tokenizer.save_pretrained(str(OUTPUT))

print(f"Training complete. Adapter saved to: {OUTPUT}", flush=True)




