
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "SmolLM2-360M-Instruct"
DATA_PATH = ROOT / "training" / "personality_v3_combined.jsonl"
OUTPUT_PATH = ROOT / "models" / "dostoevsky-lora-v3"

MAX_LENGTH = 512

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"{OUTPUT_PATH} already exists. Choose a new output folder "
        "or inspect it before proceeding."
    )

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token is None:
    raise ValueError("Tokenizer has no pad token or EOS token.")

if not tokenizer.chat_template:
    raise ValueError("No chat template found.")

# SmolLM2's template uses these message boundaries.
ASSISTANT_START = "<|im_start|>assistant\n"
MESSAGE_END = "<|im_end|>"

class ConversationDataset(Dataset):
    def __init__(self, path):
        self.rows = []

        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(records)} records.")

        for record_number, record in enumerate(records, start=1):
            messages = record.get("messages")

            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Invalid messages in record {record_number}")

            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Find the exact character spans of assistant message bodies.
            spans = []
            cursor = 0

            while True:
                start = rendered.find(ASSISTANT_START, cursor)
                if start == -1:
                    break

                body_start = start + len(ASSISTANT_START)
                body_end = rendered.find(MESSAGE_END, body_start)

                if body_end == -1:
                    raise ValueError(
                        f"Unclosed assistant message in record {record_number}"
                    )

                spans.append((body_start, body_end))
                cursor = body_end + len(MESSAGE_END)

            if not spans:
                raise ValueError(
                    f"No assistant responses in record {record_number}"
                )

            # Tokenize the exact rendered conversation and retain character
            # offsets so labels align with the actual token sequence.
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )

            input_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            labels = [-100] * len(input_ids)

            for token_index, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end:
                    continue

                for body_start, body_end in spans:
                    if token_start < body_end and token_end > body_start:
                        labels[token_index] = input_ids[token_index]
                        break

            # Truncate only after the labels have been aligned.
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
            attention_mask = [1] * len(input_ids)

            if not any(label != -100 for label in labels):
                raise ValueError(
                    f"No assistant training tokens survived in record "
                    f"{record_number}. Check template and truncation."
                )

            self.rows.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        print(f"Prepared {len(self.rows)} conversations.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class AssistantOnlyCollator:
    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids, attention_masks, labels = [], [], []

        for item in features:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"] + [tokenizer.pad_token_id] * pad_len
            )
            attention_masks.append(
                item["attention_mask"] + [0] * pad_len
            )
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


print("Preparing dataset...")
dataset = ConversationDataset(DATA_PATH)

# Sanity check: confirm the dataset has real assistant labels.
sample = dataset[0]
labelled = sum(x != -100 for x in sample["labels"])
print(f"First record: {len(sample['input_ids'])} tokens")
print(f"First record: {labelled} assistant tokens labelled for loss")

if len(dataset) != 150:
    raise ValueError(
        f"Expected 150 records, found {len(dataset)}. "
        "Check the dataset before training."
    )

print("Loading base model...")
use_cuda = torch.cuda.is_available()

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    torch_dtype=torch.float16 if use_cuda else torch.float32,
)

model.config.use_cache = False

if use_cuda:
    model.gradient_checkpointing_enable()

model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    ),
)

model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=str(ROOT / "training" / "v3-checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=use_cuda,
    gradient_checkpointing=use_cuda,
    report_to="none",
    dataloader_pin_memory=use_cuda,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=AssistantOnlyCollator(),
    processing_class=tokenizer,
)

print("Starting v3 training...")
trainer.train()

print("Saving v3 adapter...")
model.save_pretrained(str(OUTPUT_PATH))
tokenizer.save_pretrained(str(OUTPUT_PATH))

print(f"Training complete. Adapter saved to: {OUTPUT_PATH}")
