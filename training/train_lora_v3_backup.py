import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

BASE = Path(r"D:\Project-Dostoevsky")
MODEL_PATH = BASE / "models" / "SmolLM2-360M-Instruct"
DATA_PATH = BASE / "training" / "personality_v3_combined.jsonl"
OUTPUT_PATH = BASE / "models" / "dostoevsky-lora-v3"
MAX_LENGTH = 512


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    records = []

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            item = json.loads(line)
            messages = item.get("messages")

            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Invalid messages on line {line_number}")

            # Tokenize the complete conversation to preserve chat boundaries.
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )
            input_ids = tokenized["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]

            # Find assistant response spans using the template's rendered text.
            # Refuse to train if reliable spans cannot be identified.
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            assistant_marker = "<|im_start|>assistant\n"
            spans = []
            cursor = 0

            while True:
                start = rendered.find(assistant_marker, cursor)
                if start == -1:
                    break
                content_start = start + len(assistant_marker)
                end = rendered.find("<|im_end|>", content_start)
                if end == -1:
                    raise ValueError(
                        f"Could not find assistant message ending on line {line_number}"
                    )
                spans.append((content_start, end))
                cursor = end + len("<|im_end|>")

            if not spans:
                raise ValueError(
                    f"No assistant responses found on line {line_number}"
                )

            # Tokenize prefixes to determine assistant token ranges.
            labels = [-100] * len(input_ids)
            previous_token_count = 0
            for start_char, end_char in spans:
                prefix_ids = tokenizer(
                    rendered[:start_char],
                    add_special_tokens=False,
                )["input_ids"]
                through_response_ids = tokenizer(
                    rendered[:end_char],
                    add_special_tokens=False,
                )["input_ids"]

                start_token = len(prefix_ids)
                end_token = len(through_response_ids)

                if end_token > len(input_ids):
                    raise ValueError(
                        f"Token alignment failed on line {line_number}. "
                        "Stop rather than train on incorrect labels."
                    )

                for i in range(start_token, end_token):
                    labels[i] = input_ids[i]

                previous_token_count = end_token

            # Truncate consistently from the left if the example is too long.
            input_ids = input_ids[-MAX_LENGTH:]
            labels = labels[-MAX_LENGTH:]

            if not any(x != -100 for x in labels):
                raise ValueError(
                    f"No assistant training tokens remain on line {line_number}"
                )

            records.append({"input_ids": input_ids, "labels": labels})

    if not records:
        raise ValueError("Dataset is empty.")

    print("Examples loaded:", len(records))
    dataset = Dataset.from_list(records)

    class AssistantOnlyCollator:
        def __call__(self, features):
            max_len = max(len(x["input_ids"]) for x in features)
            ids_batch, masks_batch, labels_batch = [], [], []

            for feature in features:
                ids = feature["input_ids"]
                labels = feature["labels"]
                padding = max_len - len(ids)

                ids_batch.append(ids + [tokenizer.pad_token_id] * padding)
                masks_batch.append([1] * len(ids) + [0] * padding)
                labels_batch.append(labels + [-100] * padding)

            return {
                "input_ids": torch.tensor(ids_batch, dtype=torch.long),
                "attention_mask": torch.tensor(masks_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long),
            }

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

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
        output_dir=str(OUTPUT_PATH / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=AssistantOnlyCollator(),
    )

    print("Starting v3 training...")
    trainer.train()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    print("Training complete!")
    print("Adapter saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()