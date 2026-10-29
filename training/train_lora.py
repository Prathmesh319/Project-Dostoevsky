
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
DATA_PATH = BASE / "training" / "personality_v2_combined.jsonl"
OUTPUT_PATH = BASE / "models" / "dostoevsky-lora-v2"
MAX_LENGTH = 512


def prepare_example(messages, tokenizer):
    # Render the entire conversation as text first.
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Find character ranges belonging to assistant turns.
    assistant_spans = []

    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        prefix = tokenizer.apply_chat_template(
            messages[:i],
            tokenize=False,
            add_generation_prompt=True,
        )

        through_answer = tokenizer.apply_chat_template(
            messages[:i + 1],
            tokenize=False,
            add_generation_prompt=False,
        )

        if not through_answer.startswith(prefix):
            raise ValueError(
                "Chat template prefix mismatch. "
                "Stopping rather than training with incorrect labels."
            )

        start = len(prefix)
        end = len(through_answer)

        if start < end:
            assistant_spans.append((start, end))

    if not assistant_spans:
        raise ValueError("No assistant response spans found.")

    # Fast-tokenizer offsets map tokens back to character positions.
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_offsets_mapping=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]

    labels = []

    for (start, end), token_id, mask in zip(
        offsets, input_ids, attention_mask
    ):
        # Ignore padding and tokens outside assistant responses.
        if mask and any(
            start < span_end and end > span_start
            for span_start, span_end in assistant_spans
        ):
            labels.append(token_id)
        else:
            labels.append(-100)

    if all(label == -100 for label in labels):
        raise ValueError(
            "No assistant tokens found after tokenization. "
            "Check the tokenizer and chat template."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for offset mapping."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading combined dataset...")
    examples = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            messages = item.get("messages")

            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Invalid messages at line {line_number}"
                )

            examples.append(prepare_example(messages, tokenizer))

    if not examples:
        raise RuntimeError("Dataset is empty.")

    assistant_tokens = sum(
        sum(label != -100 for label in example["labels"])
        for example in examples
    )

    print("Examples:", len(examples))
    print("Assistant training tokens:", assistant_tokens)

    dataset = Dataset.from_list(examples)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Preserve our precomputed -100 labels.
    def collate_fn(features):
        return {
            key: torch.tensor(
                [feature[key] for feature in features],
                dtype=torch.long,
            )
            for key in ("input_ids", "attention_mask", "labels")
        }

    args = TrainingArguments(
        output_dir=str(OUTPUT_PATH / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=2,
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
        data_collator=collate_fn,
    )

    print("Starting LoRA training...")
    trainer.train()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    print("Training complete!")
    print("New adapter saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
