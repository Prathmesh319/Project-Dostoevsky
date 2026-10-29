
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = Path(r"D:\Project-Dostoevsky")
MODEL_PATH = BASE / "models" / "SmolLM2-360M-Instruct"
ADAPTER_PATH = BASE / "models" / "dostoevsky-lora-v2"


def main():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_PATH),
        local_files_only=True,
    )

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=(
            torch.float16 if torch.cuda.is_available()
            else torch.float32
        ),
        device_map="auto",
    )

    print("Loading Dostoevsky adapter...")

    model = PeftModel.from_pretrained(
        model,
        str(ADAPTER_PATH),
        local_files_only=True,
    )

    model.eval()
    print("\nDostoevsky is ready. Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"exit", "quit"}:
            break

        if not user_text:
            continue

        messages = [{"role": "user", "content": user_text}]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        print(f"\nDostoevsky: {answer.strip()}\n")


if __name__ == "__main__":
    main()
