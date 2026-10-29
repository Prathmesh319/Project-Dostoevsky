
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "SmolLM2-360M-Instruct"
ADAPTER = ROOT / "models" / "dostoevsky-lora-v4"

print("Loading Dostoevsky v4... Please wait.")

tokenizer = AutoTokenizer.from_pretrained(
    str(ADAPTER),
    local_files_only=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    str(BASE),
    local_files_only=True,
    torch_dtype=torch.float32
)

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER),
    local_files_only=True
)

model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

system_prompt = (
    "You are a philosophical AI inspired by the psychological depth, "
    "moral ambiguity, and existential themes of Fyodor Dostoevsky's "
    "literature. You are not literally the historical author. Think deeply "
    "about guilt, freedom, suffering, pride, faith, contradiction, and "
    "human self-deception. Give nuanced, original interpretations instead "
    "of generic motivational advice. Explain your reasoning with "
    "psychological insight. Do not falsely attribute invented quotations "
    "or opinions to Dostoevsky."
)

print("\nDostoevsky v4 is ready. Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() in {"exit", "quit"}:
        break

    if not question:
        continue

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    print(f"\nDostoevsky: {answer}\n")
