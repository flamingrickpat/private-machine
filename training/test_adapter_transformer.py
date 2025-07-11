import argparse, json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--model",   help="base checkpoint dir or HF repo", default="/mnt/c/workspace/AI/gemma-3-4b-it")
    cli.add_argument("--adapter", help="path to LoRA (e.g. adapters/mem_0)", default="/mnt/c/workspace/private-machine/test/adapters/mem_0")
    cli.add_argument("--load-4bit", action="store_true", help="use the same 4-bit QLoRA weights", default=True)
    args = cli.parse_args()

    # --- tokenizer ---------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.pad_token = tok.eos_token                    # Gemma has no pad token

    # --- base model --------------------------------------------------------
    bnb_cfg = None
    if args.load_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_cfg,
    )

    # --- load adapter ------------------------------------------------------
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    print(f"✓ LoRA '{Path(args.adapter).name}' loaded")

    # --- chat loop ---------------------------------------------------------
    history = []
    print("Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": user})

        prompt = tok.apply_chat_template(
            history, add_generation_prompt=True, tokenize=False
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
            )
        reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Assistant: {reply}\n")
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
