#!/usr/bin/env python3
"""LoRA fine-tune a small instruct model on exported chat JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TRAIN_FILE = ROOT / "data" / "finetune" / "train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "realpage-message-agent-v1"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_chat_dataset(path: Path) -> Dataset:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return Dataset.from_list(rows)


def smoke_test_generation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    example: dict,
    *,
    max_new_tokens: int = 512,
) -> str:
    messages = example["messages"][:2]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def save_merged_model(
    trainer: SFTTrainer,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    base_model: str,
) -> None:
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # transformers 5.x saves extra_special_tokens as a list; vLLM/TGI expect base format.
    from huggingface_hub import hf_hub_download

    base_tokenizer_config = hf_hub_download(base_model, "tokenizer_config.json")
    (output_dir / "tokenizer_config.json").write_bytes(
        Path(base_tokenizer_config).read_bytes()
    )
    metadata = {
        "base_model": base_model,
        "format": "merged-lora",
        "task": "realpage-message-agent-json",
    }
    (output_dir / "realpage_model.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune an instruct model on RealPage chat JSONL"
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="OpenAI/HF chat JSONL produced by export_finetune_dataset.py",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Base Hugging Face instruct model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for adapter or merged model artifacts",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge LoRA weights into the base model after training (default: true)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the base model in 4-bit for QLoRA (recommended on 6 GB GPUs)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on training rows (useful for smoke tests)",
    )
    parser.add_argument(
        "--smoke-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one on-GPU generation after training (default: true)",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required for training.", file=sys.stderr)
        return 1

    train_file = args.train_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_chat_dataset(train_file)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} training example(s) from {train_file}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    use_fp16 = not args.load_in_4bit

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="no",
        max_length=args.max_length,
        assistant_only_loss=True,
        bf16=False,
        fp16=use_fp16,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting LoRA training...")
    train_result = trainer.train()
    print(f"Training complete. Loss: {train_result.training_loss:.4f}")

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")

    deploy_dir = output_dir
    if args.merge:
        print("Merging LoRA adapter into base weights...")
        save_merged_model(trainer, tokenizer, deploy_dir, args.model)
        print(f"Saved merged model to {deploy_dir}")
    else:
        deploy_dir = adapter_dir
        print(f"Skipped merge; serve adapter from {adapter_dir}")

    if args.smoke_test:
        if args.merge:
            eval_model = trainer.model
        else:
            eval_model = trainer.model
        sample = dataset[0]
        generated = smoke_test_generation(eval_model, tokenizer, sample)
        print("\n--- Smoke test (first training example) ---")
        print(f"task_id: {sample.get('task_id', 'unknown')}")
        print("generated assistant JSON:")
        safe = generated[:2000].encode("ascii", errors="replace").decode("ascii")
        print(safe)
        if len(generated) > 2000:
            print("... [truncated]")

    print(
        "\nNext steps:\n"
        f"  1. Serve merged model: vllm serve {deploy_dir} --host 0.0.0.0 --port 8000\n"
        "  2. Set LLM_PROVIDER=local, LOCAL_BASE_URL=http://localhost:8000/v1,\n"
        "     LOCAL_MODEL=<served-model-name>\n"
        "  3. Evaluate: python run.py --input data/test_cases.jsonl --output outputs/lora_eval.jsonl"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
