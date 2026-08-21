"""LoRA supervised fine-tuning entry point for a CUDA GPU environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions(names: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "missing"
    return found


def to_prompt_completion(example: dict) -> dict:
    if "prompt" in example and "completion" in example:
        return {"prompt": example["prompt"], "completion": example["completion"]}
    messages = example["messages"]
    return {"prompt": messages[:-1], "completion": [messages[-1]]}


def token_length(processor, messages: list[dict]) -> int:
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(tokenized, dict):
        tokenized = tokenized["input_ids"]
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return len(tokenized)


def latest_checkpoint(output_path: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_path.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--train", default="corpus-v3/sft/train.jsonl")
    parser.add_argument("--validation", default="corpus-v3/sft/validation.jsonl")
    parser.add_argument("--output", default="training-runs/qwen35-08b-lora-v3")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Learning-rate warmup fraction; 0.0 reproduces the successful Colab run.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum retained checkpoints; omit to preserve every epoch checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
        help="Stop after this many evaluation rounds without improvement.",
    )
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Allow examples over max-length; disabled by default to protect labels.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Checkpoint path, or 'auto' to resume the highest checkpoint in output.",
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoProcessor, EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not detected. This script intentionally refuses impractical CPU "
            "fine-tuning; use a Colab/Kaggle/cloud GPU and keep local CPU for inference."
        )

    project_root = Path(__file__).resolve().parents[1]
    train_path = (project_root / args.train).resolve()
    validation_path = (project_root / args.validation).resolve()
    output_path = (project_root / args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    resume_checkpoint: Path | None = None
    if args.resume_from_checkpoint == "auto":
        resume_checkpoint = latest_checkpoint(output_path)
    elif args.resume_from_checkpoint:
        resume_checkpoint = Path(args.resume_from_checkpoint).expanduser().resolve()
        if not resume_checkpoint.is_dir():
            raise SystemExit(f"Resume checkpoint does not exist: {resume_checkpoint}")

    datasets = load_dataset(
        "json", data_files={"train": str(train_path), "validation": str(validation_path)}
    )
    processor = AutoProcessor.from_pretrained(args.model)
    normalized = {}
    for split in ("train", "validation"):
        normalized[split] = datasets[split].map(
            to_prompt_completion,
            remove_columns=datasets[split].column_names,
        )
    datasets = normalized

    length_report: dict[str, dict[str, int]] = {}
    overlength: list[str] = []
    for split in ("train", "validation"):
        lengths = [
            token_length(processor, [*row["prompt"], *row["completion"]])
            for row in datasets[split]
        ]
        length_report[split] = {
            "examples": len(lengths),
            "minimum": min(lengths),
            "maximum": max(lengths),
            "over_max_length": sum(length > args.max_length for length in lengths),
        }
        overlength.extend(
            f"{split}[{index}]={length}"
            for index, length in enumerate(lengths)
            if length > args.max_length
        )
    if overlength and not args.allow_truncation:
        preview = ", ".join(overlength[:10])
        raise SystemExit(
            f"Refusing to truncate {len(overlength)} SFT examples above "
            f"max_length={args.max_length}: {preview}"
        )

    use_bf16 = torch.cuda.is_bf16_supported()
    config = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        # Transformers v5 combines warmup steps and ratios in warmup_steps;
        # a float below 1 is interpreted as a fraction of total training steps.
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        model_init_kwargs={"dtype": torch.bfloat16 if use_bf16 else torch.float16},
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
        ],
    )

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": args.model,
        "method": "LoRA SFT",
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "packages": package_versions(
            ["torch", "transformers", "datasets", "accelerate", "peft", "trl"]
        ),
        "inputs": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "validation": {
                "path": str(validation_path),
                "sha256": sha256(validation_path),
            },
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "max_length": args.max_length,
            "effective_batch_size": 16,
            "lora_r": 16,
            "lora_alpha": 32,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": args.save_total_limit,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "early_stopping_patience": args.early_stopping_patience,
            "allow_truncation": args.allow_truncation,
            "completion_only_loss": True,
            "resume_from_checkpoint": (
                None if resume_checkpoint is None else str(resume_checkpoint)
            ),
        },
        "token_lengths": length_report,
    }
    (output_path / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        peft_config=lora,
        processing_class=processor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )
    trainer.train(
        resume_from_checkpoint=(
            None if resume_checkpoint is None else str(resume_checkpoint)
        )
    )
    metrics = trainer.evaluate()
    trainer.save_model(str(output_path / "best-adapter"))
    (output_path / "evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved LoRA adapter and evaluation to {output_path}")


if __name__ == "__main__":
    main()
