"""LoRA SFT with immutable run inputs and verified, resumable epoch checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
COMPLETION_MARKER = "checkpoint-complete.json"
MANIFEST_NAME = "run-manifest.json"
LORA_SETTINGS = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    """An interrupted exclusive manifest write must block a fresh training attempt."""
    temporary = path if exclusive else path.with_name(path.name + ".tmp")
    with temporary.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or invalid JSON: {path}") from exc


def append_event(output_path: Path, event: str, **fields) -> None:
    with (output_path / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "at": datetime.now(UTC).isoformat(), "event": event, **fields,
        }, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def package_versions(names: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "missing"
    return found


def sanitize_urls(text: str) -> str:
    # Redact whole URLs, not just userinfo: tokens can also live in paths,
    # queries and fragments. Never persist an unsanitized pip freeze/remote URL.
    text = re.sub(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s]+", "[REDACTED_URL]", text)
    return re.sub(r"\b[^\s@]+@[^\s:]+:[^\s]+", "[REDACTED_URL]", text)


def pip_freeze() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError("pip freeze failed; refusing to train without environment provenance")
    return sanitize_urls(result.stdout)


def git_revision(path: Path) -> dict:
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
        return {"path": str(path), "revision": revision, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(path), "revision": None, "unavailable": True}


def pinned_revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("revision must be a full 40-hex Hugging Face commit")
    return value.lower()


def to_prompt_completion(example: dict) -> dict:
    if "prompt" in example and "completion" in example:
        prompt, completion = example["prompt"], example["completion"]
    else:
        messages = example["messages"]
        prompt, completion = messages[:-1], [messages[-1]]
    # TRL 1.9.2 SFTTrainer._prepare_dataset reads this per-example column for
    # BOTH prompt and full-conversation tokenization (not an SFTConfig option).
    return {
        "prompt": prompt,
        "completion": completion,
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
    }


def token_length(tokenizer, messages: list[dict], chat_template_kwargs=None) -> int:
    kwargs = dict(CHAT_TEMPLATE_KWARGS) if chat_template_kwargs is None else chat_template_kwargs
    if kwargs != CHAT_TEMPLATE_KWARGS:
        raise ValueError("Token counting and SFT must use enable_thinking=False")
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, truncation=False, **kwargs,
    )
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return len(tokenized)


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-([0-9]+)", path.name)
    if match is None:
        raise ValueError(f"Invalid checkpoint name: {path}")
    return int(match[1])


def checkpoint_files(path: Path, world_size: int, requires_scaler: bool) -> dict[str, Path]:
    """Check required standard Trainer/PEFT artifacts without importing torch."""
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Missing checkpoint directory: {path}")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Invalid checkpoint world_size")
    files = {}
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"Checkpoint contains a symlink: {item}")
        if item.is_file() and item.name not in (COMPLETION_MARKER, COMPLETION_MARKER + ".tmp"):
            files[item.relative_to(path).as_posix()] = item
    required = {"adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"}
    required.update(
        {"rng_state.pth"} if world_size == 1
        else {f"rng_state_{rank}.pth" for rank in range(world_size)}
    )
    if requires_scaler:
        required.add("scaler.pt")
    missing = required - files.keys()
    if not {"adapter_model.safetensors", "adapter_model.bin"}.intersection(files):
        missing.add("adapter_model.safetensors (or adapter_model.bin)")
    if missing:
        raise ValueError(f"Incomplete checkpoint {path}: missing {', '.join(sorted(missing))}")
    if any(item.stat().st_size == 0 for item in files.values()):
        raise ValueError(f"Empty checkpoint artifact in {path}")
    state = read_json(path / "trainer_state.json")
    if state.get("global_step") != checkpoint_step(path):
        raise ValueError(f"Trainer step does not match checkpoint directory: {path}")
    read_json(path / "adapter_config.json")
    return files


def complete_checkpoint(
    path: Path, run_fingerprint: str, *, world_size: int = 1, requires_scaler: bool = False,
) -> dict:
    files = checkpoint_files(path, world_size, requires_scaler)
    marker = {
        "schema_version": 1,
        "complete": True,
        "global_step": checkpoint_step(path),
        "run_fingerprint": run_fingerprint,
        "world_size": world_size,
        "requires_scaler": requires_scaler,
        "sha256": {name: sha256(item) for name, item in sorted(files.items())},
        "completed_at": datetime.now(UTC).isoformat(),
    }
    write_json(path / COMPLETION_MARKER, marker)
    return marker


def verify_checkpoint(path: Path, run_fingerprint: str | None = None) -> dict:
    marker = read_json(path / COMPLETION_MARKER)
    if marker.get("schema_version") != 1 or marker.get("complete") is not True:
        raise ValueError(f"Checkpoint has no completion attestation: {path}")
    if marker.get("global_step") != checkpoint_step(path):
        raise ValueError(f"Checkpoint marker step mismatch: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(marker.get("run_fingerprint", ""))):
        raise ValueError(f"Invalid checkpoint run fingerprint: {path}")
    if run_fingerprint is not None and marker["run_fingerprint"] != run_fingerprint:
        raise ValueError(f"Checkpoint belongs to different run inputs: {path}")
    if type(marker.get("requires_scaler")) is not bool:
        raise ValueError(f"Missing checkpoint scaler policy: {path}")
    files = checkpoint_files(path, marker.get("world_size"), marker["requires_scaler"])
    hashes = marker.get("sha256")
    if not isinstance(hashes, dict) or hashes.keys() != files.keys():
        raise ValueError(f"Checkpoint file inventory mismatch: {path}")
    for name, item in files.items():
        if hashes[name] != sha256(item):
            raise ValueError(f"Checkpoint SHA256 mismatch: {item}")
    return marker


def latest_checkpoint(output_path: Path, run_fingerprint: str | None = None) -> Path | None:
    checkpoints = []
    for path in output_path.glob("checkpoint-*"):
        try:
            verify_checkpoint(path, run_fingerprint)
            checkpoints.append((checkpoint_step(path), path))
        except (OSError, ValueError):
            continue
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def load_manifest(output_path: Path) -> dict:
    manifest = read_json(output_path / MANIFEST_NAME)
    immutable = manifest.get("immutable")
    if (manifest.get("schema_version") != 1 or not isinstance(immutable, dict)
            or manifest.get("run_fingerprint") != json_sha256(immutable)):
        raise ValueError("Missing or invalid immutable run manifest; use a new output directory")
    return manifest


def verify_manifest(saved: dict, current: dict) -> None:
    """Only immutable settings participate; attempt timestamps/paths do not."""
    if saved["immutable"] != current["immutable"]:
        changed = sorted(
            key for key in saved["immutable"].keys() | current["immutable"].keys()
            if saved["immutable"].get(key) != current["immutable"].get(key)
        )
        raise ValueError(f"Resume would change immutable run settings: {', '.join(changed)}")


def resolve_resume(output_path: Path, requested: str | None) -> Path | None:
    occupied = output_path.exists() and any(output_path.iterdir())
    if not occupied and requested in (None, "auto"):
        return None
    if not requested:
        raise ValueError("Output is not empty; explicitly resume a verified run or use a new directory")
    manifest = load_manifest(output_path)
    fingerprint = manifest["run_fingerprint"]
    latest = latest_checkpoint(output_path, fingerprint)
    if latest is None:
        raise ValueError("Existing run has no complete, hash-verified checkpoint; refusing to restart")
    chosen = latest if requested == "auto" else Path(requested).expanduser().resolve()
    if chosen.parent.resolve() != output_path.resolve():
        raise ValueError("Resume checkpoint must belong to the output run directory")
    verify_checkpoint(chosen, fingerprint)
    if checkpoint_step(chosen) < checkpoint_step(latest):
        raise ValueError("Resuming an older checkpoint would overwrite retained epochs; use the latest")
    state = read_json(chosen / "trainer_state.json")
    # Trainer may load the earlier loss-best checkpoint at the end of training.
    # Verify it now too, rather than accepting a corrupted model dependency.
    if state.get("best_model_checkpoint"):
        best = Path(state["best_model_checkpoint"])
        if best.parent.resolve() != output_path.resolve():
            raise ValueError("Stored loss-best checkpoint is outside this run directory")
        verify_checkpoint(best, fingerprint)
    return chosen


def training_settings(args, revision: str, dtype: str) -> dict:
    return {
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        # Transformers 5.15 interprets a float below 1 as a warmup fraction.
        "warmup_steps": args.warmup_ratio,
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "lr_scheduler_type": "cosine",
        "lr_scheduler_kwargs": {},
        "max_grad_norm": 1.0,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": None,
        "save_only_model": False,
        "logging_steps": 5,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "restore_callback_states_from_checkpoint": True,
        "ignore_data_skip": False,
        "seed": args.seed,
        "data_seed": args.seed,
        # Enforce the limit in preflight; never silently truncate in the trainer.
        "max_length": args.max_length if args.allow_truncation else None,
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "packing": False,
        "loss_type": "chunked_nll",
        "gradient_checkpointing": True,
        "bf16": dtype == "bfloat16",
        "fp16": dtype == "float16",
        "report_to": "none",
        "model_init_kwargs": {"revision": revision, "dtype": dtype},
    }


def build_manifest(args, revision: str, tokenizer, settings: dict, inputs: dict, runtime: dict) -> dict:
    template_hash = hashlib.sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest()
    immutable = {
        "base_model": args.model,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "chat_template_sha256": template_hash,
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
        "inputs": {split: item["sha256"] for split, item in inputs.items()},
        "max_length": args.max_length,
        "allow_truncation": args.allow_truncation,
        "early_stopping_patience": args.early_stopping_patience,
        "training": settings,
        "lora": LORA_SETTINGS,
        "world_size": runtime["world_size"],
        "effective_batch_size": runtime["effective_batch_size"],
        "packages": runtime["packages"],
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "run_fingerprint": json_sha256(immutable),
        "immutable": immutable,
        "inputs": inputs,
        "runtime": runtime,
        "adapter_selection": {
            "best-adapter": "loss-best only (minimum validation eval_loss)",
            "task_winner_selected": False,
            "behavioral_selection": "pending; compare all retained epoch checkpoints separately",
        },
    }


def make_callbacks(callback_base, early_stopping_class, patience: int, fingerprint: str, synchronize):
    class DurabilityCallback(callback_base):
        def on_save(self, args, state, control, **kwargs):
            # on_save follows Trainer's model/optimizer/scheduler/RNG/state saves.
            # All ranks must finish writing RNG files before rank zero hashes them.
            synchronize()
            if args.should_save:
                path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                complete_checkpoint(
                    path, fingerprint, world_size=args.world_size, requires_scaler=args.fp16,
                )
                append_event(Path(args.output_dir), "checkpoint_complete", checkpoint=str(path))
            synchronize()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if args.should_save:
                append_event(
                    Path(args.output_dir), "metrics", step=state.global_step,
                    epoch=state.epoch, metrics=logs or {},
                )
            return control

    callbacks = [DurabilityCallback()]
    if patience > 0:
        callbacks.append(early_stopping_class(early_stopping_patience=patience))
    return callbacks


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", type=pinned_revision, help="Optional full 40-hex HF commit; otherwise resolve once and pin it.")
    parser.add_argument("--train", default="corpus-v3/sft/train.jsonl")
    parser.add_argument("--validation", default="corpus-v3/sft/validation.jsonl")
    parser.add_argument("--output", default="training-runs/qwen35-08b-lora-v3")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--save-total-limit", type=int, default=None, help="Deprecated: only 0 (unlimited) is accepted; all epochs are retained.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--early-stopping-patience", type=int, default=1, help="Rounds without loss improvement; 0 disables early stopping.")
    parser.add_argument("--allow-truncation", action="store_true", help="Allow examples over max-length; disabled by default to protect labels.")
    parser.add_argument("--resume-from-checkpoint", help="Checkpoint path, or 'auto' for the latest hash-verified complete checkpoint.")
    args = parser.parse_args(argv)
    if args.save_total_limit not in (None, 0):
        parser.error("All epoch checkpoints must be retained; omit --save-total-limit")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be >= 0")
    if args.epochs < 1 or args.max_length < 1 or not args.learning_rate > 0:
        parser.error("epochs, max-length and learning-rate must be positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    train_path = (project_root / args.train).resolve()
    validation_path = (project_root / args.validation).resolve()
    output_path = (project_root / args.output).resolve()
    resume_checkpoint = resolve_resume(output_path, args.resume_from_checkpoint)
    saved = load_manifest(output_path) if resume_checkpoint else None

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not detected. This script intentionally refuses impractical CPU "
            "fine-tuning; use a Colab/Kaggle/cloud GPU and keep local CPU for inference."
        )

    requested_revision = args.revision or (saved["immutable"]["model_revision"] if saved else None)
    model_config = AutoConfig.from_pretrained(args.model, revision=requested_revision)
    revision = pinned_revision(getattr(model_config, "_commit_hash", "") or "")
    if requested_revision and revision != requested_revision:
        raise ValueError("Resolved model commit does not match the requested pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    settings = training_settings(args, revision, dtype)
    config = SFTConfig(output_dir=str(output_path), **settings)
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [{
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory": torch.cuda.get_device_properties(index).total_memory,
        } for index in range(torch.cuda.device_count())],
        "world_size": config.world_size,
        "effective_batch_size": config.train_batch_size * config.gradient_accumulation_steps * config.world_size,
        "packages": package_versions(["torch", "transformers", "datasets", "accelerate", "peft", "trl", "safetensors"]),
        "pip_freeze": pip_freeze(),
        "git": {"repo": git_revision(project_root), "fpy": git_revision(project_root.parent / "fpy")},
        "training_script_sha256": sha256(Path(__file__)),
    }
    inputs = {
        "train": {"path": str(train_path), "sha256": sha256(train_path)},
        "validation": {"path": str(validation_path), "sha256": sha256(validation_path)},
    }
    manifest = build_manifest(args, revision, tokenizer, settings, inputs, runtime)
    if saved:
        verify_manifest(saved, manifest)

    datasets = load_dataset(
        "json", data_files={"train": str(train_path), "validation": str(validation_path)},
    )
    datasets = {
        split: datasets[split].map(to_prompt_completion, remove_columns=datasets[split].column_names)
        for split in ("train", "validation")
    }
    length_report = {}
    overlength = []
    for split in ("train", "validation"):
        lengths = [
            token_length(tokenizer, [*row["prompt"], *row["completion"]], row["chat_template_kwargs"])
            for row in datasets[split]
        ]
        if not lengths:
            raise ValueError(f"Empty {split} dataset")
        length_report[split] = {
            "examples": len(lengths), "minimum": min(lengths), "maximum": max(lengths),
            "over_max_length": sum(length > args.max_length for length in lengths),
        }
        overlength.extend(
            f"{split}[{index}]={length}" for index, length in enumerate(lengths)
            if length > args.max_length
        )
    if overlength and not args.allow_truncation:
        raise ValueError(
            f"Refusing to truncate {len(overlength)} SFT examples above "
            f"max_length={args.max_length}: {', '.join(overlength[:10])}"
        )
    # Catch edits during dataset loading/preflight before publishing provenance.
    if sha256(train_path) != inputs["train"]["sha256"] or sha256(validation_path) != inputs["validation"]["sha256"]:
        raise ValueError("Training inputs changed during preflight")
    manifest["token_lengths"] = length_report
    if config.should_save:
        output_path.mkdir(parents=True, exist_ok=True)
        if not saved:
            # Recheck after preflight in case another attempt populated the directory.
            resolve_resume(output_path, None)
            write_json(output_path / MANIFEST_NAME, manifest, exclusive=True)
        append_event(
            output_path, "run_resumed" if saved else "run_started",
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
            runtime=runtime, run_fingerprint=manifest["run_fingerprint"],
        )
        # Keep failed writes for inspection, but never let Trainer overwrite them
        # (or accidentally attest leftover files) when replaying the next epoch.
        for path in output_path.glob("checkpoint-*"):
            try:
                verify_checkpoint(path, manifest["run_fingerprint"])
            except (OSError, ValueError):
                quarantine = path.with_name(f"incomplete-{path.name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}")
                path.rename(quarantine)
                append_event(output_path, "checkpoint_quarantined", source=str(path), destination=str(quarantine))

    trainer = SFTTrainer(
        model=args.model, args=config,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        peft_config=LoraConfig(**LORA_SETTINGS), processing_class=tokenizer,
    )
    # Use the Accelerator barrier without importing CUDA libraries in helpers/tests.
    for callback in make_callbacks(
        TrainerCallback, EarlyStoppingCallback, args.early_stopping_patience,
        manifest["run_fingerprint"], trainer.accelerator.wait_for_everyone,
    ):
        trainer.add_callback(callback)
    trainer.accelerator.wait_for_everyone()
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    metrics = trainer.evaluate()
    trainer.save_model(str(output_path / "best-adapter"))
    if config.should_save:
        selection = {**manifest["adapter_selection"], "loss_best_checkpoint": trainer.state.best_model_checkpoint}
        write_json(output_path / "best-adapter" / "selection.json", selection)
        write_json(output_path / "evaluation.json", {**metrics, "adapter_selection": selection})
        append_event(output_path, "run_finished", metrics=metrics, selection=selection)
        print(f"Saved loss-best ONLY adapter to {output_path}; behavioral selection remains pending.")


if __name__ == "__main__":
    main()
