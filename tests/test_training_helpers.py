import argparse
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections import UserDict
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import training.train_lora as training
from training.train_lora import (
    CHAT_TEMPLATE_KWARGS,
    COMPLETION_MARKER,
    LORA_SETTINGS,
    MANIFEST_NAME,
    append_event,
    build_manifest,
    complete_checkpoint,
    json_sha256,
    latest_checkpoint,
    load_manifest,
    make_callbacks,
    parse_args,
    pinned_revision,
    pip_freeze,
    resolve_resume,
    sanitize_urls,
    sha256,
    to_prompt_completion,
    token_length,
    training_settings,
    verify_checkpoint,
    verify_manifest,
    write_json,
)


REVISION = "a" * 40
FINGERPRINT = "b" * 64


def make_checkpoint(root, step, fingerprint=FINGERPRINT, *, complete=True, world_size=1, scaler=False):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "training_args.bin"):
        (path / name).write_bytes(f"{name}-{step}".encode())
    rng_names = ["rng_state.pth"] if world_size == 1 else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    for name in rng_names + (["scaler.pt"] if scaler else []):
        (path / name).write_bytes(b"state")
    write_json(path / "adapter_config.json", {"r": 16})
    write_json(path / "trainer_state.json", {"global_step": step, "epoch": 1, "best_model_checkpoint": None})
    if complete:
        complete_checkpoint(path, fingerprint, world_size=world_size, requires_scaler=scaler)
    return path


def manifest_fixture(argv=None):
    args = parse_args(argv or [])
    tokenizer = Mock()
    tokenizer.get_chat_template.return_value = "template with enable_thinking"
    runtime = {"world_size": 1, "effective_batch_size": 16, "packages": {"trl": "1.9.2"}}
    inputs = {
        "train": {"path": "original/train.jsonl", "sha256": "c" * 64},
        "validation": {"path": "original/val.jsonl", "sha256": "d" * 64},
    }
    return build_manifest(args, REVISION, tokenizer, training_settings(args, REVISION, "bfloat16"), inputs, runtime)


class TrainingHelperTests(unittest.TestCase):
    def test_latest_checkpoint_uses_highest_numeric_complete_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_checkpoint(root, 17)
            newest = make_checkpoint(root, 68)
            make_checkpoint(root, 100, complete=False)
            (root / "checkpoint-invalid").mkdir()
            (root / "checkpoint-200").write_text("not a directory")
            self.assertEqual(latest_checkpoint(root), newest)

    def test_latest_checkpoint_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(latest_checkpoint(Path(temporary)))

    def test_unmarked_checkpoint_is_not_resumable_even_when_files_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_checkpoint(root, 1, complete=False)
            with self.assertRaisesRegex(ValueError, "Missing or invalid JSON"):
                verify_checkpoint(path)
            self.assertIsNone(latest_checkpoint(root))

    def test_marker_hashes_every_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_checkpoint(Path(temporary), 17, complete=False)
            (path / "tokenizer.json").write_text("tokenizer")
            marker = complete_checkpoint(path, FINGERPRINT)
            self.assertTrue(marker["complete"])
            self.assertEqual(marker["global_step"], 17)
            self.assertEqual(set(marker["sha256"]), {p.name for p in path.iterdir()} - {COMPLETION_MARKER})
            for name, digest in marker["sha256"].items():
                self.assertEqual(digest, sha256(path / name))
            self.assertEqual(verify_checkpoint(path, FINGERPRINT), marker)

    def test_missing_or_corrupt_required_files_reject_checkpoint(self):
        required = (
            "adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
            "trainer_state.json", "training_args.bin", "rng_state.pth",
        )
        for name in required:
            for damage in ("delete", "empty", "change"):
                with self.subTest(name=name, damage=damage), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    previous = make_checkpoint(root, 1)
                    newest = make_checkpoint(root, 2)
                    if damage == "delete":
                        (newest / name).unlink()
                    else:
                        (newest / name).write_bytes(b"" if damage == "empty" else b"corrupted")
                    with self.assertRaises(ValueError):
                        verify_checkpoint(newest)
                    self.assertEqual(latest_checkpoint(root), previous)

    def test_missing_artifact_never_gets_completion_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_checkpoint(Path(temporary), 1, complete=False)
            (path / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "optimizer.pt"):
                complete_checkpoint(path, FINGERPRINT)
            self.assertFalse((path / COMPLETION_MARKER).exists())

    def test_malformed_false_or_incomplete_marker_is_rejected(self):
        mutations = [
            lambda marker: marker.update(complete=False),
            lambda marker: marker.update(global_step=10),
            lambda marker: marker.update(schema_version=99),
            lambda marker: marker.update(sha256={}),
            lambda marker: marker.update(world_size=None),
            lambda marker: marker.pop("requires_scaler"),
            lambda marker: marker.update(run_fingerprint=""),
            lambda marker: marker["sha256"].update({"../outside": "f" * 64}),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = make_checkpoint(root, 1)
                marker = verify_checkpoint(path)
                mutate(marker)
                write_json(path / COMPLETION_MARKER, marker)
                self.assertIsNone(latest_checkpoint(root))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_checkpoint(root, 1)
            (path / COMPLETION_MARKER).write_text("{interrupted")
            self.assertIsNone(latest_checkpoint(root))

    def test_changed_inventory_rejected_even_if_required_files_still_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_checkpoint(Path(temporary), 1)
            (path / "unexpected.pt").write_bytes(b"leftover")
            with self.assertRaisesRegex(ValueError, "inventory"):
                verify_checkpoint(path)

    def test_trainer_step_must_match_directory_before_marking_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_checkpoint(Path(temporary), 1, complete=False)
            write_json(path / "trainer_state.json", {"global_step": 2})
            with self.assertRaisesRegex(ValueError, "Trainer step"):
                complete_checkpoint(path, FINGERPRINT)

    def test_fp16_requires_scaler_and_distributed_rng_requires_every_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_checkpoint(Path(temporary), 1, scaler=True, world_size=2)
            marker = verify_checkpoint(path)
            self.assertTrue(marker["requires_scaler"])
            for name in ("rng_state_0.pth", "rng_state_1.pth", "scaler.pt"):
                content = (path / name).read_bytes()
                (path / name).unlink()
                with self.assertRaisesRegex(ValueError, "missing"):
                    verify_checkpoint(path)
                (path / name).write_bytes(content)
            verify_checkpoint(path)

    def test_binary_adapter_format_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_checkpoint(Path(temporary), 1, complete=False)
            (path / "adapter_model.safetensors").rename(path / "adapter_model.bin")
            complete_checkpoint(path, FINGERPRINT)
            verify_checkpoint(path)

    def test_checkpoint_from_other_run_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_checkpoint(root, 1)
            with self.assertRaisesRegex(ValueError, "different run"):
                verify_checkpoint(path, "e" * 64)
            self.assertIsNone(latest_checkpoint(root, "e" * 64))

    def test_auto_starts_only_in_empty_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(resolve_resume(root, "auto"))
            self.assertIsNone(resolve_resume(root / "new", None))
            (root / "unrelated.txt").write_text("must not overwrite")
            for requested in (None, "auto"):
                with self.subTest(requested=requested), self.assertRaises(ValueError):
                    resolve_resume(root, requested)

    def test_existing_incomplete_run_fails_without_overwriting_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = manifest_fixture()
            write_json(root / MANIFEST_NAME, manifest)
            original = (root / MANIFEST_NAME).read_bytes()
            make_checkpoint(root, 10, manifest["run_fingerprint"], complete=False)
            for requested in (None, "auto", str(root / "checkpoint-10")):
                with self.subTest(requested=requested), self.assertRaises(ValueError):
                    resolve_resume(root, requested)
            self.assertEqual((root / MANIFEST_NAME).read_bytes(), original)

    def test_resume_requires_manifest_and_rejects_tampered_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_checkpoint(root, 1)
            with self.assertRaises(ValueError):
                resolve_resume(root, "auto")
            manifest = manifest_fixture()
            manifest["immutable"]["model_revision"] = "f" * 40
            write_json(root / MANIFEST_NAME, manifest)
            with self.assertRaisesRegex(ValueError, "immutable run manifest"):
                load_manifest(root)

    def test_resume_selects_verified_latest_and_refuses_older_or_external_paths(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as other:
            root = Path(temporary)
            manifest = manifest_fixture()
            fingerprint = manifest["run_fingerprint"]
            write_json(root / MANIFEST_NAME, manifest)
            old = make_checkpoint(root, 1, fingerprint)
            latest = make_checkpoint(root, 2, fingerprint)
            partial = make_checkpoint(root, 3, fingerprint, complete=False)
            external = make_checkpoint(Path(other), 4, fingerprint)
            self.assertEqual(resolve_resume(root, "auto"), latest)
            self.assertEqual(resolve_resume(root, str(latest)), latest.resolve())
            for path in (old, partial, external, root / "checkpoint-999"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    resolve_resume(root, str(path))

    def test_loss_best_dependency_is_verified_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = manifest_fixture()
            fingerprint = manifest["run_fingerprint"]
            write_json(root / MANIFEST_NAME, manifest)
            best = make_checkpoint(root, 1, fingerprint)
            latest = make_checkpoint(root, 2, fingerprint, complete=False)
            write_json(latest / "trainer_state.json", {"global_step": 2, "best_model_checkpoint": str(best)})
            complete_checkpoint(latest, fingerprint)
            self.assertEqual(resolve_resume(root, "auto"), latest)
            (best / "adapter_model.safetensors").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                resolve_resume(root, "auto")

    def test_manifest_records_revision_template_optimizer_and_loss_only_selection(self):
        manifest = manifest_fixture(["--epochs", "2", "--early-stopping-patience", "0"])
        immutable = manifest["immutable"]
        self.assertEqual(immutable["model_revision"], REVISION)
        self.assertEqual(immutable["tokenizer_revision"], REVISION)
        self.assertEqual(immutable["chat_template_sha256"], hashlib.sha256(b"template with enable_thinking").hexdigest())
        self.assertEqual(immutable["chat_template_kwargs"], CHAT_TEMPLATE_KWARGS)
        self.assertEqual(immutable["lora"], LORA_SETTINGS)
        self.assertEqual(immutable["training"]["optim"], "adamw_torch")
        self.assertEqual(immutable["training"]["lr_scheduler_type"], "cosine")
        self.assertEqual(immutable["training"]["max_grad_norm"], 1.0)
        self.assertEqual(immutable["training"]["model_init_kwargs"], {"revision": REVISION, "dtype": "bfloat16"})
        self.assertEqual(immutable["training"]["num_train_epochs"], 2)
        self.assertEqual(immutable["early_stopping_patience"], 0)
        self.assertFalse(manifest["adapter_selection"]["task_winner_selected"])
        self.assertIn("loss-best only", manifest["adapter_selection"]["best-adapter"])
        self.assertEqual(manifest["run_fingerprint"], json_sha256(immutable))

    def test_resume_equivalence_ignores_attempt_paths_timestamps_and_provenance(self):
        saved = manifest_fixture()
        current = manifest_fixture(["--output", "new-output", "--resume-from-checkpoint", "some/checkpoint-1"])
        current["created_at"] = "later"
        current["inputs"]["train"]["path"] = "relocated/train.jsonl"
        current["runtime"]["git"] = {"repo": {"revision": "new revision"}}
        current["resume_from_checkpoint"] = "some/checkpoint-1"
        verify_manifest(saved, current)
        self.assertEqual(saved["run_fingerprint"], current["run_fingerprint"])

    def test_changed_cli_training_settings_are_immutable(self):
        changes = [
            ["--epochs", "2"], ["--learning-rate", "0.01"], ["--warmup-ratio", "0.1"],
            ["--seed", "43"], ["--max-length", "4096"], ["--early-stopping-patience", "0"],
            ["--allow-truncation"], ["--model", "different/model"],
        ]
        saved = manifest_fixture()
        for argv in changes:
            with self.subTest(argv=argv), self.assertRaisesRegex(ValueError, "immutable"):
                verify_manifest(saved, manifest_fixture(argv))

    def test_every_immutable_field_is_checked_including_data_and_optimizer(self):
        saved = manifest_fixture()
        mutations = [
            ("model_revision",), ("tokenizer_revision",), ("chat_template_sha256",),
            ("chat_template_kwargs", "enable_thinking"),
            ("inputs", "train"), ("inputs", "validation"),
            ("lora", "r"), ("lora", "lora_alpha"), ("lora", "lora_dropout"), ("lora", "target_modules"),
            ("world_size",), ("effective_batch_size",), ("packages", "trl"),
        ] + [("training", key) for key in saved["immutable"]["training"]]
        for keys in mutations:
            with self.subTest(keys=keys):
                current = copy.deepcopy(saved)
                target = current["immutable"]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = "changed"
                with self.assertRaisesRegex(ValueError, "immutable"):
                    verify_manifest(saved, current)

    def test_prompt_completion_rows_use_supported_trl_chat_kwargs_without_metadata(self):
        prompt = [{"role": "user", "content": "input"}]
        completion = [{"role": "assistant", "content": "output"}]
        result = to_prompt_completion({
            "prompt": prompt, "completion": completion, "metadata": {"task": "extract"},
            "chat_template_kwargs": {"enable_thinking": True},
        })
        self.assertEqual(result, {
            "prompt": prompt, "completion": completion, "chat_template_kwargs": {"enable_thinking": False},
        })
        result["chat_template_kwargs"]["enable_thinking"] = True
        self.assertEqual(CHAT_TEMPLATE_KWARGS, {"enable_thinking": False})

    def test_legacy_messages_are_still_supported(self):
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "input"},
            {"role": "assistant", "content": "output"},
        ]
        self.assertEqual(to_prompt_completion({"messages": messages}), {
            "prompt": messages[:-1], "completion": [messages[-1]],
            "chat_template_kwargs": {"enable_thinking": False},
        })

    def test_length_preflight_matches_training_kwargs_without_truncating(self):
        row = to_prompt_completion({"messages": [
            {"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"},
        ]})
        messages = row["prompt"] + row["completion"]
        for tokenized in ([1, 2, 3], {"input_ids": [1, 2, 3]}, UserDict(input_ids=[[1, 2, 3]])):
            with self.subTest(tokenized=tokenized):
                tokenizer = Mock()
                tokenizer.apply_chat_template.return_value = tokenized
                self.assertEqual(token_length(tokenizer, messages, row["chat_template_kwargs"]), 3)
                tokenizer.apply_chat_template.assert_called_once_with(
                    messages, tokenize=True, add_generation_prompt=False, truncation=False,
                    **row["chat_template_kwargs"],
                )
        with self.assertRaisesRegex(ValueError, "enable_thinking=False"):
            token_length(Mock(), messages, {"enable_thinking": True})

    def test_training_defaults_preserve_patience_and_disable_truncation_and_pruning(self):
        args = parse_args([])
        settings = training_settings(args, REVISION, "bfloat16")
        self.assertEqual(args.early_stopping_patience, 1)
        self.assertFalse(args.allow_truncation)
        self.assertIsNone(settings["max_length"])
        self.assertIsNone(settings["save_total_limit"])
        self.assertTrue(settings["restore_callback_states_from_checkpoint"])
        self.assertNotIn("chat_template_kwargs", settings)  # Not supported by SFTConfig 1.9.2.
        self.assertEqual(training_settings(parse_args(["--allow-truncation"]), REVISION, "float16")["max_length"], 3072)
        self.assertTrue(training_settings(args, REVISION, "float16")["fp16"])

    def test_pinned_revision_requires_full_hf_commit(self):
        self.assertEqual(pinned_revision("A" * 40), REVISION)
        for revision in ("main", "v1", "abc1234", "g" * 40, "a" * 41, "a" * 39):
            with self.subTest(revision=revision), self.assertRaises(argparse.ArgumentTypeError):
                pinned_revision(revision)
        self.assertEqual(parse_args(["--revision", REVISION]).revision, REVISION)

    def test_invalid_patience_pruning_or_warmup_rejected_before_training_imports(self):
        for argv in (["--early-stopping-patience", "-1"], ["--save-total-limit", "1"], ["--warmup-ratio", "1"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_zero_patience_disables_early_stopping_but_not_checkpoint_completion(self):
        early_stopping = Mock()
        callbacks = make_callbacks(object, early_stopping, 0, FINGERPRINT, Mock())
        self.assertEqual(len(callbacks), 1)
        early_stopping.assert_not_called()
        enabled = make_callbacks(object, early_stopping, 1, FINGERPRINT, Mock())
        self.assertEqual(len(enabled), 2)
        early_stopping.assert_called_once_with(early_stopping_patience=1)

    def test_save_callback_marks_after_all_rank_writes_and_appends_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_checkpoint(root, 1, complete=False)
            synchronize = Mock()
            callback = make_callbacks(object, Mock(), 0, FINGERPRINT, synchronize)[0]
            args = SimpleNamespace(output_dir=str(root), should_save=True, world_size=1, fp16=False)
            state = SimpleNamespace(global_step=1, epoch=1.0)
            control = object()
            append_event(root, "run_started")
            callback.on_log(args, state, control, logs={"loss": 1.0})
            self.assertFalse((path / COMPLETION_MARKER).exists())
            self.assertIs(callback.on_save(args, state, control), control)
            verify_checkpoint(path, FINGERPRINT)
            self.assertEqual(synchronize.call_count, 2)
            append_event(root, "run_resumed")
            callback.on_log(args, state, control, logs={"loss": 0.5})
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], [
                "run_started", "metrics", "checkpoint_complete", "run_resumed", "metrics",
            ])
            self.assertEqual(events[1]["metrics"], {"loss": 1.0})

    def test_non_writer_callback_does_not_mark_or_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synchronize = Mock()
            callback = make_callbacks(object, Mock(), 0, FINGERPRINT, synchronize)[0]
            args = SimpleNamespace(output_dir=str(root), should_save=False)
            callback.on_save(args, SimpleNamespace(global_step=1), None)
            callback.on_log(args, SimpleNamespace(global_step=1), None)
            self.assertEqual(synchronize.call_count, 2)
            self.assertEqual(list(root.iterdir()), [])

    def test_failed_marker_publish_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_checkpoint(root, 1, complete=False)
            with patch.object(Path, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    complete_checkpoint(path, FINGERPRINT)
            self.assertFalse((path / COMPLETION_MARKER).exists())
            self.assertIsNone(latest_checkpoint(root))

    def test_immutable_manifest_creation_never_clobbers_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / MANIFEST_NAME
            write_json(path, manifest_fixture(), exclusive=True)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json(path, {"different": "run"}, exclusive=True)
            self.assertEqual(path.read_bytes(), original)

    def test_pip_freeze_redacts_urls_including_query_fragment_and_path_tokens(self):
        raw = "\n".join([
            "torch==2.9.0",
            "adapter @ git+https://user:password@example.test/repo@commit#egg=adapter",
            "private @ https://example.test/secret-path/wheel.whl?token=secret#auth=another",
            "-e git+ssh://git:credential@example.test/project",
            "-e git@example.test:repo.git",
        ])
        sanitized = sanitize_urls(raw)
        self.assertIn("torch==2.9.0", sanitized)
        for secret in ("password", "secret", "another", "credential", "example.test"):
            self.assertNotIn(secret, sanitized)
        with patch("training.train_lora.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=raw)) as run:
            self.assertEqual(pip_freeze(), sanitized)
            self.assertEqual(run.call_args.args[0], [sys.executable, "-m", "pip", "freeze", "--all"])
        with patch("training.train_lora.subprocess.run", return_value=SimpleNamespace(returncode=1)):
            with self.assertRaises(RuntimeError):
                pip_freeze()

    def test_helpers_import_without_cuda_or_training_dependencies(self):
        project = Path(__file__).resolve().parents[1]
        code = (
            "import sys; import training.train_lora; "
            "assert not {'torch', 'transformers', 'trl', 'peft', 'datasets'}.intersection(sys.modules)"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=project, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class TrainingEntrypointTests(unittest.TestCase):
    """Exercise fresh/resumed orchestration without installing CUDA libraries."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.output = self.root / "run"
        self.train_file = self.root / "train.jsonl"
        self.validation_file = self.root / "validation.jsonl"
        row = {"messages": [
            {"role": "user", "content": "input"}, {"role": "assistant", "content": "output"},
        ]}
        for path in (self.train_file, self.validation_file):
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        class Dataset(list):
            column_names = ["messages"]

            def map(self, function, remove_columns):
                return Dataset(function(row) for row in self)

        self.tokenizer = Mock()
        self.tokenizer.get_chat_template.return_value = "fixed template"
        self.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        self.auto_tokenizer = Mock()
        self.auto_tokenizer.from_pretrained.return_value = self.tokenizer
        self.auto_config = Mock()
        self.auto_config.from_pretrained.side_effect = lambda model, revision: SimpleNamespace(_commit_hash=revision or REVISION)
        self.trainers = []
        trainers = self.trainers

        class Trainer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.args = kwargs["args"]
                self.callbacks = []
                self.accelerator = SimpleNamespace(wait_for_everyone=Mock())
                self.state = SimpleNamespace(global_step=0, epoch=0, best_model_checkpoint=None)
                trainers.append(self)

            def add_callback(self, callback):
                self.callbacks.append(callback)

            def train(self, resume_from_checkpoint):
                start = int(Path(resume_from_checkpoint).name.split("-")[-1]) if resume_from_checkpoint else 0
                self.resume = resume_from_checkpoint
                self.state.global_step = start
                self.state.best_model_checkpoint = str(Path(self.args.output_dir) / "checkpoint-1")
                for callback in self.callbacks:
                    callback.on_log(self.args, self.state, None, logs={"loss": 0.5})
                for step in range(start + 1, int(self.args.num_train_epochs) + 1):
                    path = make_checkpoint(Path(self.args.output_dir), step, complete=False)
                    self.state.global_step = step
                    self.state.epoch = float(step)
                    write_json(path / "trainer_state.json", vars(self.state))
                    for callback in self.callbacks:
                        callback.on_save(self.args, self.state, None)

            def evaluate(self):
                return {"eval_loss": 0.5}

            def save_model(self, path):
                Path(path).mkdir(exist_ok=True)

        def sft_config(**kwargs):
            return SimpleNamespace(**kwargs, world_size=1, should_save=True, train_batch_size=1)

        cuda = SimpleNamespace(
            is_available=lambda: True, is_bf16_supported=lambda: True,
            get_device_name=lambda index: "fake GPU", get_device_capability=lambda index: (8, 0),
            get_device_properties=lambda index: SimpleNamespace(total_memory=16_000), device_count=lambda: 1,
        )
        modules = {
            "torch": SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="test"), backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 1))),
            "datasets": SimpleNamespace(load_dataset=lambda *args, **kwargs: {"train": Dataset([row]), "validation": Dataset([row])}),
            "peft": SimpleNamespace(LoraConfig=lambda **kwargs: kwargs),
            "transformers": SimpleNamespace(AutoConfig=self.auto_config, AutoTokenizer=self.auto_tokenizer, TrainerCallback=object, EarlyStoppingCallback=Mock()),
            "trl": SimpleNamespace(SFTConfig=sft_config, SFTTrainer=Trainer),
        }
        self.stack.enter_context(patch.dict(sys.modules, modules))
        self.stack.enter_context(patch.object(training, "pip_freeze", return_value="trl==1.9.2\n"))
        self.stack.enter_context(patch.object(training, "package_versions", return_value={"trl": "1.9.2"}))
        self.stack.enter_context(patch.object(training, "git_revision", return_value={"revision": "e" * 40}))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def run_main(self, extra=()):
        argv = [
            "train_lora.py", "--train", str(self.train_file), "--validation", str(self.validation_file),
            "--output", str(self.output), "--epochs", "2", "--early-stopping-patience", "0", *extra,
        ]
        with patch.object(sys, "argv", argv):
            training.main()

    def test_fresh_and_auto_resume_pin_tokenizer_and_model_preserve_manifest_and_logs(self):
        self.run_main()
        original = (self.output / MANIFEST_NAME).read_bytes()
        before_logs = (self.output / "events.jsonl").read_bytes()
        saved = load_manifest(self.output)
        self.assertEqual(saved["runtime"]["pip_freeze"], "trl==1.9.2\n")
        self.assertEqual(set(saved["runtime"]["git"]), {"repo", "fpy"})
        self.assertEqual(saved["runtime"]["devices"][0]["name"], "fake GPU")
        trainer = self.trainers[0]
        self.assertIs(trainer.kwargs["processing_class"], self.tokenizer)
        for split in ("train_dataset", "eval_dataset"):
            self.assertEqual(trainer.kwargs[split][0]["chat_template_kwargs"], CHAT_TEMPLATE_KWARGS)
        self.assertEqual(trainer.args.model_init_kwargs["revision"], REVISION)
        self.assertEqual(len(trainer.callbacks), 1)
        self.auto_tokenizer.from_pretrained.assert_called_once_with("Qwen/Qwen3.5-0.8B", revision=REVISION)
        for step in (1, 2):
            verify_checkpoint(self.output / f"checkpoint-{step}", saved["run_fingerprint"])
        partial = make_checkpoint(self.output, 3, complete=False)
        self.run_main(["--resume-from-checkpoint", "auto"])
        self.assertEqual(self.trainers[-1].resume, str(self.output / "checkpoint-2"))
        self.auto_config.from_pretrained.assert_called_with("Qwen/Qwen3.5-0.8B", revision=REVISION)
        self.assertEqual((self.output / MANIFEST_NAME).read_bytes(), original)
        self.assertTrue((self.output / "events.jsonl").read_bytes().startswith(before_logs))
        self.assertFalse(partial.exists())
        self.assertEqual(len(list(self.output.glob("incomplete-checkpoint-3-*"))), 1)
        selection = json.loads((self.output / "best-adapter" / "selection.json").read_text())
        self.assertFalse(selection["task_winner_selected"])
        self.assertIn("loss-best only", selection["best-adapter"])

    def test_changed_config_and_data_fail_before_resume_writes(self):
        self.run_main()
        before = {path.relative_to(self.output): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.run_main(["--resume-from-checkpoint", "auto", "--learning-rate", "0.1"])
        self.train_file.write_text("changed data", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.run_main(["--resume-from-checkpoint", "auto"])
        after = {path.relative_to(self.output): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(len(self.trainers), 1)

    def test_overlength_refused_before_run_is_created(self):
        self.tokenizer.apply_chat_template.return_value = list(range(3073))
        with self.assertRaisesRegex(ValueError, "Refusing to truncate"):
            self.run_main()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.trainers, [])
        self.assertFalse(self.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_explicit_revision_is_used_for_resolution_tokenizer_and_trainer(self):
        pinned = "f" * 40
        self.run_main(["--revision", pinned])
        self.auto_config.from_pretrained.assert_called_once_with("Qwen/Qwen3.5-0.8B", revision=pinned)
        self.auto_tokenizer.from_pretrained.assert_called_once_with("Qwen/Qwen3.5-0.8B", revision=pinned)
        self.assertEqual(self.trainers[0].args.model_init_kwargs["revision"], pinned)


if __name__ == "__main__":
    unittest.main()
