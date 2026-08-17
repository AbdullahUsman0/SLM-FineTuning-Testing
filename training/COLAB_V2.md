# Colab retraining: Qwen3.5-0.8B LoRA v2

The easiest repeatable workflow is:

- GitHub stores source code and the small synthetic datasets.
- Google Drive stores checkpoints and the final adapter.
- Colab clones or pulls GitHub, while training writes directly into Drive.

This avoids repeated ZIP uploads. It also permits a free Colab runtime to
resume from the latest Drive checkpoint after a disconnect.

The prepared notebook is `notebooks/qwen35_lora_v2_colab.ipynb`. Open it in
Colab, change `REPO_URL` in the first code cell, select a GPU runtime, and run
the cells from top to bottom.

## One-time GitHub setup on Windows

The current local repository has no commits or remote yet. Create an empty
GitHub repository named `local-slm-lab` without adding a README, then run:

```powershell
cd "C:\Users\Mahad Enterprises\OneDrive\Desktop\FYP\local-slm-lab"
git add -A
git commit -m "Prepare Qwen LoRA corpus v2 and Colab training"
git branch -M main
git remote add origin https://github.com/AbdullahUsman0/SLM-FineTuning-Testing.git
git push -u origin main
```

Large directories such as `.peft-deps`, `training-runs`, models, and generated
result JSON files are already ignored and will not be pushed.

For later updates, the Windows side is only:

```powershell
git add -A
git commit -m "Describe the update"
git push
```

Then rerun the notebook's Git synchronization cell; it performs a fast-forward
`git pull`.

## Private GitHub repository

For a private repository, create a fine-grained GitHub token with read access
to only this repository. Add it in Colab under the key icon / Secrets as
`GITHUB_TOKEN`, and enable notebook access. The notebook sends it as a temporary
HTTP header and does not save it in the cloned remote URL.

## Equivalent manual Colab command

After cloning and mounting Drive, run:

```bash
cd /content/local-slm-lab
python -m pip install -r training/requirements.txt
python scripts/build-corpus-v2.py
python scripts/verify-corpus-v2.py
python -m unittest tests.test_corpus_v2 tests.test_training_helpers -v
python training/train_lora.py \
  --train corpus-v2/sft/train.jsonl \
  --validation corpus-v2/sft/validation.jsonl \
  --output /content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v2 \
  --epochs 4 \
  --early-stopping-patience 1 \
  --max-length 3072 \
  --resume-from-checkpoint auto
```

The script performs a tokenizer-length preflight and fails instead of silently
truncating labels. Evaluation and checkpoint saving happen each epoch;
`load_best_model_at_end=True`, `metric_for_best_model=eval_loss`, and
`greater_is_better=False`. Early stopping prevents continuing after validation
stops improving.

The output is already in Drive, so no archive is required. If Colab disconnects,
open a new GPU runtime and rerun the notebook: `--resume-from-checkpoint auto`
selects the highest saved checkpoint.

An optional portable handoff ZIP can still be created after completion:

```bash
cd /content/local-slm-lab
zip -r /content/qwen35-v2-handoff.zip \
  /content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v2 \
  corpus-v2/manifest.json \
  training/train_lora.py \
  training/COLAB_V2.md
```

Download `qwen35-v2-handoff.zip` and restore it beside the retained v1 run.
