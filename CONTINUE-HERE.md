# CONTINUE HERE — v4 Training Handoff (2026-09-14)

Read this first on the new laptop. It picks up the FYP local-SLM work exactly
where the last session stopped.

## TL;DR

- Repo: https://github.com/AbdullahUsman0/SLM-FineTuning-Testing (branch `main`)
- Corpus v4 is **built and verified**: 420 scenarios, 2991 SFT examples
  (2322 train / 669 validation), 48 held-out test scenarios in
  `corpus-v4/splits/test.jsonl`.
- v4 Colab training was **interrupted after epoch 1** to beat the free Colab
  session limit. Epoch 1 = `checkpoint-146`, saved at
  `/content/qwen35-08b-lora-v4/checkpoint-146` in the old runtime
  (ephemeral — gone if that session ended).
- No quality claims have been made about v4. Final step = run the same
  held-out component/slot evaluation as v3 and compare.

## v3 reference numbers (what v4 must beat)

| Metric                        | v3 LoRA              | OpenAI baseline |
| ----------------------------- | -------------------- | --------------- |
| Held-out component slot F1    | 0.0751               | 0.4068          |
| eval_loss                     | 0.0013825836358591914| —               |
| token accuracy                | 0.9996911381249842   | —               |

Do not claim v4 is better unless it beats v3 on the same held-out evaluation.

## New laptop setup

```bash
git clone https://github.com/AbdullahUsman0/SLM-FineTuning-Testing.git
cd SLM-FineTuning-Testing

# Eval scripts import forecasting contracts from the fpy repo (pinned commit)
git clone https://github.com/int-abd-5/fpy.git ../fpy
git -C ../fpy fetch origin 04d52c015d1e3ecdefe92b87116f209361509b4b
git -C ../fpy checkout --detach 04d52c015d1e3ecdefe92b87116f209361509b4b

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r training/requirements-inference.txt  # local inference/eval
pip install -r training/requirements.txt            # only if training locally

# Sanity check
python -m unittest tests.test_corpus_v4 tests.test_training_helpers -v
```

`corpus-v4/`, `.cache/`, `models/`, `runtime/`, and `results/*.json` are
gitignored — the corpus is rebuilt on Colab by the setup cell, and adapters are
downloaded/mounted at run time.

## Continue training on Colab (primary path)

1. Open `notebooks/Qwen_LoRA_v4_Training_Colab.ipynb` in Google Colab
   (GitHub tab → this repo → that file). Select **Runtime > Change runtime
   type > T4 GPU**.
2. Add your `OPENAI_API_KEY` via the **Secrets (🔑) panel** (User secrets).
   The setup cell rebuilds corpus v4, which calls OpenAI for paraphrases
   (cached in `.cache/paraphrases_v4.json`). Never paste the key into a cell
   or commit it.
3. Run the setup cell, then the training cell. Checkpoints are written to
   Google Drive at `/content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v4`
   and the cell resumes with `--resume-from-checkpoint auto`.
4. Run the verification cell to confirm `best-adapter/` and `evaluation.json`.

**Timing:** epoch 1 took **4h39m** on a T4. Three epochs (~13.5h) exceed the
free Colab session limit, so either:

- keep `--epochs 3` and re-run the training cell in a second session if the
  first is cut off (auto-resume continues from the highest Drive checkpoint;
  early-stopping patience 1 may also finish it sooner), or
- change the training cell to `--epochs 2` for a guaranteed single-session
  finish (~9.5h).

### If the old Colab session is still alive

The epoch-1 checkpoint lives at `/content/qwen35-08b-lora-v4/checkpoint-146`
(NOT on Drive). Copy it first, then let the training cell resume from it:

```python
import shutil
from pathlib import Path
dst = Path("/content/drive/MyDrive/FYP-model-runs/qwen35-08b-lora-v4")
dst.mkdir(parents=True, exist_ok=True)
shutil.copytree("/content/qwen35-08b-lora-v4", dst, dirs_exist_ok=True)
```

That session also has a corrupted extra cell (training code merged into the
verification cell, referencing `/content/qwen35-08b-lora-v4`) — delete it
before running anything. If the session is dead, ignore this section; the
fresh path above rebuilds and retrains from scratch.

## Evaluate v4 against v3 (required before any claim)

1. Copy the Drive folder `FYP-model-runs/qwen35-08b-lora-v4` (`best-adapter/`
   and `evaluation.json`) into this repo at
   `training-runs/qwen35-08b-lora-v4/`.
2. Run the held-out component evaluation against the 48 held-out v4 test
   scenarios with the fine-tuned adapter (same methodology as the v3 run;
   slot matching is token-F1 in `local_slm_lab/component_eval.py`):

   ```bash
   python scripts/evaluate-components.py \
       --provider local-sft \
       --cases corpus-v4/splits/test.jsonl \
       --config config.evaluation.json \
       --output results/v4-component-eval.json
   ```

3. Compare slot F1 / token accuracy against the v3 table above and report
   both side by side. Also run `--provider openai` on the same cases if you
   want the refreshed baseline.

## Known limitations

- One training record (`complete-kse_index / reviewed_base / train`) is 3119
  tokens > `max_length` 3072; training uses `--allow-truncation`, which may
  drop its completion tokens. Mention this when reporting.
- If only 2 epochs finish, that is an acceptable final v4 run given the free
  session limit — report it as such.

## Security rules (unchanged)

- `OPENAI_API_KEY` only via Colab Secret or runtime environment variable.
  Never in chat, notebook cells, source files, or commits.
- Never commit Colab runtime/proxy tokens or `.env` files.
