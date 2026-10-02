# Training completion summary

Run and TRAIN_RUN: `D:\SLM\FYP-model-runs\qwen35-2b-lora-v5-anydesk-20261001T141017Z`

- GPU: NVIDIA RTX A4000, 16 GB; CUDA, bfloat16, effective batch 16.
- Repository: `b0a3a74f299aa6ecf0d3d30c4623136e4c89b306`
- fpy: `04d52c015d1e3ecdefe92b87116f209361509b4b`
- Base: Qwen/Qwen3.5-2B at `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Data: frozen v5-20260919-r1; 4,435 training and 948 validation examples; no truncation.
- Settings: two fixed epochs, LR 5e-5, length limit 3584, checkpoint interval 10, early stopping 0.
- Started UTC: 2026-10-01T14:20:07.614113+00:00
- Training and adapter save completed UTC: 2026-10-02T08:39:16.300720+00:00
- Wall time including interruptions: 18h 19m 8s.
- Approximate summed Trainer elapsed time: 9h 3m 23s; includes epoch validation and replayed steps.
- Two Windows-restart interruptions; resumed at verified steps 100 and 370; eight optimizer steps replayed.
- Final step/epoch: 556 / 2.0.
- Retained, independently verified checkpoints: every 10 steps from 10 through 550, plus final step 556.
- Selection: final-step adapter; loss-best and task-quality selection were not applied.
- Final training metrics: {"train_runtime": "1.121e+04", "train_samples_per_second": "0.791", "train_steps_per_second": "0.05", "train_loss": "6.567e-06", "epoch": "2"}
- Last logged training window loss (step 555): 1.9928252731915565e-05.
- Resume accounting: the Trainer-reported train_loss is not an uninterrupted two-epoch mean.
- Final validation loss: 3.3780084777390584e-05.
- Core adapter bytes: 67334101; complete adapter directory bytes: 87343831.
- Offline synthetic extraction smoke: passed; base and adapter loaded on CUDA and returned valid extraction JSON.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| adapter_model.safetensors | 67332688 | `e97f5e9808713a1daaaa105b3251b96f2f273834eca0e4e6da53989c652d7080` |
| adapter_config.json | 1413 | `fd6f87135e9a032d0bdbeed093a379d5f3871d4796888df27dcacb032be3ac50` |

Training manifest SHA-256: `0ca890459933b589aa0c5a866e890324c8f4467bbe08ad20cfa335ece48cecc1`

Authorization: user-confirmed corpus inspection and full training, not independent or formally audited human review. Matched pilot comparison was incomplete. No behavioral quality claim is made from training loss or this loading smoke. No sealed final split or paid OpenAI evaluation was used.

Environment warnings and recovery details are in RUN_NOTES.md. Detailed verification, epoch losses, provenance, and timings are in completion-audit.json and launch-provenance.json.
Smoke limitation: the single valid response extracted the seven-day forecast horizon but omitted the daily frequency, electricity-demand target, and CSV-file information. This smoke establishes loading/generation only; behavioral evaluation remains pending.
