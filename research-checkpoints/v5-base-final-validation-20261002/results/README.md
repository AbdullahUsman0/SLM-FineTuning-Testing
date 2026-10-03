# Stock base versus final adapter: development validation

Both reports are complete: **231 scenarios, 717 extraction calls and 231 question calls per model**.

| Metric | Stock base | Final adapter |
| --- | ---: | ---: |
| inclusive tuple micro F1 | 0.000000 | 0.999524 |
| nonintent tuple micro F1 | 0.000000 | 0.999454 |
| raw_json_valid | 0.833333 | 1.000000 |
| raw_schema_valid | 0.265823 | 1.000000 |
| provider_success | 0.265823 | 1.000000 |
| scenario_exact_nonintent | 0.000000 | 0.991342 |
| reducer_transition_accuracy | 0.004184 | 0.997211 |
| correction_flag_accuracy | 0.029289 | 1.000000 |
| correction_tuple_accuracy | 0.000000 | 1.000000 |
| forbidden_call_rate | 0.573222 | 0.002789 |
| hallucinated_slot_call_rate | 0.574616 | 0.002789 |
| empty_update_collapse | 0.969163 | 0.000000 |
| question_contract | 0.995671 | 1.000000 |
| question_exact_match | 0.831169 | 1.000000 |

Final-adapter minus stock-base non-intent F1: **0.999454**;
paired source-cluster percentile 95% interval **[0.998586, 1.000000]**
(10,000 samples, seed 42). See [comparison.json](comparison.json) for both inclusive and non-intent intervals.

Model: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. Same BF16 CUDA runtime,
RTX A4000, pinned prompts, greedy decoding, 1,024 generated-token limit, no retry or repair.
The original v5 component protocol supplies gold context at each turn; these are component
results, not an end-to-end conversation assessment. Warmup uses the first development smoke
scenario before each session; warmup calls are excluded. Completed scenarios survive interruption;
only interrupted scenarios are replayed. Timing may vary across sessions.

The adapter is optimizer step 556, epoch 2.0, chosen by final step. This compares one adapter;
it does not select a winner across retained checkpoints. Development validation was also used
for teacher-forced validation loss during training. **No sealed final test or paid API was used.**
Training loss and smoke loading do not establish extraction quality. Training authorization
does not stand in for an independent formal corpus review; that review remains pending.

Study files:

- [Base metrics and provenance](base-metrics.json), [adapter metrics and provenance](lora-metrics.json).
- [Per-slot comparison](per-slot.csv), [paired extraction errors](paired-extraction-errors.csv).
- [Base full report](base-validation.json.gz), [adapter full report](lora-validation.json.gz):
  lossless gzip JSON with all raw responses, gold, contexts, parse failures, timing and token counts.
  Question outputs are included in these full reports.
- [Training summary](training-summary.md), [independent completion audit](training-audit.json).
- [Artifact hashes](artifact-hashes.json). No weights, checkpoints, cache, environment files or secrets are included.

Decompress a report with Python:

```python
import gzip, json
with gzip.open("lora-validation.json.gz", "rt", encoding="utf-8") as f:
    report = json.load(f)
print(report["metrics"])
```

To reproduce locally, run `scripts/evaluate-v5-resumable.py` separately with `--provider base`
and `--provider lora --adapter PATH`, the pinned model/revision above and distinct new `--run-dir`
directories. Then use `scripts/compare-v5.py --left BASE/combined-report.json
--right LORA/combined-report.json --output comparison.json`.
Adapter weights remain at the local path recorded in the metadata; reproducing the LoRA run
requires those weights. The evaluation source commit and dependency pins are in each report's provenance.
