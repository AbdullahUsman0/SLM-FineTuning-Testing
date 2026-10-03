"""Package completed matched development reports for study; never publishes weights."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_slm_lab.v5_eval import paired_bootstrap, sha256, strict_json, confusion, prf


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def package(run_root, training_run, output):
    reports = {}
    for name in ('base', 'lora'):
        reports[name] = strict_json((run_root / name / 'combined-report.json').read_text(encoding='utf-8'))
        report = reports[name]
        if report.get('status') != 'complete' or report['scenario_count'] != 231:
            raise ValueError('requires all 231 completed validation scenarios')
        if report['metrics']['calls'] != {'total': 948, 'extract': 717, 'ask': 231}:
            raise ValueError('unexpected validation call counts')
        if report['metadata']['input']['split'] != 'validation':
            raise ValueError('development validation only')
        if report['metadata']['settings']['provider'] != name:
            raise ValueError('incorrect provider role')
    comparison = paired_bootstrap(reports['base'], reports['lora'], samples=10000, seed=42)
    audit = strict_json((training_run / 'completion-audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'passed':
        raise ValueError('training audit did not pass')
    adapter = training_run / 'full/best-adapter'
    expected = {name: sha256(adapter / name) for name in ('adapter_model.safetensors', 'adapter_config.json')}
    if reports['lora']['metadata']['settings']['adapter_sha256'] != expected:
        raise ValueError('evaluation adapter differs from audited training adapter')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    comparison['reports'] = {}
    for name, report in reports.items():
        source = run_root / name / 'combined-report.json'
        target = output / f'{name}-validation.json.gz'
        with source.open('rb') as handle, target.open('wb') as destination:
            with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0) as compressed:
                shutil.copyfileobj(handle, compressed)
        # Read back the published copy, including every raw response.
        with gzip.open(target, 'rt', encoding='utf-8') as handle:
            if strict_json(handle.read()) != report:
                raise ValueError('archive verification failed')
        comparison['reports'][name] = {'file': target.name, 'report_sha256': sha256(source),
                                      'gzip_sha256': sha256(target), 'gzip_bytes': target.stat().st_size}
        write_json(output / f'{name}-metrics.json', {'metrics': report['metrics'],
                   'category_metrics': report['category_metrics'], 'metadata': report['metadata']})
    write_json(output / 'comparison.json', comparison)
    for source, name in (('TRAINING_COMPLETION_SUMMARY.md', 'training-summary.md'),
                         ('completion-audit.json', 'training-audit.json')):
        (output / name).write_text((training_run / source).read_text(encoding='utf-8'),
                                  encoding='utf-8', newline='\n')
    with (output / 'per-slot.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['slot', 'base_precision', 'base_recall', 'base_f1', 'adapter_precision', 'adapter_recall', 'adapter_f1'])
        base_slots = reports['base']['metrics']['per_slot']
        adapter_slots = reports['lora']['metrics']['per_slot']
        for slot in sorted(set(base_slots) | set(adapter_slots)):
            values = base_slots.get(slot, prf(0, 0, 0))
            candidate = adapter_slots.get(slot, prf(0, 0, 0))
            writer.writerow([slot] + [values[key] for key in ('precision', 'recall', 'f1')]
                            + [candidate[key] for key in ('precision', 'recall', 'f1')])
    with (output / 'paired-extraction-errors.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['scenario', 'category', 'call_key', 'message', 'gold', 'base_output', 'adapter_output',
                         'base_tp_fp_fn', 'adapter_tp_fp_fn', 'base_success', 'adapter_success'])
        right = {tuple(record['key']): record for record in reports['lora']['records']}
        for left in reports['base']['records']:
            if left['task'] != 'extract':
                continue
            candidate = right[tuple(left['key'])]
            counts = [confusion(record, nonintent=True) for record in (left, candidate)]
            if all(value[1:] == (0, 0) for value in counts) and left['prediction_valid'] and candidate['prediction_valid']:
                continue
            writer.writerow([left['scenario_id'], left['category'], json.dumps(left['key']), left['context']['message'],
                             json.dumps(left['expected'], ensure_ascii=False),
                             json.dumps(left.get('emitted'), ensure_ascii=False),
                             json.dumps(candidate.get('emitted'), ensure_ascii=False),
                             *[json.dumps(value) for value in counts], left['provider_success'], candidate['provider_success']])
    rows = ['| Metric | Stock base | Final adapter |', '| --- | ---: | ---: |']
    for key in ('inclusive', 'nonintent'):
        rows.append(f"| {key} tuple micro F1 | {reports['base']['metrics'][key]['f1']:.6f} | {reports['lora']['metrics'][key]['f1']:.6f} |")
    for key in ('raw_json_valid', 'raw_schema_valid', 'provider_success', 'scenario_exact_nonintent',
                'reducer_transition_accuracy', 'correction_flag_accuracy', 'correction_tuple_accuracy',
                'forbidden_call_rate', 'hallucinated_slot_call_rate', 'empty_update_collapse',
                'question_contract', 'question_exact_match'):
        values = [report['metrics'][key]['rate'] for report in reports.values()]
        rows.append('| ' + key + ' | ' + ' | '.join('unavailable' if value is None else f'{value:.6f}' for value in values) + ' |')
    delta = comparison['nonintent']
    settings = reports['lora']['metadata']['settings']
    text = f'''# Stock base versus final adapter: development validation

Both reports are complete: **231 scenarios, 717 extraction calls and 231 question calls per model**.

{chr(10).join(rows)}

Final-adapter minus stock-base non-intent F1: **{delta['delta']:.6f}**;
paired source-cluster percentile 95% interval **[{delta['ci95'][0]:.6f}, {delta['ci95'][1]:.6f}]**
(10,000 samples, seed 42). See [comparison.json](comparison.json) for both inclusive and non-intent intervals.

Model: `{settings['model']}`, revision `{settings['revision']}`. Same BF16 CUDA runtime,
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
'''
    (output / 'README.md').write_text(text, encoding='utf-8', newline='\n')
    write_json(output / 'artifact-hashes.json', {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
               for p in sorted(output.iterdir()) if p.is_file()})
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.run_root, args.training_run, args.output)['nonintent']))
