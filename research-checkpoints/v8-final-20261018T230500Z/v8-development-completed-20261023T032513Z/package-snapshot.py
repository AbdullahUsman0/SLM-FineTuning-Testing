"""Package completed v8 development evidence without touching ongoing evaluation."""
import csv
import gzip
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r'D:\SLM\SLM-FineTuning-Testing')
RUN = Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v8-final-20261018T230500Z')
CONTROL = Path(r'D:\SLM\FYP-experiment-control\v8-final-20261018T230500Z')
SOURCE = RUN / 'development-evaluation'
DEST = REPO / 'research-checkpoints/v8-final-20261018T230500Z/v8-development-completed-20261023T032513Z'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def main():
    if DEST.exists():
        raise FileExistsError('Use a new snapshot; existing evidence is never replaced')
    spec = importlib.util.spec_from_file_location('frozen_followup', CONTROL / 'followup-r4.py')
    followup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(followup)
    config = read(CONTROL / 'workflow-config-r4.json')
    followup.check_sources(config, REPO)
    summary = followup.validate_runtime(SOURCE)
    freeze = read(SOURCE / 'freeze.json')
    assert sha(SOURCE / 'freeze.json') == summary['freeze_sha256']
    for name, expected in freeze['adapter_files'].items():
        assert sha(RUN / 'full/best-adapter' / name) == expected, name
    sys.path.insert(0, str(REPO / 'training/final_round'))
    import evaluate_development as evaluator
    cases = evaluator.select_cases(list(evaluator.rows(evaluator.CORPUS / 'splits/validation.jsonl.gz')))
    assert evaluator.digest(cases) == freeze['cases_sha256']
    expected_keys = {(track, c['scenario_id'], t['turn_id']) for track in ('component', 'rollout') for c in cases for t in c['turns']}
    raw = [json.loads(line) for line in (SOURCE / 'calls.jsonl').read_bytes().splitlines()]
    with gzip.open(SOURCE / 'scored-records.jsonl.gz', 'rt', encoding='utf-8') as stream:
        scored = [json.loads(line) for line in stream]
    assert len(raw) == len(scored) == len(expected_keys) == 996
    raw_by_key = {tuple(r['key']): r for r in raw}
    assert set(raw_by_key) == {tuple(r['key']) for r in scored} == expected_keys
    for row in scored:
        for key, value in raw_by_key[tuple(row['key'])].items():
            assert row[key] == value, (row['key'], key)
    for track, metrics in summary['tracks'].items():
        items = [r for r in scored if r['key'][0] == track]
        assert metrics['calls'] == len(items) == 498
        assert metrics['nonintent_exact'] == evaluator.prf([r['confusion'] for r in items])
        for field in ('json_valid', 'schema_valid', 'admitted'):
            assert metrics[field] == sum(r[field] for r in items)
        assert metrics['median_extraction_seconds'] == statistics.median(r['latency_seconds'] for r in items)
    DEST.mkdir()
    for name in ('summary.json', 'freeze.json', 'runtime.json'):
        shutil.copyfile(SOURCE / name, DEST / name)
    archive = followup.checked_zip(DEST / 'v8-development-raw.zip', {name: SOURCE / name for name in ('calls.jsonl', 'scored-records.jsonl.gz', 'freeze.json', 'runtime.json', 'summary.json')})
    write(DEST / 'raw-archive-receipt.json', archive)
    write(DEST / 'failure-findings.json', {'scope': 'Automated development triage; independent human adjudication pending', 'calls': followup.fail_rows(SOURCE)})
    write(DEST / 'granular-domain-breakdown.json', {'scope': 'Prespecified descriptive breakdown of 20 development domains, no significance test', 'v8': followup.granular_breakdown(SOURCE)})
    fields = ['track', 'calls', 'exact_f1', 'precision', 'recall', 'json_valid', 'schema_valid', 'admitted', 'extra_slot_flags_including_rejected', 'forbidden_slot_flags_including_rejected', 'correct_correction_calls', 'correction_calls', 'complete_state_after_correction', 'complete_state_turns', 'unknown_successful_empty_calls', 'unknown_preserved_nonintent_state', 'unknown_turns', 'median_extraction_seconds']
    with (DEST / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for track, metrics in summary['tracks'].items():
            row = {k: metrics[k] for k in fields if k not in ('track', 'exact_f1', 'precision', 'recall')}
            row.update(track=track, exact_f1=metrics['nonintent_exact']['f1'], precision=metrics['nonintent_exact']['precision'], recall=metrics['nonintent_exact']['recall'])
            writer.writerow(row)
    write(DEST / 'SNAPSHOT_STATUS.json', {
        'status': 'verified_v8_development_complete_v6_comparison_pending',
        'verified_at_UTC': datetime.now(timezone.utc).isoformat(),
        'run': str(RUN), 'completed_calls': 996, 'conversations': 100,
        'checks': ['frozen sources unchanged', 'adapter SHA-256 matches evaluation freeze', 'journal and freeze SHA-256 match summary', 'selected development case digest matches freeze', 'all 996 expected keys present once', 'scored raw fields match durable journal', 'primary F1 and structure counts recomputed', 'ZIP CRC and every member SHA-256 verified'],
        'source_files_sha256': {p.name: sha(p) for p in SOURCE.iterdir() if p.name in ('calls.jsonl', 'scored-records.jsonl.gz', 'summary.json', 'freeze.json', 'runtime.json')},
        'v6_status_at_snapshot': 'running_separately_not_included', 'openai_comparison': 'not_run_on_this_cohort',
        'independent_human_review_complete': False, 'final_test_accessed': False,
        'model_winner_established': False, 'confidence_intervals_implemented': False, 'new_paid_API_calls': 0,
    })
    (DEST / 'README.md').write_text('''# Completed v8 development diagnostic; v6 comparison pending

The experimental final-step v8 adapter completed all 996 calls on 100 fixed synthetic development conversations (498 turns in each of two tracks). This snapshot publishes v8 immediately while v6 continues separately. It does not claim the combined experiment or an OpenAI comparison is complete.

| Measurement | Component: supplied gold prior | Rollout: own accepted prior |
|---|---:|---:|
| Exact non-intent slot/value/status micro F1 | 0.963953 | 0.963953 |
| Precision / recall | 0.965126 / 0.962783 | 0.965126 / 0.962783 |
| JSON syntax / declared schema valid | 498/498 | 498/498 |
| Application admission | 491/498 | 491/498 |
| Exact complete state at a turn | 486/498 | 458/498 |
| Correct correction updates | 120/120 | 120/120 |
| Complete state after correction | 120/120 | 111/120 |
| Successful empty unknown responses | 103/103 | 103/103 |
| Unknown turns preserving non-intent state | 103/103 | 103/103 |
| Extra current-turn slot flags, including rejected | 1 | 1 |
| Forbidden-slot flags, including rejected | 0 | 0 |
| Median extraction seconds | 7.343 | 7.353 |

F1 is exact slot ID + decoded value + status, excluding intent; it is not a percentage of successful conversations. Complete-state counts above are turns, not conversations. Valid JSON does not imply application admission or correct requirements. Extra-slot flags do not count every incorrect value or every possible invented requirement; false positives and detailed raw outputs must also be inspected. Exact correction updates can coexist with an accumulated-state failure from an earlier turn.

The main language weakness is Roman Urdu: exact F1 0.771536 versus English 0.985417 and mixed 1.000000, in each track. There are only 10 Roman Urdu conversations and 10 mixed conversations; these are descriptive subgroup results. Both tracks have 46 false negatives and 43 false positives. Excluding the explicit seeded state-repair category gives F1 0.958739 in both tracks. The full summaries retain all language/domain-family and seeded-intervention exclusion results; granular-domain-breakdown.json covers all 20 individual domains. failure-findings.json is automated triage, not human adjudication.

The raw ZIP contains calls.jsonl (all raw responses and latency), scored-records.jsonl.gz, freeze.json, runtime.json, summary.json and an internal SHA-256 manifest. Its adjacent checksum and raw-archive-receipt.json permit integrity checks. SNAPSHOT_STATUS.json records independent packaging consistency checks; artifact-manifest.json hashes the publication files. No base weights, adapter weights, optimizer state, credentials or caches are included here.

Generation used the active v8 LoRA on NVIDIA RTX A4000, CUDA BF16, the pinned Qwen/Qwen3.5-2B revision 15852e8c16360a2fea060d615a32b45270f8a8fc, greedy decoding, thinking disabled, max_new_tokens 1024, one call per turn and no repair. Source training/final_round/evaluate_development.py and provider hashes, the cohort digest, instructions policy and adapter hashes are frozen in freeze.json. The matched v6 run uses a separate output folder and will be published by the existing companion after completion.

These inspected synthetic development conversations come from the public v8 validation split; they are not an untouched final test. Synthetic source/template dependence, pending independent human annotation, small subgroups and explicit seeded state fixtures limit generalization. No significance test, statistical equivalence, final-test result or model winner is established. The older OpenAI comparison uses different cases/policy and cannot be compared numerically with these v8 scores. No paid API calls or sealed final-label access occurred.

Training evidence and the downloadable adapter are documented in ../training-completed-20261023T002215Z/LOCAL_CODEX_HANDOFF.md and ../ADAPTER_RELEASE_NOTICE.md. This snapshot does not change the training run or the ongoing v6 diagnostic. Reproduce with the supplied training/final_round instructions in a NEW diagnostic output directory; do not rerun or overwrite the original completed outputs. package-snapshot.py records the exact read-only source checks and publication packaging used here, with paths specific to the training PC.
''', encoding='utf-8')
    shutil.copyfile(__file__, DEST / 'package-snapshot.py')
    token = re.compile(r'(?<![A-Za-z0-9])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,})')
    for path in DEST.iterdir():
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as package:
                for name in package.namelist():
                    body = package.read(name)
                    if name.endswith('.gz'):
                        body = gzip.decompress(body)
                    assert not token.search(body.decode('utf-8-sig')), 'Credential-shaped literal in ZIP'
        else:
            assert not token.search(path.read_text(encoding='utf-8-sig')), 'Credential-shaped literal in publication'
    write(DEST / 'artifact-manifest.json', {'files_sha256': {p.name: sha(p) for p in DEST.iterdir() if p.is_file()}, 'weights_optimizer_credentials_caches_included': False})
    print(json.dumps({'snapshot': str(DEST), 'archive': archive, 'status': 'verified_ready_to_commit'}))


if __name__ == '__main__':
    main()
