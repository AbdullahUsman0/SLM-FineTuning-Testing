"""Post-hoc paper-validity audit. No model calls or replacement primary scores.

Run in the recorded evaluation environment, with its isolated offline-deps on
PYTHONPATH. Supply the unpacked corrective result folder and the exact patched
fpy checkout. This audits representation/admission differences, not production
performance under a different prompt or an unobserved rollout trajectory.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deny_outbound(sock, address):
    raise RuntimeError('Network disabled for saved-output audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--fpy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve earlier audit outputs'
    os.environ.pop('OPENAI_API_KEY', None)
    socket.socket.connect = deny_outbound
    socket.socket.connect_ex = deny_outbound
    evidence = args.evidence.resolve()
    sys.path.insert(0, str(evidence / 'evaluator'))
    import pipeline as p
    from controls import digest
    p.configure(args.fpy, evidence)
    from forecasting_assistant.domain.models import ExtractorResult
    protocol = json.loads((evidence / 'protocol.json').read_bytes())
    for name, wanted in protocol['fpy_source_sha256'].items():
        assert sha(args.fpy / name) == wanted, 'Patched fpy source mismatch'
    for name, wanted in protocol['runner_source_sha256'].items():
        assert sha(evidence / 'evaluator' / name) == wanted, 'Frozen evaluator source mismatch'
    cases = json.loads((evidence / 'main-cases.json').read_bytes())
    assert digest(cases) == protocol['main_case_sha256']
    turns = {(c['scenario_id'], t['turn_id']): t for c in cases for t in c['turns']}
    rows = [json.loads(line) for line in gzip.decompress((evidence / 'main-scored-records.jsonl.gz').read_bytes()).splitlines()]
    primary = json.loads((evidence / 'main-summary.json').read_bytes())
    report = {'scope': 'Post-hoc saved-output audit, not fresh inference or a replacement primary score',
              'new_model_calls': 0, 'paid_requests': 0, 'network_disabled': True,
              'original_predictions_gold_and_primary_scores_modified': False,
              'fpy_and_evaluator_source_hashes_verified': True,
              'fpy_commit': protocol['fpy_commit'], 'protocol_sha256': sha(evidence / 'protocol.json'),
              'audit_script_sha256': sha(__file__), 'groups': {}}
    verified = 0
    for arm in ('v8', 'openai'):
        for track in ('component', 'rollout'):
            group = [r for r in rows if r['key'][1:3] == [arm, track]]
            assert len(group) == 101
            counts = Counter(native_Pydantic_parser_accepts=0, experimental_parser_accepts=0,
                             exact_matches_lost_to_whole_response_rejection=0,
                             native_ignored_status_updates=0, unknown_turns_with_only_ignored_nonintent_updates=0)
            for row in group:
                turn = turns[tuple(row['key'][3:])]
                metrics = p.raw_score(row, turn, protocol['response_schema'])
                assert all(row[k] == value for k, value in metrics.items())
                verified += 1
                obj = json.loads(row['raw_output'])
                try:
                    ExtractorResult.model_validate(obj)
                    counts['native_Pydantic_parser_accepts'] += 1
                except Exception:
                    pass
                counts['experimental_parser_accepts'] += row['raw_contract_admitted']
                matches = sum((Counter(p.tuples(obj)) & Counter(p.tuples(turn['gold_extraction']))).values())
                if not row['raw_contract_admitted']:
                    counts['exact_matches_lost_to_whole_response_rejection'] += matches
                updates = [u for u in obj['updates'] if u['slot_id'] != 'intent']
                ignored = [u for u in updates if u['status'] in {'unmentioned', 'dont_care'}]
                counts['native_ignored_status_updates'] += len(ignored)
                counts['unknown_turns_with_only_ignored_nonintent_updates'] += bool(row['unknown_turn'] and updates and len(updates) == len(ignored))
            assert p.prf(group) == primary[arm][track]['raw_exact']
            report['groups'][arm + '/' + track] = {'counts': dict(counts), 'unchanged_primary_raw_exact': p.prf(group)}
    assert verified == len(rows) == 404
    report['primary_record_scores_reproduced'] = verified
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
