"""Replay completed raw journals offline, without generation or ledger mutation."""
import json
import os
from pathlib import Path
import socket
import sys

RUN = Path(__file__).resolve().parent
CONFIG = json.loads((RUN / 'config.json').read_bytes())
sys.path[:0] = [str(RUN / 'offline-deps'), CONFIG['code']]
os.environ.pop('OPENAI_API_KEY', None)


original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex


def guarded_connect(sock, address):
    if not isinstance(address, tuple) or address[0] not in {'127.0.0.1', '::1', 'localhost'}:
        raise RuntimeError('Outbound network disabled during raw-response replay')
    return original_connect(sock, address)


def guarded_connect_ex(sock, address):
    if not isinstance(address, tuple) or address[0] not in {'127.0.0.1', '::1', 'localhost'}:
        raise RuntimeError('Outbound network disabled during raw-response replay')
    return original_connect_ex(sock, address)


socket.socket.connect = guarded_connect
socket.socket.connect_ex = guarded_connect_ex

from controls import atomic, digest, read_journal, sha
import pipeline as p
p.configure(RUN / 'fpy', Path(CONFIG['repo']))
from runner import validate_protocol
from forecasting_assistant.domain.schema import create_initial_state, load_schema
from forecasting_assistant.application.state_reducer import apply_extraction


def main():
    protocol = validate_protocol(RUN, Path(CONFIG['repo']))
    assert json.loads((RUN / 'main-completion.json').read_bytes())['status'] == 'complete'
    cases = json.loads((RUN / 'main-cases.json').read_bytes())
    saved = {tuple(r['key']): r for r in [json.loads(line) for line in (RUN / 'main-scored-records.jsonl').read_bytes().splitlines()]}
    journal = read_journal(RUN / 'calls.jsonl')
    schema = load_schema()
    checked = 0
    for arm in ('v8', 'openai'):
        for track in ('component', 'rollout'):
            for case in cases:
                state = create_initial_state(schema)
                for turn in case['turns']:
                    if track == 'component':
                        state = p.restore(turn, schema)
                    else:
                        p.intervention(state, turn.get('state_intervention'))
                    for prior in state.turns:
                        prior.assistant_message = None
                    before = {k: v for k, v in p.snapshot(state).items() if k != 'intent'}
                    key = ('main', arm, track, case['scenario_id'], turn['turn_id'])
                    record, archived = journal[key], saved[key]
                    context = p.extraction_context(state, turn['message'], schema)
                    assert context == record['current_input']
                    assert digest([protocol['instructions'], context, turn['gold_extraction'], turn['expected_state_after']]) == record['identity']
                    metrics = p.raw_score(record, turn, protocol['response_schema'])
                    assert all(archived[k] == v for k, v in metrics.items())
                    reducer_admitted = False
                    reducer_error = None
                    try:
                        proposed = p.admit(record['raw_output'], protocol['response_schema'], record.get('capped', False), record.get('refused', False))
                        reduced = apply_extraction(state, proposed, schema, len(state.turns) + 1, turn['message'])
                        reducer_admitted = True
                        reduced_state = p.snapshot(reduced)
                    except Exception as error:
                        reducer_error = type(error).__name__
                        reduced_state = p.snapshot(state)
                    assert reducer_admitted == archived['reducer_admitted']
                    assert reducer_error == archived['reducer_error_type']
                    assert reduced_state == archived['reducer_state_before_recovery']
                    state, provenance = p.run_pipeline(state, turn['message'], schema, record, protocol['response_schema'])
                    snapshot = p.snapshot(state)
                    assert snapshot == archived['pipeline_state']
                    assert (snapshot == turn['expected_state_after']) == archived['complete_pipeline_state']
                    assert provenance['recovered_slot_ids'] == archived['provenance']['recovered_slot_ids']
                    assert (before == {k: v for k, v in snapshot.items() if k != 'intent'}) == archived['unknown_preserved']
                    checked += 1
    assert checked == len(saved) == 404
    result = {'passed': True, 'raw_responses_rescored': checked, 'current_pipeline_states_replayed': checked,
              'component_and_rollout_inputs_reproduced': True, 'outbound_network_disabled_except_loopback': True,
              'additional_model_calls': 0, 'budget_mutation': False,
              'scored_records_sha256': sha(RUN / 'main-scored-records.jsonl'),
              'script_sha256': sha(__file__), 'protocol_sha256': sha(RUN / 'protocol.json')}
    atomic(RUN / 'recorded-main-replay-verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
