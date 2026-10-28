"""Freeze native-admission rerun and prove reused SLM inputs/trajectories."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from controls import AMENDMENT, atomic, append, digest, read_journal, sha
import pipeline as p


def prepare(run):
    config=json.loads((run/'config.json').read_bytes());repo=Path(config['repo']);code=Path(config['code']);prior=Path(config['prior_run'])
    assert not (run/'protocol.json').exists(), 'Preserve prepared version'
    p.configure(run/'fpy',repo)
    old=json.loads((prior/'protocol.json').read_bytes())
    for name,wanted in old['fpy_source_sha256'].items():assert sha(run/'fpy'/name)==wanted
    for name,wanted in old['adapter_sha256'].items():assert sha(Path(old['adapter'])/name)==wanted
    assert p.build_instructions(repo)==old['instructions'] and p.wire_schema()==old['response_schema']
    db=sqlite3.connect(config['shared_ledger'])
    baseline=db.execute('SELECT key,context,bound,actual,state,response_sha,usage FROM calls ORDER BY key').fetchall()
    assert len(baseline)==440 and all(r[4]=='reconciled' for r in baseline)
    assert sum(r[3] for r in baseline)==AMENDMENT['prior_actual_nanousd']
    before=sqlite3.connect(run/'budget-before-amendment.sqlite3');db.backup(before);before.close();db.close()
    atomic(run/'budget-at-start.json',{'actual_USD':AMENDMENT['prior_actual_nanousd']/1e9,'prior_requests':len(baseline),'old_call_rows_sha256':digest(baseline),'snapshot_sha256':sha(run/'budget-before-amendment.sqlite3'),'reserved_USD':0,'uncertain_USD':0})
    atomic(run/'budget-authorization-amendment.json',AMENDMENT)
    correction={'version':'r4-native-admission-actionable-1','user_request':'rerun OpenAI for direct SLM comparison; additional USD1 approved','changes':['Native ExtractorResult.model_validate_json admission for both arms','Native candidate-value decoder once, with native fallback for plain strings','Exclude unmentioned/dont_care from actionable tuple scoring for both arms; report counts separately'],'unchanged':['20 main conversations and 2 pilot conversations','raw gold and approved state adjudication','common R3 instructions and typed payload','adapter/base','decoding limits','application reducer/recovery'],'scope':'corrective inspected development; common augmented adapter, not unmodified production prompt; no semantic synonym normalization or inference of new labels; human review incomplete'}
    atomic(run/'interface-correction.json',correction)
    from forecasting_assistant.domain.schema import create_initial_state,load_schema
    schema=load_schema();source_rows=[json.loads(line) for line in (prior/'calls.jsonl').read_bytes().splitlines()]
    source={tuple(row['key']):row for row in source_rows};assert len(source)==len(source_rows)
    reused=[];proof=[]
    for stage in ('pilot','main'):
        cases=json.loads((run/(stage+'-cases.json')).read_bytes())
        assert digest(cases)==old[stage+'_case_sha256']
        archived={tuple(r['key']):r for r in [json.loads(x) for x in (prior/(stage+'-scored-records.jsonl')).read_bytes().splitlines()]}
        for track in ('component','rollout'):
            for case in cases:
                state=create_initial_state(schema)
                for turn in case['turns']:
                    if track=='component':state=p.restore(turn,schema)
                    else:p.intervention(state,turn.get('state_intervention'))
                    for prior_turn in state.turns:prior_turn.assistant_message=None
                    key=(stage,'v8',track,case['scenario_id'],turn['turn_id']);record=source[key]
                    context=p.extraction_context(state,turn['message'],schema)
                    assert context==record['current_input']
                    assert digest([old['instructions'],context,turn['gold_extraction'],turn['expected_state_after']])==record['identity']
                    state,trace=p.run_pipeline(state,turn['message'],schema,record,old['response_schema'])
                    assert p.snapshot(state)==archived[key]['pipeline_state'], 'Changed SLM trajectory; reuse forbidden'
                    proof.append({'key':list(key),'source_record_sha256':digest(record),'input_identity':record['identity'],'pipeline_state_sha256':digest(p.snapshot(state))})
                    reused.append({**record,'emission_origin':'reused_r3_cuda_v8','source_run':str(prior),'source_record_sha256':digest(record)})
    assert len(reused)==220
    atomic(run/'slm-reuse-verification.json',{'passed':True,'calls':220,'main_calls':202,'pilot_calls':18,'source_journal_sha256':sha(prior/'calls.jsonl'),'all_inputs_and_pipeline_trajectories_exactly_unchanged':True,'proof':proof,'fresh_local_inference':False})
    assert not (run/'calls.jsonl').exists()
    for record in reused:append(run/'calls.jsonl',record)
    atomic(run/'local-technical-gate.json',{'passed':True,'interface_correction_sha256':sha(run/'interface-correction.json'),'SLM_reuse_verified':True,'reuse_verification_sha256':sha(run/'slm-reuse-verification.json')})
    protocol=deepcopy(old)
    protocol.update(version='current-pipeline-comparison-5',cohort='current-pipeline-dev20-20261028-r4-native',dispatch_namespace=run.name,interface_version=correction['version'],interface_correction_sha256=sha(run/'interface-correction.json'),local_technical_gate_sha256=sha(run/'local-technical-gate.json'),budget_start_sha256=sha(run/'budget-at-start.json'),budget_amendment_sha256=sha(run/'budget-authorization-amendment.json'),slm_reuse_sha256=sha(run/'slm-reuse-verification.json'),runner_source_sha256={f.name:sha(f) for f in code.glob('*.py')},adjudication_authorization_sha256=sha(run/'adjudication-authorization.json'),adjudication_changes_sha256=sha(run/'adjudication-changes.json'),raw_score='Exact actionable non-intent slot/native-decoded-value/status multiset; ignore unmentioned/dont_care for both arms, record them separately; Pydantic-native admission; capped/refused zero TP; no pipeline recovery or semantic/synonym repair',scope='corrective development: fresh OpenAI versus trajectory-verified reused SLM; common R3 adapter, not unmodified production provider; not independent human-reviewed test',fresh_v8_inference=False,planned_fresh_hosted_calls=220,amended_task_budget_USD=2,new_rerun_dispatch_ceiling_USD=.9,prepared_UTC=datetime.now(timezone.utc).isoformat())
    atomic(run/'protocol.json',protocol)
    print(json.dumps({'run':str(run),'protocol_sha256':sha(run/'protocol.json'),'reused_SLM_calls':220,'all_SLM_inputs_and_trajectories_unchanged':True,'paid_requests':0}))


if __name__=='__main__':prepare(Path(sys.argv[1]).resolve())
