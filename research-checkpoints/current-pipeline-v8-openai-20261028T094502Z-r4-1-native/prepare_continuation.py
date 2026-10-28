"""Versioned renderer-only continuation; never repeats a paid dispatch."""
from copy import deepcopy
import json
from pathlib import Path
import sys

RUN=Path(__file__).resolve().parent
C=json.loads((RUN/'config.json').read_bytes());OLD=Path(C['resume_from']);CODE=Path(C['code'])
sys.path[:0]=[str(RUN/'offline-deps'),str(CODE)]
from controls import atomic,digest,sha
import pipeline as p
p.configure(RUN/'fpy',Path(C['repo']))
from forecasting_assistant.domain.schema import create_initial_state,load_schema


def main():
    assert not (RUN/'protocol.json').exists()
    prior=json.loads((OLD/'protocol.json').read_bytes())
    records=[json.loads(x) for x in (RUN/'calls.jsonl').read_bytes().splitlines()]
    saved={tuple(r['key']):r for r in records};assert len(saved)==len(records)==352
    assert sum(r['key'][1]=='openai' for r in records)==132
    slm=json.loads((OLD/'slm-reuse-verification.json').read_bytes())
    slm_states={tuple(r['key']):r['pipeline_state_sha256'] for r in slm['proof']}
    schema=load_schema();verified=[];renderer=[];checked_slm=0
    for stage in ('pilot','main'):
        cases=json.loads((RUN/(stage+'-cases.json')).read_bytes());assert digest(cases)==prior[stage+'_case_sha256']
        for arm in ('v8','openai'):
            for track in ('component','rollout'):
                for case in cases:
                    state=create_initial_state(schema)
                    for turn in case['turns']:
                        key=(stage,arm,track,case['scenario_id'],turn['turn_id'])
                        if key not in saved:break
                        if track=='component':state=p.restore(turn,schema)
                        else:p.intervention(state,turn.get('state_intervention'))
                        for t in state.turns:t.assistant_message=None
                        context=p.extraction_context(state,turn['message'],schema);record=saved[key]
                        assert record['current_input']==context and record['identity']==digest([prior['instructions'],context,turn['gold_extraction'],turn['expected_state_after']]),'Existing paid/local input drift'
                        state,trace=p.run_pipeline(state,turn['message'],schema,record,prior['response_schema'])
                        state_sha=digest(p.snapshot(state))
                        if arm=='v8':assert state_sha==slm_states[key];checked_slm+=1
                        for issue in trace['static_question_renderer_failures']:renderer.append({'key':list(key),**issue})
                        verified.append({'key':list(key),'identity':record['identity'],'source_record_sha256':digest(record),'pipeline_state_sha256':state_sha})
    assert len(verified)==352 and checked_slm==220 and renderer
    report={'passed':True,'existing_responses_replayed':352,'existing_paid_responses_replayed_without_dispatch':132,'SLM_unchanged_trajectories':220,'same_paid_request_namespace':prior['dispatch_namespace'],'same_model_inputs_prompt_gold_and_decoding':True,'original_blocked_run_preserved':str(OLD),'original_status':json.loads((OLD/'status.json').read_bytes()),'renderer_failures_handled':renderer,'proof':verified,'new_paid_requests':0}
    atomic(RUN/'renderer-resume-verification.json',report)
    slm['verified_under_renderer_continuation']=True;atomic(RUN/'slm-reuse-verification.json',slm)
    correction=json.loads((OLD/'interface-correction.json').read_bytes());correction['version']='r4.1-native-actionable-bounded-static-renderer';correction['changes'].append('Catch only static QuestionOutput length validation; emit bounded unscored fallback, removed before next model input')
    atomic(RUN/'interface-correction.json',correction)
    atomic(RUN/'local-technical-gate.json',{'passed':True,'interface_correction_sha256':sha(RUN/'interface-correction.json'),'SLM_reuse_verified':True,'resume_verified':True})
    protocol=deepcopy(prior);protocol.update(version='current-pipeline-comparison-6-r4.1',cohort='current-pipeline-dev20-20261028-r4.1-native',interface_version=correction['version'],interface_correction_sha256=sha(RUN/'interface-correction.json'),local_technical_gate_sha256=sha(RUN/'local-technical-gate.json'),slm_reuse_sha256=sha(RUN/'slm-reuse-verification.json'),renderer_resume_sha256=sha(RUN/'renderer-resume-verification.json'),runner_source_sha256={f.name:sha(f) for f in CODE.glob('*.py')},versioned_continuation_from=str(OLD),paid_request_namespace_preserved=True,question_generation='fixed-script static renderer only; catch 300-character QuestionOutput error with bounded fallback; no questions fed to model')
    atomic(RUN/'protocol.json',protocol)
    print(json.dumps({'run':str(RUN),'existing_paid_requests_preserved':132,'SLM_trajectories_verified':220,'renderer_failures_handled':len(renderer),'new_paid_requests':0,'protocol_sha256':sha(RUN/'protocol.json')}))


if __name__=='__main__':main()
