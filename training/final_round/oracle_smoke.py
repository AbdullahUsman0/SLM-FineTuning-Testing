"""CPU-only oracle fixture for evaluator integrity. NOT a model-quality measurement."""
import argparse,contextlib,io,json,sys,types
from pathlib import Path
from data_tools import CORPUS,rows,use_fpy,dumps,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--fpy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();use_fpy(a.fpy)
    from evaluate_development import select_cases,main as evaluate
    selected={c['scenario_id'] for c in select_cases(list(rows(CORPUS/'splits/validation.jsonl.gz')))}
    mapping={e['messages'][1]['content']:e['messages'][2]['content'] for e in rows(CORPUS/'sft/validation.jsonl.gz') if e['metadata']['scenario_id'] in selected}
    fake_torch=types.ModuleType('torch');fake_torch.__version__='CPU_ORACLE_FIXTURE_NOT_GPU'
    fake_torch.cuda=types.SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,synchronize=lambda:None,get_device_name=lambda n:'CPU oracle fixture NOT GPU')
    fake_torch.set_num_threads=lambda n:None
    fake_peft=types.ModuleType('local_slm_lab.peft_provider');fake_peft.PeftInferenceConfig=lambda **kw:types.SimpleNamespace(**kw)
    fake_v5=types.ModuleType('local_slm_lab.v5_provider');calls=[]
    class Oracle:
        def __init__(self,config,schema):
            self._torch=fake_torch;self._model=types.SimpleNamespace(active_adapters=['CPU_ORACLE_FIXTURE'],config=types.SimpleNamespace(_commit_hash=config.revision),parameters=lambda:[types.SimpleNamespace(device='CPU_ORACLE_FIXTURE')]);self.chat_template_sha256='CPU_ORACLE_FIXTURE'
        def _generate_raw(self,system,user):
            calls.append(user)
            raw=mapping.get(user)
            if raw is None:
                if 'Forecast air temperature for the next seven days.' not in user:raise ValueError('Oracle input differs from exact SFT gold context')
                raw='{"intent":"create_forecast","intent_confidence":1,"updates":[],"correction_detected":false,"unsupported_claims":[]}'
            return raw,1,1,False
    fake_v5.V5TransformersProvider=Oracle
    sys.modules.update({'torch':fake_torch,'local_slm_lab.peft_provider':fake_peft,'local_slm_lab.v5_provider':fake_v5})
    adapter=a.output/'ORACLE_NOT_ADAPTER';adapter.mkdir(parents=True,exist_ok=False)
    for name in ['adapter_model.safetensors','adapter_config.json']:(adapter/name).write_text('CPU ORACLE FIXTURE NOT REAL WEIGHTS')
    sys.argv=['evaluate_development.py','--fpy',str(a.fpy),'--adapter',str(adapter),'--output',str(a.output/'oracle-journal')]
    with contextlib.redirect_stdout(io.StringIO()):evaluate()
    first_count=len(calls)
    with contextlib.redirect_stdout(io.StringIO()):evaluate()
    if len(calls)!=first_count:raise AssertionError('Resume repeated generation')
    summary=json.loads((a.output/'oracle-journal/summary.json').read_bytes())
    for track,result in summary['tracks'].items():
        assert result['nonintent_exact']['f1']==1.,(track,result)
        assert result['complete_state_turns']==result['calls'],(track,result)
        assert result['correct_correction_calls']==result['correction_calls']
        assert result['unknown_successful_empty_calls']==result['unknown_turns']
    receipt={'status':'pass','fixture_only':True,'GPU_model_inference_performed':False,'training_performed':False,
        'oracle_F1_is_not_a_model_result':True,'both_tracks_gold_context_and_state_replay':'exact',
        'resume_without_repeat_generation':True,'scored_calls':sum(r['calls'] for r in summary['tracks'].values())}
    atomic(a.output/'oracle-test-receipt.json',receipt);print(dumps(receipt))

if __name__=='__main__':main()
