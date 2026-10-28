"""Fresh LOCAL-ONLY positive controls on the pre-existing disjoint pilot set."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
from controls import append,atomic,dumps,sha,strict_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--mode',choices=['legacy','explicit'],required=True);args=parser.parse_args()
    config=json.loads(args.config.read_bytes());run=Path(config['run']);repo=Path(config['repo']);prior=Path(config['prior_run']);mode=args.mode
    os.environ['HF_HOME']=r'D:\SLM\hf-cache';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.pop('OPENAI_API_KEY',None)
    import pipeline as p
    p.configure(Path(config['legacy_fpy'] if mode=='legacy' else config['fpy']),repo)
    from forecasting_assistant.domain.schema import load_schema
    from forecasting_assistant.application.state_reducer import apply_extraction
    from local_slm_lab.v5_prompts import build_v5_extractor_input
    from local_slm_lab.v5_provider import V5TransformersProvider
    from local_slm_lab.peft_provider import PeftInferenceConfig
    schema=load_schema();protocol=json.loads((prior/'protocol.json').read_bytes())
    for name,wanted in protocol['adapter_sha256'].items():assert sha(Path(protocol['adapter'])/name)==wanted
    training=(repo/'corpus-v8/v8-final-20261018-r1/extractor-instructions.txt').read_text(encoding='utf-8')
    if mode=='legacy':instructions=training
    else:
        from forecasting_assistant.prompts.extractor import build_extractor_instructions
        instructions=build_extractor_instructions()+'\n\nExplicit common extraction interface and unchanged annotation policy:\n'+training+'\nThe response_schema describes the OUTPUT only. Priority rows with aspect/dimension/question are INPUT context, not an output format. Do not copy those rows as updates. Return every update with exactly slot_id, candidate_value, status, confidence, evidence_text. Both confidence fields must be between 0 and 1.\n'
    atomic(run/(mode+'-probe-freeze.json'),{'mode':mode,'instructions':instructions,'pilot_case_SHA256':sha(prior/'pilot-cases.json'),'base_revision':protocol['base_revision'],'adapter_sha256':protocol['adapter_sha256'],'fpy':str(config['legacy_fpy'] if mode=='legacy' else config['fpy']),'scope':'LOCAL-only diagnostic; pilot cases excluded from main; not a replacement main result'})
    import torch
    torch.set_num_threads(4)
    provider=V5TransformersProvider(PeftInferenceConfig(base_model=protocol['base_model'],revision=protocol['base_revision'],variant='custom',adapter_path=Path(protocol['adapter']),device='cuda',dtype='bfloat16',max_new_tokens=1024),schema)
    assert provider._model.active_adapters and all(x.device.type=='cuda' for x in provider._model.parameters())
    cases=json.loads((prior/'pilot-cases.json').read_bytes());records=[]
    path=run/(mode+'-probe-calls.jsonl')
    if path.exists():raise FileExistsError('Preserve the completed probe, do not overwrite')
    for case in cases:
        for turn in case['turns']:
            state=p.restore(turn,schema)
            context=build_v5_extractor_input(turn['message'],state,schema) if mode=='legacy' else p.extraction_context(state,turn['message'],schema)
            torch.cuda.synchronize();start=time.perf_counter();raw,input_tokens,output_tokens,capped=provider._generate_raw(instructions,context);torch.cuda.synchronize()
            record={'key':['local_probe',mode,case['scenario_id'],turn['turn_id']],'raw_output':raw,'current_input':context,'prompt_tokens':input_tokens,'completion_tokens':output_tokens,'latency_seconds':time.perf_counter()-start,'capped':capped,'refused':False}
            record.update(p.raw_score(record,turn,p.wire_schema()))
            append(path,record);records.append(record)
            print(dumps({'mode':mode,'completed':len(records),'schema_valid':record['schema_valid'],'raw_contract_admitted':record['raw_contract_admitted'],'confusion':record['confusion']}),flush=True)
    summary={'mode':mode,'calls':len(records),'exact':p.prf(records),'json_valid':sum(r['json_valid'] for r in records),'schema_valid':sum(r['schema_valid'] for r in records),'raw_contract_admitted':sum(r['raw_contract_admitted'] for r in records),'positive_turns_admitted':sum(r['raw_contract_admitted'] and r['confusion'][0]>0 for r in records),'new_paid_requests':0,'scope':'LOCAL-only disjoint pilot diagnostic, not main result','instructions_SHA256':sha(run/(mode+'-probe-freeze.json'))}
    atomic(run/(mode+'-probe-summary.json'),summary);print(dumps(summary),flush=True)


if __name__=='__main__':main()
