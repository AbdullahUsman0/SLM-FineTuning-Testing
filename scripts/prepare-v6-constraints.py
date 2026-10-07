"""Freeze original-prompt unconstrained versus XGrammar v6 development evaluation."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from local_slm_lab.v5_eval import load_cases,load_schema,build_call_plan,digest,sha256,now
from local_slm_lab.v5_prompts import build_v5_extractor_instructions
from local_slm_lab.v6_constrained_provider import BACKEND,COMPILE_OPTIONS

def prepare():
    previous=ROOT/'evaluation/v6-prompt-audit-20261006'
    destination=ROOT/'evaluation/v6-constraints-20261007';destination.mkdir(exist_ok=False)
    protocol=json.loads((previous/'protocol.json').read_text())
    data=(previous/'natural.jsonl').read_bytes()
    assert sha256(previous/'natural.jsonl')==protocol['natural_sha256']
    (destination/'natural.jsonl').write_bytes(data)
    cases,_=load_cases(destination/'natural.jsonl',split='validation');plan=build_call_plan(cases,load_schema())
    assert len(cases)==32 and len(plan)==96
    assert digest([c['key'] for c in plan])==protocol['ordered_keys_sha256']
    assert digest([[c['key'],digest(c['context']),digest(c['expected'])] for c in plan])==protocol['context_gold_sha256']
    spec=importlib.util.spec_from_file_location('export',ROOT/'scripts/export-v5-output-schemas.py')
    export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
    schema=export.build_schemas()['extractor-result']
    previous_schema=ROOT/'evaluation/v5-structured-output/schemas/extractor-result.schema.json'
    assert schema==json.loads(previous_schema.read_text())
    (destination/'extractor-result.schema.json').write_bytes(previous_schema.read_bytes())
    prompt=build_v5_extractor_instructions()
    assert digest(prompt)==protocol['prompt_sha256']['original']
    (destination/'original-instructions.txt').write_text(prompt+'\n',encoding='utf-8',newline='\n')
    protocol.update(version='v6-constraints-20261007-1',frozen_before_new_inference_utc=now(),
        arms=['original','constrained'],prompt_sha256=dict.fromkeys(('original','constrained'),digest(prompt)),
        order='Original/constrained order alternates per scenario; serial batch one after equal unscored warmups',
        hypothesis='With the original prompt unchanged, masking grammar-invalid tokens improves structural validity without reducing semantic extraction quality.',
        comparison='Original versus constrained decoding in one session; same weights, prompt, user script, gold labels, validator and greedy settings.',
        backend=BACKEND,backend_dependencies={'xgrammar':'0.2.7','apache-tvm-ffi':'0.1.14.post1'},
        compile_options=COMPILE_OPTIONS,output_schema_sha256=sha256(previous_schema),
        constraints='Required keys, no extra keys, declared JSON types and slot-name enum. Fixed object-field order is an additional decoding restriction used because arbitrary-order mode admitted duplicate object keys in contract tests. No per-case semantic hints or retries. Inner JSON-text validity, duplicate slot IDs and semantic grounding remain application checks.',
        success_criteria='Target 96/96 schema-valid calls on each track. Strict and application-normalized F1 and rollout full-state correction accuracy must not fall versus same-session original control; valid unknown abstention must not fall; new unmentioned slots must not increase. Report all domain tradeoffs, paired bootstrap intervals and latency. This exploratory screen does not authorize automatic adoption.',
        limitations='Same agent-authored development conversations already inspected; independent human and unseen-case evaluation pending. One frozen decoding candidate after unscored implementation checks. Fixed key order may affect generation; grammar does not establish semantic correctness.',
        historical_references={track:{'path':f'research-checkpoints/v6-prompt-audit-20261006/results/original-{track}.json.gz',
            'sha256':sha256(ROOT/f'research-checkpoints/v6-prompt-audit-20261006/results/original-{track}.json.gz')} for track in ('natural_component','natural_rollout')},
        training_started=False,sealed_final_accessed=False,paid_apis_used=False)
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8',newline='\n')
    (destination/'README.md').write_text('# Frozen v6 generation-time schema study\n\n384 extractions compare original and XGrammar constrained decoding across the same 32 three-turn validation conversations and two context tracks. The original prompt is identical for both arms. Weights, user script, gold labels, strict validator and greedy settings are unchanged.\n\nThe grammar uses the existing exported schema, full model vocabulary and actual generation EOS IDs. XGrammar produces CPU token masks; the explicit torch_native kernel applies masks to CUDA logits, avoiding a Triton dependency on Windows. One fresh matcher per request. Object keys follow schema order; this restriction was selected during unscored contract checks, before this protocol was frozen. Duplicate slot IDs and JSON encoded inside candidate_value remain strict application checks. No postprocessing repair or retry.\n\nSee protocol.json for hashes, criteria and limitations. This is an exploratory development screen. No training, sealed final labels, paid APIs or default inference changes.\n',encoding='utf-8',newline='\n')
    print('Frozen 32 cases, 384 calls, identical prompts and exported structural schema.')

if __name__=='__main__':prepare()
