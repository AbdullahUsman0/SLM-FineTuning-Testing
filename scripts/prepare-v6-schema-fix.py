"""Freeze a format-only follow-up; original prompt retained as historical reference."""
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_slm_lab.v5_eval import build_call_plan,digest,load_cases,load_schema,now,sha256
from local_slm_lab.v6_prompt_revision import build_revised_instructions
from local_slm_lab.v6_schema_fix import build_schema_fixed_instructions,SCHEMA_FIX

def prepare():
    previous=ROOT/'evaluation/v6-prompt-audit-20261006'
    destination=ROOT/'evaluation/v6-schema-fix-20261006'
    destination.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((previous/'protocol.json').read_text())
    source=previous/'natural.jsonl';assert sha256(source)==protocol['natural_sha256']
    (destination/'natural.jsonl').write_bytes(source.read_bytes())
    cases,_=load_cases(source,split='validation');plan=build_call_plan(cases,load_schema())
    assert len(cases)==32 and len(plan)==96
    assert digest([c['key'] for c in plan])==protocol['ordered_keys_sha256']
    assert digest([[c['key'],digest(c['context']),digest(c['expected'])] for c in plan])==protocol['context_gold_sha256']
    prompts={'revised':build_revised_instructions(),'schema_fixed':build_schema_fixed_instructions()}
    assert prompts['schema_fixed']==prompts['revised']+SCHEMA_FIX
    assert digest(prompts['revised'])==protocol['prompt_sha256']['revised']
    reference_root=ROOT/'research-checkpoints/v6-prompt-audit-20261006/results'
    references={}
    for arm in ('original','revised'):
        for track in ('natural_component','natural_rollout'):
            path=reference_root/f'{arm}-{track}.json.gz'
            report=json.loads(gzip.decompress(path.read_bytes()))
            assert report['status']=='complete' and len(report['records'])==96
            references[f'{arm}-{track}']={'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)}
    protocol.update(version='v6-schema-fix-20261006-1',frozen_before_new_inference_utc=now(),
        arms=['revised','schema_fixed'],prompt_sha256={arm:digest(text) for arm,text in prompts.items()},
        historical_references=references,
        hypothesis='Appending only an explicit field whitelist and separate-update construction rules reduces extra-field and merged-update contract failures without undoing completeness and abstention gains.',
        comparison='Revised versus schema_fixed in the same session, matched labels and gold component contexts; original prompt is a clearly identified historical reference, with no direct cross-session latency claim.',
        success_criteria='Schema-fixed validity at least original reference (93/96) on each track; maintain 32/32 valid unknown abstentions and no observed new unmentioned slots; normalized F1 and full-state accuracy must not fall versus same-session revised control. Report per-domain tradeoffs, raw typed corrections, latency, bootstrap intervals and every failure. This is a development screen, not deployment approval.',
        limitations='One format-only candidate designed after earlier development outputs; no unseen-case or independent human-review claim. Revised control is rerun; compare its raw outputs to the previous run to disclose any repeat drift.',
        training_started=False,sealed_final_accessed=False,paid_apis_used=False)
    for arm,text in prompts.items():(destination/f'{arm}-instructions.txt').write_text(text+'\n',encoding='utf-8',newline='\n')
    (destination/'schema-only-addition.txt').write_text(SCHEMA_FIX+'\n',encoding='utf-8',newline='\n')
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8',newline='\n')
    (destination/'README.md').write_text('# Focused v6 output-schema prompt test\n\nOne candidate appends only a final output-contract reminder to the previous revised prompt. Its completeness, evidence, correction, unknown-information and illustrative content remains byte-identical as a prefix. The unchanged 32 development conversations and labels are frozen in natural.jsonl.\n\n384 new calls compare revised and schema_fixed in one session on gold-context component and own-state rollout tracks. The original prompt is retained as a historical reference; its latency is not treated as a same-session measurement. Revised-repeat equivalence will be checked. See protocol.json for hashes, success criteria and limits.\n\nNo training, parsing repair, retries, paid APIs or sealed final labels. Default chat behavior is not changed by this test.\n',encoding='utf-8',newline='\n')
    print(json.dumps({'study':str(destination),'calls':384,'arms':protocol['arms'],'schema_only_addition_characters':len(SCHEMA_FIX)},indent=2))

if __name__=='__main__':prepare()
