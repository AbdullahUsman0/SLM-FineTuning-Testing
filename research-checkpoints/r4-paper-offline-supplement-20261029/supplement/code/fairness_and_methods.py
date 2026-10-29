"""Audit existing public receipts; distinguish raw verification from narrative provenance."""
from pathlib import Path
import json
from common import C,OUT,E,dump,write,csv_write,sha


def verify_manifest(path,base,field):
    data=json.loads(path.read_bytes());checks=[]
    for name,entry in data[field].items():
        wanted=entry['sha256'] if isinstance(entry,dict) else entry
        target=base/name
        checks.append({'file':str(target),'expected_sha256':wanted,'present':target.is_file(),'actual_sha256':sha(target) if target.is_file() else None,'matches':target.is_file() and sha(target)==wanted})
    return {'manifest':str(path),'manifest_sha256':sha(path),'checks':checks,'all_match':all(x['matches'] for x in checks)}


def main():
    repo=Path(C['repo']);out=OUT/'fairness-and-methods';out.mkdir(exist_ok=True)
    protocol=json.loads((E/'protocol.json').read_bytes());runtime=json.loads((E/'local-runtime.json').read_bytes())
    fairness=[
      {'dimension':'model/base revision','local':'Qwen/Qwen3.5-2B @ '+protocol['base_revision'],'hosted':protocol['hosted_model']},
      {'dimension':'adapter','local':dump(protocol['adapter_sha256']),'hosted':'No task-specific adapter documented; hosted parameters/training not inspectable'},
      {'dimension':'training/development exposure','local':'Fine-tuned synthetic forecasting domain/annotation policy; related generator/templates; these public validation cases were previously inspected','hosted':'No task-specific fine-tuning documented; pretraining/data exposure unknown; these cases/prompts were inspected during study development'},
      {'dimension':'instructions','local':'Same frozen R3 augmented common instructions; SHA '+__import__('hashlib').sha256(protocol['instructions'].encode()).hexdigest(),'hosted':'Same instruction string; schema also enforced through hosted transport'},
      {'dimension':'schema and input','local':'Same 79-slot registry and visible response schema; same component inputs; own rollout state','hosted':'Same registry/schema/script; component identities verified; own rollout state'},
      {'dimension':'admission/values','local':'Native Pydantic candidate decoder once; no extra inner-JSON rejection; capped/refused excluded','hosted':'Identical native parser/one-decode policy'},
      {'dimension':'scoring','local':'Actionable non-intent slot/decoded-value/status exact multiset; ignore unmentioned/dont_care and report counts','hosted':'Identical rules; duplicate predictions retained; no synonym/paraphrase/numeric repair'},
      {'dimension':'decoding','local':dump(protocol['local_generation']),'hosted':dump(protocol['hosted_generation'])},
      {'dimension':'output budget','local':'1024 new tokens; greedy; thinking disabled','hosted':'2048 output tokens including reasoning; minimal reasoning; native Structured Outputs'},
      {'dimension':'runtime','local':dump(runtime),'hosted':'Pinned snapshot via direct api.openai.com; host/network transport; cannot inspect server hardware'},
      {'dimension':'fresh/reused','local':'202 main + 18 pilot responses reused from R3 only after identical input and state-trajectory proof; newly rescored','hosted':'202 main + 18 pilot responses fresh for R4; 132 already dispatched calls replayed on R4.1 continuation without redispatch'},
      {'dimension':'latency definition','local':'Original warm CUDA generation wall time, synchronized; load/warmup excluded; earlier session/thermal conditions retained','hosted':'Synchronous request wall time including network/provider generation; no automatic retries; no question-generation calls'},
      {'dimension':'pipeline/question context','local':'Current fpy plus bundled fixes; same reducer/recovery; fixed scripted turns; question text cleared before next input','hosted':'Same application path; R4.1 wrapper bounds unscored static-question length error for both; not an unmodified production-prompt benchmark'},
      {'dimension':'review/validity','local':'Agent-authored/inspected synthetic development; independent human review incomplete; no sealed final labels','hosted':'Same cohort and limitations; no general superiority/equivalence claim'}]
    csv_write(out/'fairness_audit.csv',fairness)
    legacy=repo/'research-checkpoints/policy-openai-slm-matched-20261008';grammar=repo/'research-checkpoints/v6-constraints-20261007'
    checks={}
    checks['historical_31_manifest']=verify_manifest(legacy/'checkpoint-manifest.json',legacy,'files')
    checks['configured_XGrammar_review']=verify_manifest(grammar/'review/independent-verification.json',grammar/'results','artifacts')
    docs=[repo/'evaluation/v5-structured-output/README.md',repo/'STRUCTURED_OUTPUT_RESEARCH_CHECKPOINT_2026-09-29.md',repo/'research-checkpoints/v7-dataset-preparation-20261008/DATASETS_AND_SLOT_FILLING_2026-10-08.md',repo/'research-checkpoints/2026-09-20-v3-validation/README.md']
    checks['document_provenance']=[{'path':str(p),'sha256':sha(p),'interpretation':'Document verified as present; historical performance numbers are not reverified from raw data merely by hashing this prose'} for p in docs]
    write(out/'receipt_verification.json',checks)
    matrix=[
      {'method':'A','configuration':'Prompt-only JSON, no retry','models_and_cohorts':'R4 v8 local greedy A-style: 20 matched development conversations, 202 main reused emissions; historical hosted A: 32 natural conversations, 96 logical calls/track (documented separately)','policy':'R4 common native/actionable policy differs from historical strict inner-JSON wire policy','verified_scope_here':'R4 404-call replay and source ZIP verified; historical A narrative/source provenance only','missing_or_limited':'No matched v8-vs-hosted A ablation; do not equate distinct prompts/cohorts'},
      {'method':'B','configuration':'One validation-guided retry','models_and_cohorts':'Historical gpt-5-mini-2025-08-07: 32 natural conversations plus separate 16-scenario coverage diagnostic','policy':'Original retry decisions/strict wire admission; documented post-hoc component rescoring is not fresh rollout','verified_scope_here':'Implementation and teacher report present; primary historical B raw completion receipt unavailable on GPU checkout','missing_or_limited':'No R4 v8 retry experiment; do not call this a complete matched B comparison'},
      {'method':'C','configuration':'Local XGrammar JSON-schema constrained decoding','models_and_cohorts':'Configured v6 Qwen3.5-2B: 32 three-turn conversations; original vs constrained, component/rollout; 384 extractions','policy':'Same local weights/prompt/greedy setting; xgrammar 0.2.7, fixed field order, torch_native mask; strict application checks retained','verified_scope_here':'Completion 384 calls and stored artifact hashes rechecked; exact receipt outcomes in receipt_verification.json','missing_or_limited':'No hosted XGrammar arm; no v8 C arm; configured backend result is not universal constrained-decoding evidence'},
      {'method':'D','configuration':'Slot-wise probes with aggregation','models_and_cohorts':'Documented hosted pilot: ONE weather conversation, three turns/track; 159 component and 237 rollout API calls','policy':'Required-probe/whole-extraction aggregation; not the full 32-conversation comparison','verified_scope_here':'Teacher-report provenance and implementation only; primary raw D receipt unavailable here','missing_or_limited':'Pilot not a full comparison; zero pilot F1 does not establish inherent slot-wise inferiority'},
      {'method':'E','configuration':'Deterministic rules; zero LLM calls','models_and_cohorts':'Documented 231-scenario synthetic validation: 717 extraction + 231 question calls; separate historical 32 natural conversations','policy':'Partial-coverage rules; exact non-intent scoring; 948 structural outputs includes question calls','verified_scope_here':'Checkpoint records report SHA and completion; 41MB full report absent here, so raw performance not newly reverified','missing_or_limited':'No model identity for E; not OpenAI-powered; no fresh R4 E comparison'},
      {'method':'F','configuration':'Hosted native Structured Outputs','models_and_cohorts':'R4 gpt-5-mini-2025-08-07: 20 conversations, 202 fresh main + 18 pilot calls; separate verified historical 31 conversations/186 calls','policy':'R4 native/actionable policy; historical strict comparison policy kept separate','verified_scope_here':'R4 sealed ZIP and all 404 score/state replays; historical checkpoint manifest checked','missing_or_limited':'R4 is one configured local-vs-hosted comparison, not six arms for both models'}]
    csv_write(out/'six_method_evidence_matrix.csv',matrix)
    md=['# Fairness and six-method coverage audit','', 'This is NOT a completed two-model-by-six-method factorial study. A-F describe method families across different cohorts/configurations. J (JSON mode) is an additional historical control, not a seventh row silently merged into A-F.','', '| Dimension | Local system | Hosted system |','|---|---|---|']
    for r in fairness:md.append('| '+r['dimension']+' | '+r['local'].replace('|','/')+' | '+r['hosted'].replace('|','/')+' |')
    md+=['','## A-F evidence matrix','','| Method | Configuration | Completed/documented scope | Verification and limits |','|---|---|---|---|']
    for r in matrix:md.append('| '+r['method']+' | '+r['configuration']+' | '+r['models_and_cohorts']+' | '+r['verified_scope_here']+'. '+r['missing_or_limited']+' |')
    md+=['','Raw receipts for the historical hosted A/B/D and full E rule result are not present in this checkout. Their documented scopes are retained with a lower evidence tier; this supplement does not manufacture independent verification. Supply those exact public receipts from the main PC if a full provenance recheck is needed.','', 'Historical 0.8B and 2B training runs differ in data, prompts, adapters and available raw evidence; they are not a controlled model-size experiment. The 48-case 2B-v3 comparison and older 0.8B claims must remain separate. No size conclusion is derived here.','', 'Distinct hosted schema enforcement and local greedy decoding remain confounds of configured-system comparison. Reused local timing and fresh hosted network timing are descriptive, not hardware/energy or general speed claims.']
    (out/'FAIRNESS_AND_SIX_METHODS.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(dump({'fairness_dimensions':len(fairness),'method_rows':6,'historical_manifest_all_match':checks['historical_31_manifest']['all_match'],'XGrammar_artifacts_all_match':checks['configured_XGrammar_review']['all_match'],'factorial_completed':False,'historical_raw_receipt_gaps':['A','B','D','full E']}))


if __name__=='__main__':main()
