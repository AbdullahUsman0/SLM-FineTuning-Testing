"""Offline token preflight, prospective protocol and unapproved budget request."""
import csv
import gzip
import importlib.metadata as md
import json
import math
from pathlib import Path
import statistics
import sys
import runtime as rt
from shared_policy import SharedPolicy,VERSION
from providers import hosted_request,worst_attempt_cost,OPENAI_MODEL,OPENAI_OUTPUT_CAP,API_INPUT_BYTES_CAP,PRICES
from evaluate import plans


def stats(values):return {'minimum':min(values),'maximum':max(values),'mean':statistics.mean(values),'median':statistics.median(values)}


def main():
    intake=rt.read(rt.CHECKPOINT/'workspace-intake.json');private=Path(intake['private']);repo=Path(intake['repo'])
    candidate=private/'candidate/conversations.jsonl';cases=[rt.strict_json(line) for line in candidate.read_text(encoding='utf-8').splitlines()]
    policy=SharedPolicy();schema=rt.load_schema();plan=plans(cases,schema,policy)
    sys.path.insert(0,str(private/'token-count-deps'))
    import tiktoken
    tokenizer=tiktoken.encoding_for_model(OPENAI_MODEL)
    import os
    os.environ.update(HF_HOME=r'D:\SLM\hf-cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from transformers import AutoTokenizer
    local=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B',revision=rt.BASE_REVISION,local_files_only=True)
    input_tokens=[];wire_bytes=[];output_tokens=[];qwen_outputs=[];qwen_inputs=[]
    for case in cases:
        for call in plan[case['scenario_id']]:
            request=hosted_request(call['turn']['message'],call['before'],schema,policy)
            serialized=rt.dumps(request);wire_bytes.append(len(serialized.encode('utf-8')))
            # Tokenize the complete serialized request including schema for conservative planning; exact API accounting remains usage-based.
            input_tokens.append(len(tokenizer.encode(serialized,disallowed_special=())))
            gold_wire=rt.dumps(call['turn']['gold_extraction'])
            output_tokens.append(len(tokenizer.encode(gold_wire,disallowed_special=())))
            qwen_outputs.append(len(local.encode(gold_wire,add_special_tokens=False)))
            text=local.apply_chat_template(request['input'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            qwen_inputs.append(len(local.encode(text,add_special_tokens=False)))
    assert max(wire_bytes)<=API_INPUT_BYTES_CAP
    assert max(qwen_outputs)<1024
    old=json.loads(gzip.decompress((repo/'research-checkpoints/policy-openai-slm-matched-20261008/hosted-source/component-report.json.gz').read_bytes()))
    old_reasoning=[r.get('reasoning_tokens',0) for r in old['records']]
    historical_reasoning_mean=statistics.mean(old_reasoning)
    hosted_calls=len(input_tokens)*2
    planned_output=min(OPENAI_OUTPUT_CAP,statistics.mean(output_tokens)*1.5+historical_reasoning_mean+128)
    input_allowance=statistics.mean(input_tokens)+128
    estimate=hosted_calls*(input_allowance*PRICES['input_per_million']+planned_output*PRICES['output_per_million'])/1e6
    max_attempts=hosted_calls+math.ceil(hosted_calls*.10)+1
    upper=max_attempts*worst_attempt_cost()
    suggested=math.ceil(upper/5)*5
    schedule=[]
    for number in range(12):
        group=f'authored-family-{number:02d}'
        for arm in (['v6','v7'] if number%2==0 else ['v7','v6'])+['openai']:
            schedule.append({'group':group,'arm':arm})
    protocol={'protocol_version':'fresh-extraction-matched-20261012-1','document_type':'prospective experiment protocol',
        'research_task':'Forecasting-requirement extraction into a tabular time-series 79-slot schema, not actual forecasting accuracy.',
        'upstream_source_commit':intake['source_commit'],'models':rt.read(rt.CHECKPOINT/'models-freeze.json'),
        'candidate_sha256':rt.sha(candidate),'cohort_size':240,'calls_per_track_per_model':len(input_tokens),'planned_scored_calls':len(input_tokens)*6,
        'domain_quotas':{'weather':60,'inflation':60,'stocks':60,'crypto':60},
        'language_per_domain':{'en':48,'ur':6,'mixed':6},'balance_differs_from_training':'Intentionally equal domains; 80/10/10 language quota. Not the earlier training domain mixture.',
        'synthetic_authorship':'Agent-authored parameterized synthetic candidate; independent human review required. Not independent human-authored requests.',
        'template_clustering':'Twelve authored macro-template groups, with related variants/translations across domains. Shared generic clauses across groups remain a limitation.',
        'primary_pair':'v7 minus OpenAI','primary_endpoint':'Exact non-intent slot/value/status micro F1 in rollout',
        'primary_hypothesis':{'null':'The cohort-specific exact rollout F1 difference v7 minus OpenAI is zero.',
            'alternative':'The cohort-specific difference is nonzero (two-sided).','decision':'Prespecified paired percentile 95% interval; no post-hoc equivalence/noninferiority claim.'},
        'secondary_pairs':['v6 minus OpenAI','v7 minus v6'],'secondary_endpoints':[
            'component exact F1','precision','recall','slot-name F1','raw JSON validity','schema validity','admission','exact evidence grounding',
            'extra current-turn slot flags','duplicate slots','correction flag and exact replacement','unrelated-state preservation',
            'complete state after correction','unknown semantic/metadata preservation and intent','complete conversation semantic state',
            'original generation latency','tokens and hosted cost','numeric-canonical and status-free diagnostics'],
        'tracks':{'component':'Exact same gold-replayed prior state, schema and logical instructions for all arms.',
            'rollout':'Fixed reviewed user scripts; each model uses its own accepted accumulated state.'},
        'seeded_state_corruption_in_primary':False,'bootstrap':{'resamples':10000,'seed':42,'unit':'authored source/template-family group','micro_F1_recomputed':True},
        'subgroup_analysis':'Domain/language/template-family secondary exploratory; no universal conclusions or multiplicity-adjusted claims.',
        'policy':{'version':VERSION,'reference':policy.reference.fingerprint,'evidence':'Exact nonblank current-message substring, case and whitespace sensitive.',
            'candidate_wire':'String containing JSON text; historical ordinary-string acceptance retained. Exactly one decoding pass; explicitly encoded literal-looking strings remain strings.',
            'invalid_calls':'Readable emitted tuples count as FP, every gold tuple as FN, zero TP on rejected calls; no silent repair, defaults, gold-conditioned postprocessing or semantic retries.',
            'application':'Pinned fpy reducer; reject contract/evidence failures as a whole; preserve prior slots on rejection and append actual user history.'},
        'local':{'device':'CUDA','base_precision':'bfloat16','PEFT_parameter_precision':'Default FP32 LoRA parameters preserved, same verified PEFT loader for both adapters',
            'greedy':True,'enable_thinking':False,'seed':42,'repetition_penalty':1.0,'max_new_tokens':1024,'batch_size':1,
            'unscored_warmups':'One per actual GPU model-loading session; recorded separately; none count in quality metrics.'},
        'hosted':{'model':OPENAI_MODEL,'API':'Responses','structured_outputs':'Strict closed shared schema','reasoning':'minimal, previously verified for this exact snapshot; current access/settings checked before scoring',
            'max_output_tokens_including_reasoning':OPENAI_OUTPUT_CAP,'input_request_bytes_cap':API_INPUT_BYTES_CAP,
            'concurrency':1,'SDK_retries':0,'max_attempts_per_call':2,'global_retry_allowance':math.ceil(hosted_calls*.10),
            'maximum_attempts':max_attempts,'unscored_warmups':1,'pricing':PRICES,'pricing_verified_at':rt.now(),
            'pricing_source':'https://developers.openai.com/api/docs/models/gpt-5-mini',
            'deprecated_snapshot':'Official catalog marks snapshot deprecated; availability must be confirmed. No automatic substitution.',
            'budget':'Conservative reservation persisted before every attempt. Unknown transport outcomes retain worst-case reservation; completed usage includes reasoning and cached tokens.'},
        'schedule':schedule,'schedule_limitation':'Local order alternates within 20-conversation family blocks; hosted requests follow the paired GPU block. Hardware, transport, reasoning and output constraints differ.',
        'review_required':True,'budget_approval_required':True,'final_labels_enter_training_or_selection':False,
        'stopping_rule':'One reviewed frozen cohort, one planned matched experiment, independent raw verification and documented failure analysis. No additional training or outcome-driven cohort expansion.'}
    rt.atomic(rt.CHECKPOINT/'experiment-protocol.json',protocol,exclusive=True)
    preflight={'status':'offline_candidate_preflight_passed_not_final_execution','candidate_sha256':rt.sha(candidate),
        'tiktoken_version':md.version('tiktoken'),'encoding':tokenizer.name,'token_counter_installed_in_isolated_directory':True,
        'API_complete_request_token_estimate':stats(input_tokens),'API_complete_request_UTF8_bytes':stats(wire_bytes),
        'gold_wire_OpenAI_tokens':stats(output_tokens),'gold_wire_Qwen_tokens':stats(qwen_outputs),'Qwen_prompt_tokens':stats(qwen_inputs),
        'generation_cap_change':False,'local_gold_outputs_over_1024':sum(n>=1024 for n in qwen_outputs),'model_calls_on_cohort':0}
    rt.atomic(rt.CHECKPOINT/'token-preflight.json',preflight,exclusive=True)
    request={'status':'awaiting_user_approved_budget_for_this_specific_experiment','protocol_sha256':rt.sha(rt.CHECKPOINT/'experiment-protocol.json'),
        'hosted_scored_calls':hosted_calls,'paid_unscored_warmups':1,'maximum_total_attempts_with_transport_retries':max_attempts,
        'planning_estimate_usd':estimate,'planning_output_tokens_per_call':planned_output,'historical_minimal_reasoning_mean_tokens':historical_reasoning_mean,
        'conservative_upper_bound_usd':upper,'recommended_cap_usd':suggested,'earlier_budget_does_not_authorize_this_new_study':True,
        'assumptions':['No cache discount in upper bound. UTF-8 request bytes bound input tokens plus 1,024 overhead allowance.',
            'Output cap includes billed reasoning. Retry allowance is at most 10% of planned calls and one retry per call.',
            'Estimate uses offline complete-request and proposed-output token lengths plus historical minimal-reasoning usage; it is not new measured hosted output usage.'],
        'pricing_source':'https://developers.openai.com/api/docs/models/gpt-5-mini','paid_calls_made':0}
    rt.atomic(rt.CHECKPOINT/'budget-request.json',request,exclusive=True)
    rt.atomic(private/'review/budget-approval.template.json',{'status':'pending','approved_by':None,'authorization_source':None,
        'protocol_sha256':request['protocol_sha256'],'cap_usd':suggested},exclusive=True)
    policy_record={'version':VERSION,'reference_policy_fingerprint':policy.reference.fingerprint,
        'new_study_only':True,'historical_files_modified':False,'exact_evidence_hard_gate':True,
        'structural_rules_reference':'Duplicate keys, nonfinite including overflow, closed fields, enums and duplicate-slot rejection from pinned ComparisonPolicy',
        'candidate_decoding':'Exactly once, followed by model_construct for already validated values',
        'parser_dependencies':rt.read(rt.CHECKPOINT/'runtime-lock.json')['parser_dependencies']}
    rt.atomic(rt.CHECKPOINT/'policy/policy.json',policy_record,exclusive=True)
    rt.atomic(rt.CHECKPOINT/'policy/extractor-result.schema.json',policy.schema,exclusive=True)
    (rt.CHECKPOINT/'policy/extractor-instructions.txt').write_text(rt.instructions(),encoding='utf-8',newline='\n')
    print(rt.dumps({'planned_scored_calls':len(input_tokens)*6,'hosted_calls':hosted_calls,'estimate_usd':estimate,
        'upper_bound_usd':upper,'proposed_budget_usd':suggested,'max_gold_Qwen_tokens':max(qwen_outputs),'max_API_payload_bytes':max(wire_bytes)}),flush=True)


if __name__=='__main__':main()
