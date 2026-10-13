"""Final freeze can be created only after explicit independent review and budget approval."""
import argparse
import csv
from datetime import datetime,timezone
from pathlib import Path
import math
import runtime as rt
from shared_policy import SharedPolicy,VERSION
from providers import OPENAI_MODEL,OPENAI_OUTPUT_CAP,API_INPUT_BYTES_CAP,worst_attempt_cost


def validate_human_rows(rows,expected_keys,candidate_hash,key_fields):
    keys=[tuple(str(row[field]) for field in key_fields) for row in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected_keys):raise ValueError('Human review inventory missing or duplicated')
    for row in rows:
        if row.get('candidate_sha256')!=candidate_hash:raise ValueError('Review is not bound to this exact candidate')
        if row.get('decision') not in {'approve','approved','keep'}:raise ValueError('Human review/adjudication remains pending')
        if not row.get('reviewer') or any(word in row['reviewer'].casefold() for word in ['agent','codex','chatgpt']):
            raise ValueError('Independent human reviewer identity required')
        date=datetime.fromisoformat(row.get('reviewed_at_utc','').replace('Z','+00:00'))
        if date.tzinfo is None or date>datetime.now(timezone.utc):raise ValueError('Valid past UTC review date required')
    return rows


def code_hashes():
    return {p.relative_to(rt.CHECKPOINT).as_posix():rt.sha(p) for root in ('code','policy','tests')
            for p in sorted((rt.CHECKPOINT/root).rglob('*')) if p.is_file() and p.suffix in {'.py','.json','.md'}}


def verify_final(path):
    rt.require_study_execution()
    freeze=rt.read(path)
    if freeze['status']!='independently_reviewed_and_registered':raise ValueError('Final review/registration is incomplete')
    identity=freeze['identity']
    if freeze['fingerprint']!=rt.digest(identity):raise ValueError('Final freeze fingerprint mismatch')
    if identity['source_hashes']!=code_hashes():raise ValueError('Frozen study implementation/policy changed')
    for key,name in (('runtime_lock','runtime-lock.json'),('model_freeze','models-freeze.json'),('protocol','experiment-protocol.json'),('environment','environment.json'),('execution_decision','EXECUTION_DECISION.json')):
        if identity[key]!=rt.sha(rt.CHECKPOINT/name):raise ValueError('Frozen dependency/model/protocol identity changed')
    for name,expected in identity['private_files'].items():
        if rt.sha(Path(freeze['private'])/name)!=expected:raise ValueError('Reviewed cohort or adjudication changed')
    rt.verify_sources()
    return freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=rt.read(args.config);private=Path(config['private'])
    rt.require_study_execution()
    rt.verify_sources();rt.verify_models(config)
    candidate=private/'candidate/conversations.jsonl';candidate_hash=rt.sha(candidate)
    cases=[rt.strict_json(line) for line in candidate.read_text(encoding='utf-8').splitlines()]
    with (private/'review/review-sheet.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    validate_human_rows(rows,{(c['scenario_id'],) for c in cases},candidate_hash,['scenario_id'])
    rubric_hash=rt.sha(rt.CHECKPOINT/'policy/ANNOTATION_RUBRIC.md')
    review=rt.read(private/'review/review-manifest.json')
    if review['candidate_sha256']!=candidate_hash or review['annotation_rubric_sha256']!=rubric_hash:
        raise ValueError('Candidate/rubric changed; renewed independent review binding is required')
    overlap=rt.read(private/'review/overlap-audit.json')
    if overlap['candidate_sha256']!=candidate_hash:raise ValueError('Overlap audit is stale')
    if overlap['candidate_normalized_reference_matches']:
        raise ValueError('Exact/normalized training/development matches require a revised candidate and renewed audit/review')
    queue=[rt.strict_json(line) for line in (private/'review/overlap-queue.jsonl').read_text(encoding='utf-8').splitlines()]
    with (private/'review/overlap-decisions.csv').open(encoding='utf-8-sig',newline='') as stream:decisions=list(csv.DictReader(stream))
    validate_human_rows(decisions,{(q['scenario_id'],str(q['turn'])) for q in queue},candidate_hash,['scenario_id','turn'])
    if any(not row.get('rationale') for row in decisions):raise ValueError('Near-overlap keep decisions need human rationale')
    budget=rt.read(private/'review/budget-approval.json')
    protocol=rt.read(rt.CHECKPOINT/'experiment-protocol.json')
    if budget.get('status')!='approved' or not budget.get('approved_by') or not budget.get('authorization_source'):
        raise ValueError('An actual user-approved API budget is required')
    if budget.get('protocol_sha256')!=rt.sha(rt.CHECKPOINT/'experiment-protocol.json'):
        raise ValueError('Budget approval belongs to a different experiment')
    if type(budget.get('cap_usd')) not in (int,float) or not math.isfinite(budget['cap_usd']) or budget['cap_usd']<=0:
        raise ValueError('Positive finite budget cap required')
    smoke=rt.read(rt.CHECKPOINT/'GPU-SMOKE.json')
    if smoke['status']!='GPU_readiness_verified':raise ValueError('GPU smoke checks incomplete')
    policy=SharedPolicy();schema=rt.load_schema()
    for case in cases:
        state=rt.create_initial_state(schema)
        for index,turn in enumerate(case['turns'],1):
            admission=policy.parse(rt.dumps(turn['gold_extraction']),turn['message'])
            state,accepted,error=policy.advance(state,admission,schema,index,turn['message'])
            if not accepted:raise ValueError('Reviewed gold fails common policy/reducer')
            rt.append_user(state,turn['message'])
    names=['candidate/conversations.jsonl','candidate/manifest.json','review/review-sheet.csv',
        'review/review-manifest.json','review/overlap-audit.json','review/overlap-queue.jsonl',
        'review/overlap-decisions.csv','review/budget-approval.json','preparation-code/make_candidate.py']
    identity={'source_hashes':code_hashes(),'runtime_lock':rt.sha(rt.CHECKPOINT/'runtime-lock.json'),
        'model_freeze':rt.sha(rt.CHECKPOINT/'models-freeze.json'),'protocol':rt.sha(rt.CHECKPOINT/'experiment-protocol.json'),
        'environment':rt.sha(rt.CHECKPOINT/'environment.json'),
        'execution_decision':rt.sha(rt.CHECKPOINT/'EXECUTION_DECISION.json'),
        'private_files':{name:rt.sha(private/name) for name in names},'prompt_sha256':rt.digest(rt.instructions()),
        'policy_version':VERSION,'candidate_sha256':candidate_hash}
    rt.atomic(private/'final-freeze.json',{'status':'independently_reviewed_and_registered','identity':identity,
        'fingerprint':rt.digest(identity),'private':str(private),'frozen_at':rt.now(),'final_labels_enter_training':False},exclusive=True)
    print('Final cohort and complete experiment registered; no model calls made by this command.',flush=True)


if __name__=='__main__':main()
