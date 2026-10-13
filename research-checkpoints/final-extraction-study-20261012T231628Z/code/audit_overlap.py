"""Exact/normalized and lexical-near overlap audit on explicitly allowed development sources."""
from collections import Counter,defaultdict
import csv
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile
import runtime as rt


def normalized(message):return ' '.join(message.casefold().split())


def masked(message):
    value=normalized(message)
    value=re.sub(r'https?://\S+','SOURCE',value)
    value=re.sub(r'\b[\w.-]+\.(?:csv|parquet|json|xlsx)\b','FILE',value)
    value=re.sub(r'\b\d+(?:\.\d+)?\b','NUMBER',value)
    return re.sub(r'\b(?:measure|logged_at|copperleaf)[\w_]*\b','IDENTIFIER',value)


def shingles(message):
    words=re.findall(r'\w+',masked(message));return {tuple(words[i:i+3]) for i in range(max(0,len(words)-2))}


def historical_messages(value):
    for case in value:
        for turn in case.get('turns',[]):
            message=turn.get('message',turn.get('user_message'))
            if isinstance(message,str):yield message


def main():
    intake=rt.read(rt.CHECKPOINT/'workspace-intake.json');repo=Path(intake['repo']);private=Path(intake['private'])
    sources=[];utterances=[]
    # Explicit allowlist. No sealed/final/test split is traversed or opened.
    for folder in ('corpus','corpus-v2','corpus-v3','corpus-v5/v5-20260919-r1','corpus-v6/v6-20261002-r4'):
        for split in ('train','validation'):
            for suffix in ('.jsonl','.jsonl.gz'):
                path=repo/folder/'splits'/(split+suffix)
                if path.exists():sources.append(path)
    sources += [rt.BUNDLE/'corpus-v7/v7-20261009-r8-dev/splits'/f'{split}.jsonl.gz' for split in ('train','validation')]
    inventory=[]
    for path in sources:
        assert path.name in {'train.jsonl','validation.jsonl','train.jsonl.gz','validation.jsonl.gz'}
        assert not {'sealed','final','test'} & {p.lower() for p in path.parts}
        opener=gzip.open if path.suffix=='.gz' else open
        count=0
        with opener(path,'rt',encoding='utf-8-sig') as stream:
            for case in map(json.loads,stream):
                for message in historical_messages([case]):
                    utterances.append((str(path),message));count+=1
        inventory.append({'path':str(path),'sha256':rt.sha(path),'utterances':count,'scope':'accessible training/development only'})
    # Audit actual v6 training inputs as well as source trajectories.
    v6=Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v6-20261003T113807Z')
    manifest=rt.read(v6/'full/run-manifest.json')
    for split in ('train','validation'):
        path=Path(manifest['inputs'][split]['path']);assert path.name==split+'.jsonl'
        assert rt.sha(path)==manifest['inputs'][split]['sha256']
        count=0
        with path.open(encoding='utf-8') as stream:
            for row in map(json.loads,stream):
                for message in row['messages']:
                    if message['role']=='user':
                        try: text=json.loads(message['content'])['current_message']
                        except (ValueError,KeyError,TypeError):continue
                        utterances.append((str(path),text));count+=1
        inventory.append({'path':str(path),'sha256':rt.sha(path),'utterances':count,'scope':'actual v6 materialized '+split})
    old=repo/'research-checkpoints/policy-openai-slm-matched-20261008/hosted-source/component-report.json.gz'
    report=json.loads(gzip.decompress(old.read_bytes()))
    utterances.extend((str(old),r['context']['message']) for r in report['records'])
    inventory.append({'path':str(old),'sha256':rt.sha(old),'utterances':len(report['records']),'scope':'known 31-conversation development comparison'})
    pilot=repo/'research-checkpoints/v7-r8-pilot-20261009/results/qwen35-2b-lora-v7-r8-pilot-20261010T002204Z-evaluation.zip'
    with zipfile.ZipFile(pilot) as package:
        report=json.loads(package.read('evaluation/v7/component-report.json'))
        utterances.extend((str(pilot)+'::evaluation/v7/component-report.json',r['context']['message']) for r in report['records'])
    inventory.append({'path':str(pilot),'sha256':rt.sha(pilot),'utterances':1040,'scope':'known v7 pilot development cohort'})
    unique={}
    for source,text in utterances:unique.setdefault(normalized(text),(source,text))
    reference=list(unique.values());index=defaultdict(set);bags=[]
    for number,(source,text) in enumerate(reference):
        bag=shingles(text);bags.append(bag)
        for token in bag:index[token].add(number)
    candidates=[rt.strict_json(line) for line in (private/'candidate/conversations.jsonl').read_text(encoding='utf-8').splitlines()]
    queue=[];exact=[];normalized_matches=[];internal=Counter();family_texts=defaultdict(set)
    for case in candidates:
        for turn_number,turn in enumerate(case['turns'],1):
            text=turn['message'];norm=normalized(text);internal[norm]+=1;family_texts[norm].add(case['source_group'])
            if norm in unique:
                source,prior=unique[norm]
                record={'scenario_id':case['scenario_id'],'turn':turn_number,'source':source,'text':text,'reference':prior}
                normalized_matches.append(record)
                if prior==text:exact.append(record)
            bag=shingles(text);votes=Counter()
            for token in bag:
                for number in index[token]:votes[number]+=1
            nearest=[]
            for number,overlap in votes.most_common(30):
                score=overlap/len(bag|bags[number]) if bag|bags[number] else 0
                similarity=difflib.SequenceMatcher(None,masked(text),masked(reference[number][1]),autojunk=False).ratio()
                if score>=.30 or similarity>=.70:
                    nearest.append({'source':reference[number][0],'reference':reference[number][1],
                        'trigram_Jaccard':score,'masked_character_similarity':similarity})
            if nearest:
                best=max(nearest,key=lambda value:value['masked_character_similarity'])
                queue.append({'scenario_id':case['scenario_id'],'turn':turn_number,'candidate':text,'nearest':best,
                    'decision':'pending','reviewer':None,'rationale':None})
    queue_path=private/'review/overlap-queue.jsonl'
    with queue_path.open('x',encoding='utf-8',newline='\n') as stream:
        for record in queue:stream.write(rt.dumps(record)+'\n')
    with (private/'review/overlap-decisions.csv').open('x',encoding='utf-8',newline='') as stream:
        columns=['scenario_id','turn','candidate_sha256','decision','reviewer','reviewed_at_utc','rationale']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for record in queue:writer.writerow({k:record[k] for k in ('scenario_id','turn')}|{'candidate_sha256':rt.sha(private/'candidate/conversations.jsonl'),'decision':'pending'})
    report={'status':'machine_overlap_audit_complete_human_near_overlap_review_pending',
        'sources':inventory,'reference_utterances':len(utterances),'unique_normalized_reference_utterances':len(reference),
        'candidate_utterances':sum(internal.values()),'candidate_exact_reference_matches':len(exact),
        'candidate_normalized_reference_matches':len(normalized_matches),'near_overlap_queue':len(queue),
        'within_candidate_repeated_normalized_turns':sum(n-1 for n in internal.values()),
        'shared_clauses_across_macro_groups':sum(len(groups)>1 for groups in family_texts.values()),
        'limitations':['Lexical masking and trigram/character similarity cannot certify independent human authorship.',
            'Parameterized synthetic clauses recur across macro-template families; clustered intervals remain conditional on this convenience design.',
            'Accessible training and development are audited; previously sealed final labels are deliberately excluded.'],
        'candidate_sha256':rt.sha(private/'candidate/conversations.jsonl'),'overlap_queue_sha256':rt.sha(queue_path),
        'final_gate':'Exact/normalized reference matches must be removed or adjudicated before final freeze; every near-overlap row requires independent human decision.',
        'sealed_final_accessed':False,'model_calls':0}
    rt.atomic(private/'review/overlap-audit.json',report,exclusive=True)
    rt.atomic(rt.CHECKPOINT/'overlap-audit-summary.json',report,exclusive=True)
    rt.atomic(private/'review/exact-overlap-findings.json',{'exact':exact,'normalized':normalized_matches},exclusive=True)
    print(rt.dumps({k:v for k,v in report.items() if k not in {'sources','limitations'}}),flush=True)


if __name__=='__main__':main()
