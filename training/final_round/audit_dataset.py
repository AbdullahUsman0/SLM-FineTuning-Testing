"""Lexical development overlap diagnostics, not proof of independent examples."""
from collections import Counter,defaultdict
import gzip,json,re
from data_tools import CORPUS,atomic,rows,dumps,sha

def tokens(c):
    # Exclude the repeated 79-slot prompt: compare user dialogue content only.
    return set(re.findall(r'[a-z]+',' '.join(t['message'] for t in c['turns']).casefold()))

def audit():
    cases={s:list(rows(CORPUS/'splits'/f'{s}.jsonl.gz')) for s in ('train','validation')}
    messages={s:Counter(t['message'].casefold().strip() for c in cases[s] for t in c['turns']) for s in cases}
    common=set(messages['train'])&set(messages['validation'])
    train=[tokens(c) for c in cases['train']];validation=[tokens(c) for c in cases['validation']]
    index=defaultdict(set)
    for i,words in enumerate(validation):
        for word in words:index[word].add(i)
    flagged=0;stored=0;bins=Counter();path=CORPUS/'overlap-review-queue.jsonl.gz'
    with path.open('wb') as f,gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as out:
        for i,words in enumerate(train):
            candidates=set().union(*(index[w] for w in words))
            for j in candidates:
                other=validation[j]
                if min(len(words),len(other)) < .8*max(len(words),len(other)):continue
                similarity=len(words&other)/len(words|other)
                if similarity>=.8:
                    flagged+=1;bins['>=0.9' if similarity>=.9 else '0.8-0.9']+=1
                    if stored<2000:
                        out.write((dumps({'train':cases['train'][i]['scenario_id'],'validation':cases['validation'][j]['scenario_id'],
                            'user_dialogue_word_set_jaccard':similarity,'review_status':'pending'})+'\n').encode());stored+=1
    stats={}
    for split,items in cases.items():
        stats[split]={key:dict(Counter(c[key] for c in items)) for key in ['domain','domain_family','language','category','origin']}
        stats[split]['behavior_turns']=dict(Counter(t['behavior'] for c in items for t in c['turns']))
    atomic(CORPUS/'statistics.json',stats)
    result={'method':'word-set Jaccard over user dialogue, digits/punctuation removed; repeated schema prompt excluded',
        'threshold':.8,'possible_cross_split_pairs':len(train)*len(validation),'flagged_pairs':flagged,'bins':dict(bins),
        'queue_entries':stored,'queue_truncated':flagged>stored,'queue_sha256':sha(path),'human_reviewed_pairs':0,
        'shared_casefolded_current_message_strings':len(common),'train_occurrences_of_shared_messages':sum(messages['train'][m] for m in common),
        'validation_occurrences_of_shared_messages':sum(messages['validation'][m] for m in common),
        'interpretation':'Known template dependence and generic repeated abstention clauses remain. Entity-group/input separation is not semantic independence. No zero-leakage claim.',
        'final_test_accessed':False,'independent_human_review_complete':False}
    atomic(CORPUS/'overlap-summary.json',result);print(dumps(result))

if __name__=='__main__':audit()
