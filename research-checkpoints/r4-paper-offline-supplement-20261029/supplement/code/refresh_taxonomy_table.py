"""Export explicit zero/unit rows from existing diagnostics; no engine replay."""
from collections import Counter,defaultdict
import gzip
import json
from common import OUT
from error_and_oracle_analysis import write_taxonomy_table
out=OUT/'analysis';aggregate=defaultdict(Counter)
for line in gzip.decompress((out/'error-taxonomy-per-call.jsonl.gz').read_bytes()).splitlines():
    row=json.loads(line)
    for name,n in row['category_counts'].items():aggregate[(row['key'][1],row['key'][2],name)].update(calls=1,individual_updates_or_slot_transitions=n)
write_taxonomy_table(aggregate,out)
print('Complete zero/unit rows exported from saved diagnostics; no engine/model calls.')
