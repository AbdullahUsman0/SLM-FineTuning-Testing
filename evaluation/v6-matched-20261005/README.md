# Matched stock 2B / v5 / v6 study

Inputs and annotations were frozen before any scored model output. See
study-manifest.json for hashes, domains, call counts and the predeclared protocol.
frozen.jsonl contains the complete four-domain subset of the committed v6
validation split. natural.jsonl contains 32 agent-authored three-turn diagnostics,
pending independent human review. No sealed final labels are used.

The runner evaluates 631 frozen component calls, 96 natural component calls and
96 natural predicted-state rollout calls per arm: 2,469 calls across all arms.
Exact slot/value/status, slot names, JSON/schema validity, unmentioned requirements,
correction handling, unknown-information handling and serial CUDA response time
are reported separately. Stock, v5 and v6 share one pinned stock base. The
native stock output is checked against the stock path with adapters disabled.

The scorer retains failures and raw output; no repair or application recovery
improves the model's measured score. Latency excludes loading and unscored warmup.
Arms rotate per scenario to reduce timing drift. The runner uses the original
training dependency pin and independently verifies both adapters and base weights.

Run with the existing CUDA environment and HF cache:

```powershell
$env:HF_HOME = 'D:\SLM\hf-cache'
$env:HF_HUB_OFFLINE = '1'
& .\.venv\Scripts\python.exe scripts/evaluate-v6-matched.py --output D:\SLM\FYP-model-runs\matched-v6-evaluation-20261005
```

Only completed scenarios are reused on resume; partial scenarios are preserved
and replayed. Run fingerprint mismatches stop evaluation. No training or paid
API is started. Review natural behavior before considering another training round.
