# Fresh matched OpenAI–SLM results — 8 October 2026

## Main result

The remote stock/v5/v6 reports were matched against the main PC's **fresh** pinned
OpenAI Structured Outputs reports. All four models used the same 31 reviewed
weather, inflation, stocks and crypto conversations, the same labels and the same
frozen parsing policy. Every component context matched exactly; rollout retained
each model's own previous state with the same user script.

**V6 and OpenAI have close primary point estimates on this development cohort.
The paired intervals do not establish a winner or prove statistical equivalence.**

| Model | Component F1: supplied gold prior state | Rollout F1: own accumulated state |
|---|---:|---:|
| Stock Qwen3.5-2B | 0.0000 | 0.0000 |
| Generalist v5 LoRA | 0.4213 | 0.4270 |
| Domain-specific v6 LoRA | **0.4818** | **0.4986** |
| OpenAI gpt-5-mini-2025-08-07, Structured Outputs F | **0.4844** | **0.5086** |

F1 measures exact **slot ID + value + status** tuples, excluding the intent slot.
It combines precision and recall; it is not the percentage of complete successful
conversations. Component testing supplies correct prior state. Rollout tests the
accumulation of the model's actual outputs, using scripted user turns rather than
live generated clarification dialogue. Each model has 93 calls per track.

## Paired statistical comparison

All intervals use 10,000 resamples of 31 whole conversation clusters, seed 42,
recomputing exact micro F1. Differences below are **SLM minus OpenAI**.

| Pair | Track | F1 difference | Exploratory 95% interval |
|---|---|---:|---:|
| v6 − OpenAI | Component | −0.0026 | [−0.0912, +0.0895] |
| v6 − OpenAI | Rollout | −0.0100 | [−0.0836, +0.0639] |
| v5 − OpenAI | Component | −0.0631 | [−0.1364, +0.0163] |
| v5 − OpenAI | Rollout | −0.0816 | [−0.1325, −0.0304] |

Both v6 intervals include zero. OpenAI's rollout score is higher than v5's with an
interval excluding zero in this exploratory analysis, while the component interval
includes zero. The remote v6–v5 intervals also include zero, despite higher v6 point
estimates. There is no multiple-comparison correction or unseen-user validation;
these are cohort-specific findings, not general superiority claims.

## Similar F1, different error patterns

| Rollout measurement | v5 | v6 | OpenAI F |
|---|---:|---:|---:|
| Exact tuple precision | 0.5429 | **0.6312** | 0.4332 |
| Exact tuple recall | 0.3519 | 0.4120 | **0.6157** |
| JSON syntax validity | 93/93 | 91/93 | 93/93 |
| Declared schema compliance | 93/93 | 91/93 | 93/93 |
| Frozen parser admission | 93/93 | 90/93 | 93/93 |
| Valid unknown turns with no non-intent updates | 23/31 | **27/31** | 0/31 |
| Exact complete non-intent conversations | 0/31 | 0/31 | 0/31 |
| Median extraction latency | 5.378 s | 5.262 s | 2.688 s |

V6 is more conservative: it emits fewer incorrect exact tuples, but misses more
stated gold tuples. OpenAI recovers more exact gold tuples but emits more incorrect,
redundant or unwanted updates. These false positives are not all fabricated facts:
some represent repeated state, unknown markers, value-boundary or status errors.

OpenAI's component track has 93/93 declared-schema-valid outputs but only 91/93
parser-admitted outputs because two repeat a slot ID. V6 has two JSON failures and
one duplicate-slot rejection per track. These counts illustrate why syntax,
declared schema, admission, grounding and semantic correctness are separate stages.
Stock's zero operational F1 is dominated by contract failures and does not prove
the absence of semantic knowledge.

The remote v6 run made locally correct horizon changes in 31/31 rollout corrections
and preserved the other prior slots in 31/31. Nevertheless, complete state after
correction matched gold in 0/31 conversations for both adapters. Correcting the
horizon does not repair missing or incorrect requirements from earlier turns.
Under the same exact criterion OpenAI also had 0/31 fully exact conversations.

Latency is descriptive across different local GPU sessions and a hosted network
service. CUDA arms use BF16 on RTX A4000, greedy decoding and a 1,024-token cap;
OpenAI uses minimal reasoning and a 2,048-token cap. GPU runs were serial rather
than interleaved. These measurements do not establish compute/energy efficiency
or represent STT, user response time, live question generation or forecasting.

This comparison evaluates the configured extraction pipelines: OpenAI F uses
generation-time Structured Outputs, whereas these SLM arms use prompted JSON with
greedy generation. The shared parser makes admission consistent; it does not
isolate model architecture from output-generation method. The separate historical
XGrammar study is not an arm of this fresh 31-case comparison.

## Failure review before deciding on more training

`human-review-queue.jsonl` contains **31 prioritized conversations**, including
original utterances, unchanged gold, raw predictions, exact missing/extra tuples
and source-report pointers for OpenAI, v5 and v6 in both tracks. Review decisions
remain blank/pending in `human-review-sheet.csv`. No labels or primary scores were
changed in this analysis.

Priorities:

1. **Free-text label boundaries:** does the exact target noun phrase adequately
   represent problem_statement and target_description? Are longer verbatim
   responses semantically acceptable despite exact-match failure?
2. **Correction status:** a replacement is `provided` under the frozen rubric;
   some hosted responses use `confirmed`. Record whether the human agrees with
   that distinction and its practical pipeline impact.
3. **Unknown information:** distinguish invented values from explicit unknown
   markers. The current protocol requires omission, so unmentioned-status updates
   can still be protocol violations even without a fabricated real-world value.
4. **Initial omission versus correction failure:** inspect which earlier facts
   prevent a complete accumulated state when the horizon itself is correct.
5. **Evidence and structure:** review unsupported evidence, duplicate IDs and the
   malformed v6 JSON outputs. Preserve all failed calls in the denominator.

For example: “Actually change the horizon to 8 days; leave everything else alone”
should update only horizon to `{periods:8,unit:day}`, status `provided`, correction
true. A correct horizon cannot compensate for an earlier missed target unit. “I
don't know the timezone” should leave it unset under this protocol, rather than
store the unknown utterance as a timezone value.

After adjudication, freeze any revised rubric as a **new version**, then validate
on fresh independently annotated conversations. Do not silently re-label this
cohort after seeing outcomes or infer that another training round is already needed.

## Teacher/paper explanation

**Professional:** We compared stock, generalist fine-tuning, domain-specific
fine-tuning and a pinned hosted structured-output baseline using the same reviewed
requirements and admission policy. Domain-specific v6 reached close exact F1 point
estimates to OpenAI on these development conversations and showed a different
precision–recall tradeoff. Paired conversation bootstrap intervals do not establish
v6 superiority or equivalence. Complete-state errors and annotation questions remain
targets for human review and independent validation.

**Roman Urdu:** Hum ne stock model, v5, v6 aur OpenAI ko wohi 31 conversations aur
wohi checking rules diye. V6 aur OpenAI ke F1 scores qareeb hain, lekin is chhote
test se kisi ko winner ya dono ko barabar sabit nahi kar sakte. V6 kam ghalat ya
extra slot updates deta hai, magar zyada stated facts chhor deta hai. OpenAI zyada
facts pakarta hai, lekin extra updates bhi deta hai. Horizon sahi karna aur poori
purani requirements sahi hona alag baat hai. Ab failures ko insaan review karega;
us ke baad naye unseen examples par test karke training ka faisla karna chahiye.

## Verification and reproducibility

- Remote source commit: `b809dda515e4484bdb036b4db9caffd294a1b264`.
- GitHub ZIP SHA-256: `0e80568800eb1a46998214a1b59bd8cc69d80174d2851a6b59e44635ec045391`.
- Verified 10 Git result artifacts, the GitHub download, ZIP CRC and 201 raw run
  files against remote checksums. Existing hosted reports were independently hashed.
- Recomputed primary metrics and parser admissions for **all 744 scored calls**:
  558 remote plus 186 hosted. All values reproduced exactly.
- The bundled comparison passed policy, protocol, full-cohort keys, gold, cluster
  and component-context guards for all three hosted–SLM pairs. Rollouts use each
  model's own prior state with identical user messages and gold.
- Policy SHA-256: `558581a35a0abc592c59d6f30bc7a2a3c0542368bfee668ac3018800b702189e`.
- No new model inference, paid APIs, training, final-label access or production
  defaults were involved. The earlier hosted run cost about $0.16; this analysis
  added $0. No GPU monetary/energy cost comparison is claimed.

`summary.csv` and the three `openai-vs-*.json` files contain machine-readable results.
Supplemental status-free multiset scoring is in `summary.json` and is expressly
posthoc, not the primary endpoint. It preserves duplicate counts; the earlier
remote supplemental slot-key dictionary summary collapses duplicate slot entries,
so its tiny numerical differences must not be mixed with this diagnostic.

**Limits:** agent-authored, agent-reviewed posthoc development; 31 clusters (seven
weather, eight per other domain); 10 positive non-intent slots, despite a 79-slot
prompt/schema; independent human review and unseen-case evaluation pending. The
old 32-case studies and 0.8B training rounds are separate cohorts and are not rows
in this matched comparison.
