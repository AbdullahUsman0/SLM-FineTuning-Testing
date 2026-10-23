# Completed v8 development diagnostic; v6 comparison pending

The experimental final-step v8 adapter completed all 996 calls on 100 fixed synthetic development conversations (498 turns in each of two tracks). This snapshot publishes v8 immediately while v6 continues separately. It does not claim the combined experiment or an OpenAI comparison is complete.

| Measurement | Component: supplied gold prior | Rollout: own accepted prior |
|---|---:|---:|
| Exact non-intent slot/value/status micro F1 | 0.963953 | 0.963953 |
| Precision / recall | 0.965126 / 0.962783 | 0.965126 / 0.962783 |
| JSON syntax / declared schema valid | 498/498 | 498/498 |
| Application admission | 491/498 | 491/498 |
| Exact complete state at a turn | 486/498 | 458/498 |
| Correct correction updates | 120/120 | 120/120 |
| Complete state after correction | 120/120 | 111/120 |
| Successful empty unknown responses | 103/103 | 103/103 |
| Unknown turns preserving non-intent state | 103/103 | 103/103 |
| Extra current-turn slot flags, including rejected | 1 | 1 |
| Forbidden-slot flags, including rejected | 0 | 0 |
| Median extraction seconds | 7.343 | 7.353 |

F1 is exact slot ID + decoded value + status, excluding intent; it is not a percentage of successful conversations. Complete-state counts above are turns, not conversations. Valid JSON does not imply application admission or correct requirements. Extra-slot flags do not count every incorrect value or every possible invented requirement; false positives and detailed raw outputs must also be inspected. Exact correction updates can coexist with an accumulated-state failure from an earlier turn.

The main language weakness is Roman Urdu: exact F1 0.771536 versus English 0.985417 and mixed 1.000000, in each track. There are only 10 Roman Urdu conversations and 10 mixed conversations; these are descriptive subgroup results. Both tracks have 46 false negatives and 43 false positives. Excluding the explicit seeded state-repair category gives F1 0.958739 in both tracks. The full summaries retain all language/domain-family and seeded-intervention exclusion results; granular-domain-breakdown.json covers all 20 individual domains. failure-findings.json is automated triage, not human adjudication.

The raw ZIP contains calls.jsonl (all raw responses and latency), scored-records.jsonl.gz, freeze.json, runtime.json, summary.json and an internal SHA-256 manifest. Its adjacent checksum and raw-archive-receipt.json permit integrity checks. SNAPSHOT_STATUS.json records independent packaging consistency checks; artifact-manifest.json hashes the publication files. No base weights, adapter weights, optimizer state, credentials or caches are included here.

Generation used the active v8 LoRA on NVIDIA RTX A4000, CUDA BF16, the pinned Qwen/Qwen3.5-2B revision 15852e8c16360a2fea060d615a32b45270f8a8fc, greedy decoding, thinking disabled, max_new_tokens 1024, one call per turn and no repair. Source training/final_round/evaluate_development.py and provider hashes, the cohort digest, instructions policy and adapter hashes are frozen in freeze.json. The matched v6 run uses a separate output folder and will be published by the existing companion after completion.

These inspected synthetic development conversations come from the public v8 validation split; they are not an untouched final test. Synthetic source/template dependence, pending independent human annotation, small subgroups and explicit seeded state fixtures limit generalization. No significance test, statistical equivalence, final-test result or model winner is established. The older OpenAI comparison uses different cases/policy and cannot be compared numerically with these v8 scores. No paid API calls or sealed final-label access occurred.

Training evidence and the downloadable adapter are documented in ../training-completed-20261023T002215Z/LOCAL_CODEX_HANDOFF.md and ../ADAPTER_RELEASE_NOTICE.md. This snapshot does not change the training run or the ongoing v6 diagnostic. Reproduce with the supplied training/final_round instructions in a NEW diagnostic output directory; do not rerun or overwrite the original completed outputs. package-snapshot.py records the exact read-only source checks and publication packaging used here, with paths specific to the training PC.
