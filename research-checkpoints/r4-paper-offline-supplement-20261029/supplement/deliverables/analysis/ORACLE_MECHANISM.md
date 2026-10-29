# Oracle recovery mechanism audit

Scope: 101 frozen main component turns supplied with gold extraction and gold prior. These are deterministic application replays, not generated model responses or new end-to-end rollouts.

All 20 published post-pipeline oracle mismatch keys reproduce. All 20 of those calls match expected state at the pre-recovery reducer snapshot. Thus their discrepancies arise in the later application path, not from missing/wrong oracle extraction.

| Changed slot | Mismatch calls |
|---|---:|
| output_granularity | 14 |
| target_unit | 1 |
| source_mode | 1 |
| source_reference | 4 |

All changed slots have explicit recorded recovery IDs: True.

Under this fixed-script adapter, selected-slot recovery can assign the current message to a pending free-text slot even when the message states other requirements or explicitly withholds information. This is an observed failure mode of these fixtures; it is not an estimate of its frequency in a live interview with assistant-question context.

The 14 output_granularity discrepancies, four source_reference discrepancies, one target_unit discrepancy and one source_mode discrepancy are retained with messages, expected/pre/post states and recovery IDs in oracle-20-mismatches.json. Full event traces are in oracle-traces.jsonl.gz. No engine patch or automatic label correction was made.

The raw extraction/state separation matters: refusing to invent a model update does not prevent a later fallback from assigning an unrelated value. The seven rollout unknown-state changes after correct abstention in the local arm must not be described as model hallucinations without this trace distinction.

A future fix should be assessed in a newly versioned application study, with reviewed question/answer context and declared intervention policy. Saved outputs replayed under different rollout contexts are not a fresh experiment.
