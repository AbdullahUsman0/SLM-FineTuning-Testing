# Six trace-backed worked examples

Raw excerpts below are illustrative subsets, not repaired responses. Full raw text, priors, pre/post states and traces are in worked-examples.json. Taxonomy is post-hoc; semantic acceptability remains a human-review question.

## 1. Correct current extraction harmed by recovery

Source: `v7-20261009-r8-dev/commodities/0032` / `t2`; v8, component. Raw TP/FP/FN: [2, 0, 0].

Message: The response variable is stored in column gas_settlement. The input timestamp identifier is recorded_at.

Raw excerpt: `{"correction_detected": false, "intent": "create_forecast", "updates": [{"candidate_value": "\"gas_settlement\"", "confidence": 1.0, "evidence_text": "The response variable is stored in column gas_settlement.", "slot_id": "target_column", "status": "provided"}, {"candidate_value": "\"recorded_at\"", "confidence": 1.0, "evidence_text": "The input timestamp identifier is recorded_at.", "slot_id": "time_column", "status": "provided"}]}`

Pre/post-recovery comparison: `{"output_granularity": {"after": {"confirmed_by_user": false, "status": "provided", "value": "The response variable is stored in column gas_settlement. The input timestamp identifier is recorded_at."}, "before": null, "expected": null}}`

Trace recovery slots: `["output_granularity"]`. See the corresponding full machine-readable example for complete context and trace.

## 2. Unknown abstention does not ensure state preservation

Source: `v7-20261009-r8-dev/commodities/0032` / `t5`; v8, rollout. Raw TP/FP/FN: [0, 0, 0].

Message: I cannot supply a timezone, seasonal cycle lengths, prediction interval coverage or authorized access reference at this stage.

Raw excerpt: `{"correction_detected": false, "intent": "create_forecast", "updates": []}`

Pre/post-recovery comparison: `{"business_goal": {"after": {"confirmed_by_user": false, "status": "provided", "value": "I cannot supply a timezone, seasonal cycle lengths, prediction interval coverage or authorized access reference at this stage."}, "before": null, "expected": null}}`

Trace recovery slots: `["business_goal"]`. See the corresponding full machine-readable example for complete context and trace.

## 3. Ignored unknown markers are distinct from actionable facts

Source: `v7-20261009-r8-dev/corporate_finance/0116` / `t6`; openai, component. Raw TP/FP/FN: [0, 0, 0].

Message: Timezone, seasonal cycle lengths, prediction interval levels and access details still need a decision. Keep them unspecified.

Raw excerpt: `{"correction_detected": false, "intent": "create_forecast", "updates": [{"candidate_value": "Timezone, seasonal cycle lengths, prediction interval levels and access details still need a decision. Keep them unspecified.", "confidence": 0.9, "evidence_text": "Timezone, seasonal cycle lengths, prediction interval levels and access details still need a decision. Keep them unspecified.", "slot_id": "timezone", "status": "unmentioned"}, {"candidate_value": "Timezone, seasonal cycle lengths, prediction interval levels and access details still need a decision. Keep them unspecified.", "confidence": 0.9, "evidence_text": "Timezone, seasonal cycle lengths, prediction interval levels and access details still need a decision. Keep them unspecified.", "slot_id": "seasonal_periods", "status": "unmentioned"}]}`

Pre/post-recovery comparison: `{"seasonal_periods": {"after": null, "before": null, "expected": null}, "timezone": {"after": null, "before": null, "expected": null}}`

Trace recovery slots: `[]`. See the corresponding full machine-readable example for complete context and trace.

## 4. Exact extraction discrepancy requiring human semantic review

Source: `v7-20261009-r8-dev/commodities/0032` / `t1`; openai, component. Raw TP/FP/FN: [6, 2, 2].

Message: The analyst request is as follows.  We need an outlook for natural-gas front-month settlement price for Pelmar commodities site 2203. Target measurements use USD per MMBtu. Each observed record covers 1 day. The requested future coverage is 19 days. Select the source from the data catalog. The supplied source identifier is https://inputs.example/commodities/2203. The input container uses xlsx.

Raw excerpt: `{"correction_detected": false, "intent": "create_forecast", "updates": [{"candidate_value": "\"https://inputs.example/commodities/2203\"", "confidence": 0.9, "evidence_text": "The supplied source identifier is https://inputs.example/commodities/2203.", "slot_id": "source_reference", "status": "provided"}, {"candidate_value": "\"xlsx\"", "confidence": 0.95, "evidence_text": "The input container uses xlsx.", "slot_id": "file_format", "status": "provided"}]}`

Pre/post-recovery comparison: `{"source_mode": {"after": {"confirmed_by_user": false, "status": "provided", "value": "upload"}, "before": null, "expected": {"confirmed_by_user": false, "status": "provided", "value": "catalog"}}}`

Trace recovery slots: `["source_mode"]`. See the corresponding full machine-readable example for complete context and trace.

## 5. Locally correct extraction inherits earlier rollout errors

Source: `v7-20261009-r8-dev/commodities/0032` / `t2`; openai, rollout. Raw TP/FP/FN: [2, 0, 0].

Message: The response variable is stored in column gas_settlement. The input timestamp identifier is recorded_at.

Raw excerpt: `{"correction_detected": false, "intent": "create_forecast", "updates": [{"candidate_value": "\"gas_settlement\"", "confidence": 0.9, "evidence_text": "The response variable is stored in column gas_settlement.", "slot_id": "target_column", "status": "provided"}, {"candidate_value": "\"recorded_at\"", "confidence": 0.9, "evidence_text": "The input timestamp identifier is recorded_at.", "slot_id": "time_column", "status": "provided"}]}`

Pre/post-recovery comparison: `{"output_granularity": {"after": {"confirmed_by_user": false, "status": "provided", "value": "The response variable is stored in column gas_settlement. The input timestamp identifier is recorded_at."}, "before": null, "expected": null}}`

Trace recovery slots: `["output_granularity"]`. See the corresponding full machine-readable example for complete context and trace.

## 6. Structural admission is separate from readable facts

Source: `v8-final-20261018-r1/focused/inflation_prices/validation-004` / `t2`; v8, component. Raw TP/FP/FN: [0, 4, 4].

Message: The spacing between observations is 2 hours. The forecast issuance interval is 2 days. Source ingestion repeats at 5-hour intervals. There are 6 years of historical observations available.

Raw excerpt: `{"correction_detected": false, "intent": "create_forecast", "updates": [{"candidate_value": "{\"periods\": 2, \"unit\": \"hour\"}", "evidence_text": "The spacing between observations is 2 hours.", "slot_id": "frequency", "status": "provided"}, {"candidate_value": "{\"periods\": 2, \"unit\": \"day\"}", "evidence_text": "The forecast issuance interval is 2 days.", "slot_id": "prediction_frequency", "status": "provided"}]}`

Pre/post-recovery comparison: `{"source_reference": {"after": {"confirmed_by_user": false, "status": "provided", "value": "The spacing between observations is 2 hours. The forecast issuance interval is 2 days. Source ingestion repeats at 5-hour intervals. There are 6 years of historical observations available."}, "before": null, "expected": null}}`

Trace recovery slots: `["source_reference"]`. See the corresponding full machine-readable example for complete context and trace.

