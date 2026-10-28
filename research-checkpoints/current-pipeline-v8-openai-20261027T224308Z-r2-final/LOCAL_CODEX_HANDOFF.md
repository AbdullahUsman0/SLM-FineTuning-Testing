# Local Codex handoff

GPU run: D:\SLM\FYP-evaluation-runs\current-pipeline-v8-openai-20261027T224308Z-r2
Research branch: experiment/v5-grounded-20260919; source handoff commit 0ae1e0a1215c061a5a5e4bb02bb99f4ccbc92554.
Result folder: D:\SLM\SLM-FineTuning-Testing\research-checkpoints\current-pipeline-v8-openai-20261027T224308Z-r2-verified
Completion: complete_matched_development; main 404/404 scored calls; 20 planned conversations, 101 turns per model/track; pilots are excluded from main scores.
API actual $0.250278; outstanding $0.000000. The existing shared budget was not reset.

Read completion.json, protocol.json, README.md, main-summary.json (if present), adjudication-authorization.json, adjudication-changes.json, failure-findings.json, label-audit.json, historical-openai-audit.json and the compressed raw/scored journals. Preserve existing paper/edit backups. This is new inspected development, independent human review incomplete, not an untouched final test. No universal superiority/equivalence/deployment claim is supported. Raw model scores and pipeline recovery must stay separate. Previous cohorts and archived scores remain separate; the historical hosted arithmetic reproduced.

Base: Qwen/Qwen3.5-2B at 15852e8c16360a2fea060d615a32b45270f8a8fc. The adapter hashes and actual GPU/runtime appear in protocol.json/local-runtime.json. This ZIP includes no weights, optimizer, caches, credentials or .env files. The adapter remains available from the existing v8-adapter-20261022 GitHub prerelease. Do not assume training-PC D: paths exist on the other PC.

Verify the adjacent ZIP checksum and every ARCHIVE_MANIFEST.json member SHA before use. Exact source bundle is in bundle/: run verify_bundle.py. Recreate fpy at 9bc2bd18c495514efa79a4371be1d706b9fc3824 plus its verified bundled patch/working copies in a NEW checkout. Use all source hashes in protocol.json. Do not substitute fpy main or an installed package. Preserve the raw extraction labels and the separately approved state-status adjudication.

For this server, reproduction/resume uses the existing interpreter and SAME ledger:
  D:\SLM\FYP-environments\v8-final-20261018T230255Z\Scripts\python.exe -B D:\SLM\SLM-FineTuning-Testing\evaluation\current_pipeline_v8_openai_20261027_r2\runner.py --run D:\SLM\FYP-evaluation-runs\current-pipeline-v8-openai-20261027T224308Z-r2 --repo D:\SLM\SLM-FineTuning-Testing --stage pilot
then --stage main only when the pilot receipt and affordability gate pass. Set PYTHONPATH to D:\SLM\FYP-evaluation-runs\current-pipeline-v8-openai-20261027T224308Z-r2\offline-deps and D:\SLM\SLM-FineTuning-Testing\evaluation\current_pipeline_v8_openai_20261027_r2; load the NEW Windows user credential explicitly into the worker; use https://api.openai.com/v1 and clear inherited organization/project overrides for the worker. Do not copy credentials to the other PC or reset the ledger. The frozen source and case hashes must match on resume. A spent/uncertain dispatch must not be replayed. Correct paths for a new machine only under a separately documented reproduction version and budget decision.

Tests: offline-tests.json/log contain the exact command, versions and outcome. CUDA smoke tests and pilot outputs are separate from main. All full SDK responses/usage are in hosted-responses; request JSON includes synthetic prompts/schema, never authorization headers. Do not treat metadata HTTP 200 as proof that generation succeeded; use completed receipts and usage.

Update the existing paper's current-pipeline methods/results/limitations only if completion.json indicates a complete matched study. Otherwise document the precise blocker/partial coverage and leave unmeasured results blank. Return edited paper paths and a traceable results table; do not invent a replacement manuscript. No additional paid evaluation, training or sealed-test access is authorized by this handoff.


Additional verified reporting: read FINDINGS_AND_LIMITATIONS.md, PAPER_UPDATE_TABLE.csv, PAPER_UPDATE_PROPOSAL.md, supplementary-contract-audit.json, billing-usage-analysis.json and adjudication-scope-verification.json. The common strict comparison adapter is part of this measured configuration; raw contract failures must not be described as absence of semantic knowledge. Component final snapshots have supplied gold priors and are not complete-conversation success.

The authoritative shared ledger remains on this GPU server. Its export/snapshot is for analysis, not a fresh budget on another PC. Do not rerun completed stages or issue additional paid requests. Incomplete-stage recovery must use the SAME run, frozen source/cases and authoritative shared ledger under its exclusive lock. Fresh credentials must be loaded explicitly from Windows User environment into the worker; .env is ignored and is not read by this evaluator. SDK retries remain zero and provider URL is https://api.openai.com/v1.

The first export failed only because focused cases omit optional template_variant metadata; the recovery treats absent metadata as null and preserves the failed export folder. No inference was repeated, and primary scores were not changed. The supplied recovery source and receipt document this.

Main result table:

| System | Track | Precision | Recall | Raw exact F1 | JSON | Schema | Raw contract | Exact pipeline state, turns | Median seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v8 | component | 0.0000 | 0.0000 | 0.0000 | 87/101 | 25/101 | 25/101 | 15/101 | 5.980 |
| v8 | rollout | 0.0000 | 0.0000 | 0.0000 | 101/101 | 7/101 | 7/101 | 0/101 | 5.615 |
| openai | component | 0.1347 | 0.1751 | 0.1523 | 101/101 | 101/101 | 56/101 | 37/101 | 2.590 |
| openai | rollout | 0.1060 | 0.1440 | 0.1221 | 101/101 | 101/101 | 59/101 | 0/101 | 2.491 |

Whole-study usage-rate estimate $0.23867135; guard-accounted $0.25027775; reserved/uncertain $0.

Final ZIP on GPU PC: C:\Users\Abdullah\Downloads\current-pipeline-v8-openai-20261027T224308Z-r2-verified-evidence-20261027T234645Z.zip. Its SHA-256 is in the adjacent .zip.sha256 file and the standalone handoff returned with it; an archive cannot embed its own checksum without circularity.
