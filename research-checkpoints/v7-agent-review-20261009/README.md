# V7 reviewed development bundle

Current candidate: v7-20261009-r7-dev. Extract into a NEW directory.
Read docs/research/V7_AGENT_REVIEW_2026-10-09.md and the current human review
START_REVIEW_HERE.md. No human approvals, training or model-quality claims.

python -m pip install -r requirements-v7-data.txt
python scripts/build-corpus-v7-reviewed.py --verify --output corpus-v7/v7-20261009-r7-dev
python -B -m unittest discover -s tests -p test_v7_agent_review.py
python -B -m unittest discover -s tests -p test_v7_review_gate.py

Do not call a builder to reproduce final labels. The reviewed builder filters
out final assignments before rendering. The archive contains no final labels.
The old r4 sealed final is NOT a matching r7 benchmark. Old r4/r5 inputs are held.

Included pinned fpy source and data-check dependencies reproduce the original
parser. The included old 31/32 evaluation cases support source/policy locks and
copy-exclusion guards only; they are not v7 training cases.

Current records: 2800 training / 600 validation conversations; 14560 / 3120 SFT
rows. Planned 600 final assignments are metadata only, never generated here.
The 524 human decisions remain pending. Agent checks are not human approval.
Current 14 nearest-neighbor representatives still share synthetic templates.

The tokenizer receipt covers all development rows: maxima 3205 / 3189, zero
above 3584, no truncation. No weights/adapters/keys/database included. Repeat a
tokenizer preflight in a NEW receipt if the trainer or environment changes.

The old r4 audit-v2 findings are included for provenance. To reproduce those
old-input findings, obtain the separately published r4 preparation bundle; do
not treat its obsolete inputs as training-ready. Original audit-v1 summary counts
were superseded; source snapshots are preserved on the originating machine.

Human policy decisions, independent natural-language evaluation, GPU training
and a separately controlled matching final set remain pending. The unchanged
v5 prompt's confirmation convention needs explicit adjudication before release.
