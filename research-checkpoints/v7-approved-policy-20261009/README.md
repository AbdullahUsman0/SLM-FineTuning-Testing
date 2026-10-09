# V7 r8: approved annotation policy, development data

All six conventions are agreed by the user. Human sample/overlap review remains
pending. Extract into a NEW directory and read docs/research/V7_APPROVED_POLICY_2026-10-09.md.
Use the annotation-policy.json and annotation-rubric.md INSIDE the r8 corpus;
the old root ANNOTATION_RUBRIC.md is historical, not the r8 filename convention.

python -m pip install -r requirements-v7-data.txt
python scripts/build-corpus-v7-reviewed.py --verify --output corpus-v7/v7-20261009-r8-dev
python -B -m unittest discover -s tests -p test_v7_approved_policy.py

2800/600 training/validation conversations; 14560/3120 extraction rows. Exact
30/60/10 domain and 80/10/10 source-language mixtures retained. 81 development
filename-format inferences; uppercase, extensionless and unsupported controls.
Inference uses inferred; later explicit format can upgrade to provided.

Prompt version: v7-user-approved-extraction-1. Future paired comparisons must
use this prompt and the same approved labels for every model. Historic scores
remain unchanged. Future Transformers chat uses --prompt-version v7; the existing
quantized local-server chat option continues to support v5 only. No r8 adapter
has been trained or measured, and no inference quality is claimed here.

The 524-case human sample and 14 overlap representatives are still pending.
The template explicitly has review_complete=false and overlap_review_complete=false.
Policy agreement alone cannot release the guarded materializer. No final labels
are included/generated; the 600 final assignments are metadata only. Do not use
older sealed final labels as a matching r8 benchmark or regenerate them for review.

Pinned fpy/parser source and exact data-check dependency versions are included.
Old 31/32 cases are lock/exclusion fixtures only, not v7 training examples.
Tokenizer maxima 3406/3390; no rows over 3584 and no truncation. No weights,
API keys, production database, training or paid model calls are included.
