This post-hoc, offline-only diagnostic is declared after observing the aggregate
primary scores and before computing this sensitivity. It is not preregistered
and does not replace the frozen primary endpoint or change labels/responses.

Question: how much of exact actionable F1 depends on numeric JSON spellings
(for example 1 versus 1.0 inside a duration)? Retain native admission, slot IDs,
statuses, strings, list ordering, duplicate counts and all other value distinctions.
Canonicalize only finite JSON numbers mathematically, keeping booleans distinct.
Recompute the same multiset TP/FP/FN under this sensitivity and report both sets
of counts side by side. No synonym repair, semantic relabelling or model calls.
