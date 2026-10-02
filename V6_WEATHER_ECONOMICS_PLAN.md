# v6 weather and economics requirements corpus

Date: 2026-10-02. Status: candidate preparation; no training authorized by this
document. The user confirmed the 30/60/10 domain balance in this task.

## Purpose

Train a small language model to extract forecasting requirements and ask a
clarification question. This corpus does not teach numerical weather, inflation,
price or return prediction. No live measurements, forecasts, investment advice,
provider datasets, account credentials or identifiable people are redistributed.

Keep the pinned 79-slot schema, v5 prompts and exact JSON/Pydantic output contract.
Use a new adapter initialized from the same pinned stock Qwen3.5-2B, rather than
continuing the generalist adapter. This allows a generalist/specialist comparison.
The generalist's current remote behavioral evaluation remains in progress.

## Agreed scope

| Family | Scenarios | Share | Subdomains |
| --- | ---: | ---: | --- |
| Weather | 600 | 30% | Temperature; precipitation/snow; wind; humidity/pressure; solar/cloud; extreme-event requirements |
| Economics | 1,200 | 60% | Inflation/prices; output/growth; labor/employment; rates/credit/fixed income; equities/funds; FX; commodities; crypto spot; crypto derivatives; housing; trade/fiscal; corporate finance |
| Combined | 200 | 10% | Weather–energy and weather–agriculture |

There are 100 source scenarios per subdomain. The default target is 2,000 source
scenarios. Each source has three fact-preserving linguistic variants and one
clarification example. Correction/conflict sources also retain the confirmed
prior turn. Exact final counts are recorded in statistics.json, not guessed here.

Weather is now an in-domain training family, as explicitly requested by the user.
This supersedes the earlier proposal in DOMAIN_SCOPE_RESEARCH_2026-09-27.md to
use pure weather solely as transfer evaluation. Cross-domain transfer claims must
therefore use a different separately frozen external domain.

## What specialization means

The actual slot names remain unchanged. Domain phrases occur inside explicit
target descriptions, units, scope filters, calendars, source references,
covariates, events and business goals. Examples distinguish:

- CPI index levels, annual inflation rates, monthly changes, and seasonal adjustment;
- real versus nominal GDP and production versus consumption;
- raw/adjusted prices, total returns, volumes, spreads and volatility;
- quote direction for FX, spot versus futures, funding versus open interest;
- temperatures in Celsius/Fahrenheit, rainfall totals, wind speed/direction;
- observed historical weather covariates and genuinely known future information;
- corrected requirements and conflicting changes to confirmed requirements.

These concepts currently have no separate typed finance-specific slots. Keep
them verbatim in the appropriate existing slot, rather than inventing new keys.
Do not claim independently scored typed vintage/adjustment/currency extraction.
An optional schema extension would be a separate experiment and version.

The first candidate uses explicit domain language and fictional scopes. It is a
controlled synthetic corpus; independently authored, less explicit human
utterances are needed before strong deployment or publication claims. Its shared
clause machinery is disclosed in the fuzzy-overlap audit.

## Split and leakage protocol

Freeze entity groups before rendering. Five sources share one synthetic entity.
Keep all variants, context, corrections and questions in that entity's split.
Use 70/15/15 per subdomain: 1,400 training, 300 validation and 300 final scenarios.
Training, validation and final each use disjoint named template-family pools.
Actual lexical clauses and the application schema remain shared: the template
family holdout is not proof of linguistic independence.

Final labels live in ignored training-runs/v6-sealed/v6-20261002-r1, absent from
GitHub. The seal is procedural. The public deterministic builder could regenerate
them, so independent evaluation still requires a separate final custodian.
The verifier hashes final bytes only when explicitly given the sealed path and
does not deserialize them.

Check exact and normalized full model-input duplication, group overlap, frozen
assignments, all source facts, literal evidence spans, typed values, reducer
transitions, actual selected short-answer slots, and question contract. Report
cross-split representative text/template Jaccard flags with a pending review queue.
No zero-leakage claim is made while those flags are unresolved.

## Review and improvement before training

1. Read the domain review sample and all development scenarios in the compressed
   review package. Every source starts machine-validated and human-unreviewed.
2. Review domain correctness, explicit evidence, omission handling, naturalness,
   correction/conflict behavior, plausible units/calendars and covariate availability.
3. Resolve fuzzy pairs with reviewer identity, UTC time, accept/reject and reasons.
   Lexical overlap alone is not automatically semantic leakage.
4. Add independently authored human requests, especially short colloquial requests,
   unquoted values, ambiguity, implied corrections, distractor numbers and negation.
   Rebuild as a new immutable revision after edits; do not mutate the frozen r1.
5. Add controlled contrast sets for macro release vintages and corporate-action
   handling in a later revision if they become central claims. Do not imply that
   free-text target alternatives cover every economic index or release policy.
6. Record a review ledger and manifest-specific approval before preparing inputs.

Prototype fixture tests precede the full build. Review samples are development
only; final-case review feedback must not reach the trainer/model selector.

## Packaging and training

Corpus artifacts are deterministic gzip JSONL. The manifest records compressed
and uncompressed SHA-256 hashes. The materializer verifies them and writes only
training/validation SFT plus development component cases to a new input folder.

After review, run:

```text
python scripts/verify-corpus-v6.py --output corpus-v6/v6-20261002-r1
python scripts/prepare-v6-training.py --corpus corpus-v6/v6-20261002-r1 --output <NEW_RUN>/inputs --review-approval <REVIEW_APPROVAL.json>
```

The approval must match manifest SHA-256 and include reviewer, timestamp, ledger
hash, approved training/validation labels, completed overlap review, and an explicit
declaration that final labels were excluded from training and selection. Do not
reuse the v5 user confirmation to claim this new corpus has been reviewed.

Use Qwen/Qwen3.5-2B revision 15852e8c16360a2fea060d615a32b45270f8a8fc and fpy
04d52c015d1e3ecdefe92b87116f209361509b4b. Start with the v5 optimization settings
(two epochs, LR 5e-5, effective batch 16, checkpoints every 10 steps). Measure the
new corpus token lengths before choosing max_length; do not assume v5's 3584
limit covers the new data. Prevent truncation of completion labels.

Run a small matched stock/adapter pilot and inspect actual requirement extraction
before scaling. Preserve every checkpoint and choose by behavioral validation.
The generalist loading smoke missed explicit facts despite extremely low loss;
teacher-forced loss must not serve as the sole selection metric.

## Evaluation

Compare stock 2B, frozen v5 generalist, and new specialist on identical v6
development cases. Report weather, economics, combined and every subdomain
separately, plus aggregate exact non-intent slot F1, hallucinations, JSON/schema
validity, correction/conflict accuracy and clarification contract results. Use
paired source/entity-cluster bootstrap intervals; sources share entities, so
entity clusters are the preferred uncertainty unit here.

Run the specialist on unchanged v5 development cases to measure retention.
Training data size differs from v5: full-corpus results compare practical adapters,
not domain effects alone. For causal specialization claims add matched training
token/step budgets and a generic-corpus control at the same budget. Do not pool
v3/v5/v6 scores or use numerical forecasting metrics to score this extractor.

Keep the sealed final set untouched until adapter and output method are fixed.
External OpenAI comparison remains a separate explicit budget decision.

## Primary references consulted

Retrieved 2026-10-02. They inform terminology, not synthetic labels or values.

- NOAA GHCN-Daily: temperature, precipitation/snow and quality control.
  https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- IMF topic datasets: CPI, national accounts, labor, rates and exchange rates.
  https://data.imf.org/ifs
- BLS seasonal adjustment: adjusted CPI and revised seasonal factors.
  https://www.bls.gov/cpi/seasonal-adjustment/questions-and-answers.htm
- FRED real-time periods: information available at the stated historical date.
  https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- Nasdaq Data Link: adjusted/unadjusted prices and corporate actions.
  https://data.nasdaq.com/databases/SEP/documentation
- Kraken candle documentation: market symbols, timestamps and OHLC structure.
  https://docs.kraken.com/api/docs/futures-api/charts/candles

No measured observations or copyrighted source prose are copied into the corpus.
