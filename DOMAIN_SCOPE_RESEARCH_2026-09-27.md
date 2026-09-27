# Domain scope research — 2026-09-27

## Decision

Frame the paper and next dataset around **structured forecasting-requirements extraction for economic and financial applications**. Treat weather as a related covariate domain and a separately reported out-of-domain transfer set. Do not describe the system as a price-prediction or weather-prediction model: the 2B SLM converts a user's request into the 79-slot forecasting specification and asks the next clarification question.

This scope gives the paper a clear application story without pretending that one adapter has learned the numerical dynamics of every time series. Domain-aware evaluation is justified: recent finance benchmark work reports that generic forecasting scores do not necessarily reflect financially useful behavior, while broad archives report results separately by domain and frequency rather than treating all series as interchangeable.

The frozen v5 r1 corpus remains the general-domain baseline. It contains 1,540 scenarios across 11 equally weighted domains (140 each), including `weather`, but its finance category is `finance_budget`; it has no dedicated macroeconomics, equities, fixed income, foreign exchange, commodities, or crypto taxonomy. Do not rewrite v5 r1. A finance-focused successor should receive a new corpus and schema version.

## Recommended application taxonomy

Use five primary families. “All economic indexes” is neither finite nor scientifically controlled; sample representative targets within each family and record the family, market, frequency, horizon, calendar, and decision objective.

| Family | Suggested share | Representative targets and requests |
| --- | ---: | --- |
| Macroeconomy and policy | 25% | CPI/PPI/PCE inflation, GDP and industrial production, unemployment and payrolls, policy rates, yield curves, exchange rates, money and credit |
| Equities and funds | 20% | Index and individual-equity returns, adjusted close, volume, volatility, ETF flows, sector or portfolio hierarchies |
| Fixed income, FX, and commodities | 20% | Government yields, spreads, currency pairs, oil, gas, gold, metals and agricultural prices |
| Crypto assets | 15% | Spot price or return, volume, realized volatility, market capitalization, funding and open interest; explicit 24/7 calendar and quote currency |
| Corporate and operational finance | 20% | Revenue, expenses, cash flow, budgets, demand, inventory value, working capital and scenario planning |

Weather should appear in two forms:

1. **Economic covariate cases:** temperature, precipitation, storms and degree days used as external drivers of energy demand, crop prices, logistics, retail demand or inflation.
2. **A separate transfer set:** pure temperature, precipitation, wind and extreme-event requirements. Report these results independently from in-domain economic results.

## Why this boundary is defensible

- FRED already organizes official economic series into money/banking/finance, labor markets, national accounts, production, prices and other categories. Its catalog includes interest rates, exchange rates, CPI/PCE, commodities and a small cryptocurrency category. This supports a coherent macro-financial taxonomy rather than a single undifferentiated “economy” label: <https://fred.stlouisfed.org/categories>.
- The IMF publishes topic-specific datasets for CPI, labor, national accounts, exchange rates, effective exchange rates, balance of payments and related indicators, with SDMX APIs. It explicitly replaced the old monolithic IFS dataset with topic datasets: <https://data.imf.org/en/news/accessing> and <https://data.imf.org/en/Resource-Pages/IMF-API>.
- SEC EDGAR exposes company facts and filings for company fundamentals, including bulk archives. This is a suitable provenance reference for corporate-finance scenarios, subject to the SEC automated-access policy: <https://www.sec.gov/search-filings/edgar-application-programming-interfaces>.
- NOAA's GHCN-Daily provides quality-reviewed daily temperature, precipitation, snowfall and snow-depth observations from more than 100,000 stations in 180 countries and territories. This supports weather covariate and transfer scenarios, while its documented station and coverage limitations should be retained: <https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily>.
- Alpha Vantage documents separate stock, index, FX, crypto, commodity and economic-indicator endpoints. Kraken documents crypto candles and market analytics. They are useful for realistic source/API metadata, but their licensing, access tiers and rate limits must be checked before redistributing numeric snapshots: <https://www.alphavantage.co/documentation/> and <https://docs.kraken.com/api/docs/futures-api/charts/candles>.
- The M4 competition evaluated 100,000 series by both frequency and domain, with separate macro and finance groups. The Monash archive similarly preserves heterogeneous domain characteristics. This supports stratified reporting: <https://doi.org/10.1016/j.ijforecast.2019.04.014> and <https://forecastingdata.org/>.
- FinVerse argues for finance-specific targets and decision-aligned metrics because uniform generic forecast errors can miss economic usefulness. That lesson applies to the downstream numerical forecaster, even though this SLM is evaluated on requirement extraction: <https://arxiv.org/abs/2608.03259>.

## Dataset design

### Two distinct data layers

1. **SLM instruction corpus:** synthetic or human-authored conversations plus exact structured labels. These train intent, slot extraction, corrections, abstention and clarification. They should contain no live prices and no future information.
2. **Downstream numerical benchmark:** immutable, dated time-series snapshots used only to test a forecasting engine after the requirements are collected. This is a separate model and evaluation protocol.

Mixing these layers would make the paper's claim unclear. Market observations will not teach the chat model how to emit the correct JSON contract unless they are converted into supervised language-and-label examples.

### Size and composition for a finance-focused successor

- Target 1,200–1,500 source scenarios and roughly 5,000–6,500 SFT examples, comparable to v5 r1.
- Keep 3–4 fact-preserving renderings per source plus a clarification question; all variants, corrections and questions from one source remain in one split.
- Maintain roughly 50% medium/dense turns with 3–8 updates, 15–20% turns with 9 or more updates, 20–25% correction/conflict turns, and 10–15% abstention or deliberately insufficient requests.
- Use a 70/15/15 grouped train/validation/final split. Freeze it before rendering or paraphrasing.
- Hold out at least one template family and one entity group within every financial family. Do not let the same instrument, country-indicator pair or paraphrase family cross splits.
- Keep a separate weather transfer set outside model selection. It can be final-only or a second sealed evaluation set.

### Required coverage axes

Balance scenarios across:

- frequency: minute/hourly, daily, weekly, monthly and quarterly;
- horizon: one step, short multi-step and long-range;
- calendar: exchange trading, business day, calendar day and crypto 24/7;
- target transform: level, return, growth rate, spread, volatility and volume;
- structure: single series, panel and hierarchy;
- geography: multiple countries and exchanges;
- output: point, interval, quantile and named scenarios;
- events: policy decisions, earnings, splits/dividends, market closures, data revisions and regime changes;
- ambiguity: raw versus adjusted prices, nominal versus real values, local versus UTC timestamps, quote currency, release vintage and revised macro data.

### Domain fields

Most facts can use the existing 79 slots: `target_description`, `target_unit`, `series_id_columns`, `scope_filters`, `calendar_type`, `timezone`, `data_cutoff`, `known_regime_changes`, `external_covariate_sources`, `forecast_type` and the evaluation fields.

Before generating a new version, decide whether the paper needs explicit measurement of these finance-specific concepts:

- asset or indicator family;
- instrument/ticker and exchange;
- quote and base currency;
- raw versus split/dividend-adjusted value;
- macro release vintage and revision policy;
- return/level/volatility transform;
- market calendar and trading session.

If they are central claims, add typed optional fields in a new schema version. Hiding them inside free text makes domain accuracy hard to score. Do not retrofit them into frozen v5 labels.

## Evaluation design

Compare four systems on exactly the same finance cases and prompts:

1. stock Qwen3.5-2B;
2. the general-domain v5 adapter, if it passes its current review and behavior gates;
3. the new economy/finance adapter;
4. the pinned OpenAI comparator.

Primary SLM metrics remain exact non-intent slot micro F1, strict JSON/schema validity, correction accuracy, forbidden-slot rate and clarification-question accuracy. Add per-family and per-field results, plus a paired scenario-cluster bootstrap interval. Report weather-transfer results separately.

Do not use directional accuracy, returns, RMSE or trading profit to evaluate this SLM. Those belong to the downstream numerical forecaster. If that second model is added later, use time-based backtests, immutable as-of snapshots, strong naive baselines and domain-appropriate metrics, and account for revisions, corporate actions, trading calendars, survivorship bias and transaction costs.

## Recommended sequence

1. Finish the corrected v5 pilot and human-review gates so v5 remains a reproducible generalist baseline.
2. Freeze a versioned finance/economy taxonomy and decide whether the schema needs the typed fields above.
3. Build a small 100-scenario finance prototype and manually audit every case before scaling generation.
4. Build the full finance corpus as a new version, with its own manifests, hashes, review ledger and sealed final labels.
5. Train a separate adapter; do not continue training the v5 adapter on finance data because that would confound generalist and specialist comparisons.
6. Evaluate in-domain finance performance and weather transfer without using either final set for selection.

## Paper claim supported by this design

“A compact local language model can be specialized to convert natural-language economic and financial forecasting requests into validated structured requirements, while a held-out weather set measures transfer beyond the specialization domain.”

That claim is narrower, testable and aligned with what the current model actually does.
