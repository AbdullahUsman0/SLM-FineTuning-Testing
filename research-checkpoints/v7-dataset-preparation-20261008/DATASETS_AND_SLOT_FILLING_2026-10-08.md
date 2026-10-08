# Forecasting requirement datasets and slot filling experiments

Prepared for the FYP supervisor on 8 October 2026.

## What we have built

**Professional explanation:** We studied small language models that turn forecasting requests into structured requirements. We fine-tuned Qwen3.5 at two sizes, 0.8B and 2B, using successive datasets v1 through v6. The application validates the extracted updates and maintains conversation state. The numerical forecasting model is a later stage of the FYP.

**Roman Urdu:** Hamara SLM user ki baat se requirements nikalta hai. Yeh khud aglay haftay ki temperature ya stock price predict nahi karta. v1 se v6 datasets aur experiments ke versions hain; yeh chhay different base models nahi hain. Humne do model sizes use kiye hain.

**Example:** “I need daily maximum temperature observations forecast for the next 7 days, measured in Celsius.” The requirements include the target, observation frequency, horizon and units. The extractor should not generate seven temperature predictions or invent a data source.

**Main finding:** Fine-tuning greatly improved compliance with our output contract. Domain-focused v6 gives F1 point estimates close to OpenAI on the latest development conversations, but neither statistical superiority nor equivalence is established. Missing initial facts and imperfect accumulated state remain important problems. These findings motivate a carefully reviewed v7 candidate.

### How to read the measurements

- **Scenario:** one underlying request or conversation. Several turns or paraphrases can come from it.
- **SFT example:** a model input and the desired answer used in supervised fine-tuning. It is not necessarily a unique conversation.
- **JSON validity:** the output can be read as JSON.
- **Schema compliance:** the declared fields and types are allowed.
- **Parser admission:** the entire response passes our frozen Python validation policy, including duplicate-slot checks.
- **Exact slot F1:** measures the balance between correct extracted updates and missing or incorrect updates. The primary tuple is slot ID, value and status, excluding intent.
- **Component test:** supply the correct prior state before every turn.
- **Rollout test:** feed each model its own previous state; an initial mistake can persist.
- **Training loss and token accuracy:** predict reference tokens during training evaluation. They do not measure the percentage of complete requests understood correctly.

**Roman Urdu example:** JSON ke brackets theek ho sakte hain, magar horizon 7 ki jagah 70 ho sakta hai. Yeh format pass karega, meaning fail karega. F1 0.50 ka matlab yeh nahi ke aadhi conversations poori sahi huin.

<!-- PAGE -->
## Dataset inventory

| Dataset | Source scenarios | Train SFT | Validation SFT | Main change |
| --- | ---: | ---: | ---: | --- |
| v1 | 200 | 266 | 57 | First general forecasting corpus with extraction and questions |
| v2 | 320 | 420 | 90 | Short replies, conversational intent and Roman Urdu |
| v3 | 320 | 865 | 184 | Extraction-only training and many selected-slot answers |
| v4 | 420 reconstructed scenarios | Not recovered | Not recovered | New domains and API paraphrases; exact historical SFT missing |
| v5 | 1,540 | 4,435 | 948 | Grounded multi-slot extraction and complete 79-slot prompt |
| v6 | 2,000 | 5,800 | 1,253 | Weather and economics specialization |

V1 has 140/30/30 source scenarios across train/validation/test. V2 and v3 retain 224/48/48. The reconstructed v4 scenario files have 294/78/48, but they do not recover the missing paraphrases or historical training examples. V5 has 1,078/231/231 scenarios; v6 has 1,400/300/300. Final labels for v5/v6 are separate from public development data.

V5 has 6,336 total SFT examples including extraction and question tasks. V6 has 8,320. These totals include final examples in the separately held artifact inventory; the published train and validation SFT counts above do not include final examples.

**Important distinction:** v1-v4 were developed for 0.8B. We also trained 2B on the frozen v3 dataset before building the larger v5 and v6 datasets. The interrupted v4 run is not a completed model result. V5 and v6 were fresh adapters from the stock 2B base, rather than successive continuation of one adapter.

**Roman Urdu:** Dataset bara hone ka matlab sirf rows barhana nahi. Har version mein user ki baat, label aur training task badla. Ek scenario se kai examples bante hain, is liye scenarios aur training rows ko alag count karte hain.

<!-- PAGE -->
## Dataset v1 and dataset v2

### V1 establishing the first baseline

**What it contained:** 200 deterministic synthetic forecasting scenarios across ten behavior categories and twenty domain variants. Categories included complete requests, missing requirements, ambiguity, corrections, conflicts, multiple series, probabilistic forecasts, privacy, robustness and intent boundaries. Training included both requirement extraction and clarification questions. Source domains included weather, Bitcoin, sales, energy, staffing and operations.

**Professional explanation:** V1 established a working training and evaluation pipeline. A modest, controlled synthetic corpus was useful for verifying that LoRA could learn the output format.

**Roman Urdu:** Pehla dataset setup check karne ke liye tha. Humne dekha ke model user ki requirements aur follow-up question ka format seekh sakta hai ya nahi.

**Illustrative example:** “Use Lahore temperature data from a CSV upload, observed daily, and forecast 14 days.” A complete request needs several distinct updates. A missing-source variant should ask for the source rather than invent one.

**Where it lacked:** The corpus was small and used recurring synthetic language. Extraction and question generation shared the training budget. More epochs were not automatically better: recorded validation loss was lowest after epoch 1, approximately 0.2275, and increased to about 0.2947 after epoch 2. The original adapter and raw held-out behavior reports are missing, so no reconstructed v1 behavioral score is claimed.

### V2 adding conversational behavior

**What it contained:** V1's 200 scenarios plus 120 conversational-intent scenarios. New cases included “yes,” “yeah,” “haan,” short forecast requests, Roman Urdu, explicit negative intent, prior assistant questions and state progression. Compact prompts were used. There were 289 extraction and 221 question examples across train and validation.

**Professional explanation:** V2 broadened the ways a user can respond, but the model frequently returned structurally valid answers with few or no useful slot updates.

**Roman Urdu:** “Haan” ya chhota jawab samajhne ki training add hui. Lekin model kai dafa sahi JSON banata tha aur updates khaali chhor deta tha. Format seekhna, saari information nikalna nahi hota.

**Illustrative example:** Assistant: “Do you want to create a forecast?” User: “Haan.” Context resolves intent. Separately, “daily for seven days” requires frequency and horizon rather than an empty updates array.

**Where it lacked:** Mixed tasks and short replies could dominate the limited training data. The historical 16-scenario F1 of 0.0216 and empty-update collapse of 93.33% are recorded handoff claims, not freshly reproduced results with recovered raw files. Recorded validation loss improved to 0.1770, which did not establish reliable extraction.

<!-- PAGE -->
## Dataset v3 and dataset v4

### V3 focusing on extraction

**What it contained:** The same 320 source scenarios as v2, with 1,049 extraction-only prompt/completion examples. Across train and validation: 272 selected-slot short answers, 272 natural-sentence answers, 272 non-answers and 233 unique base extraction turns. Only the assistant completion contributed to the training loss. Questions were delegated to application/schema behavior.

**Professional explanation:** V3 clarified the model's job and removed question generation from training. However, much of the data rewarded zero or one update, while real initial requests require several updates.

**Roman Urdu:** Ab model ko sirf extraction sikhayi. Magar dataset mein bohat se examples aik slot ya koi slot nahi nikalne wale thay. Model ko ek jumlay se bohat si requirements nikalne ki practice kam mili.

**Illustrative example:** “Use weather.csv as an uploaded CSV, with timestamp date, daily observations and a 7-day horizon.” Returning only source_reference misses source_mode, file_format, time_column, frequency and horizon.

**Observed gap:** On the verified 48-scenario v3 validation cohort, 2B LoRA F1 was 0.0938, stock 2B was 0.0714 and OpenAI was 0.3409. The adapter recovered 12 of 234 gold non-intent updates. This is separate from the older 0.8B v3 F1 claim of 0.0751, whose raw report is absent. Token accuracy near 99.97% did not imply high slot recall.

### V4 attempting more language variation

**What it contained:** 320 previous scenarios plus 100 new-domain scenarios. The planned extraction-only SFT added OpenAI paraphrases of base messages. The historical handoff reported 2,991 examples, but the exact paraphrase cache and generated SFT files were not recovered.

**Professional explanation:** V4 addressed repetitive wording, but exact reproduction and semantic label preservation were not established. Training stopped after the first epoch according to the handoff.

**Roman Urdu:** Humne aik hi requirement ko mukhtalif alfaaz mein likhwaya. Lekin original paraphrase files nahi milin aur training ruk gayi. Is liye v4 ko successful completed improvement nahi keh sakte.

**Illustrative example:** “A week ahead” can express a seven-day horizon. A paraphrase that changes “daily observations” to “produce forecasts daily” changes the slot meaning and cannot retain the same frequency label without review.

**Where it lacked:** The builder retained original labels and used the entire paraphrase as evidence; structural containment alone could not prove that all facts survived. V7 must link each label to a fact and a current-message span, and retain reproducible source bytes. V4 has no completed behavioral score.

<!-- PAGE -->
## Dataset v5 and dataset v6

### V5 broad grounded extraction

**What it contained:** 1,540 scenarios across eleven broad domains, 140 per domain. There were 4,796 extraction examples and 1,540 question examples. About 50.9% of extraction turns supplied 3-8 updates and 16.5% supplied 9 or more; 352 correction turns, 594 short replies and 264 abstention turns were recorded. The prompt exposed all 79 schema slots, fixing the earlier 24-slot cap. All required slots had positive coverage.

**Professional explanation:** V5 targeted multi-slot omissions, current-message evidence and correction behavior. It was a substantial improvement on the engineered component validation task.

**Roman Urdu:** v5 mein humne aik user message se kai slots nikalna sikhaya. Model ko poori schema dikhayi aur kaha ke jo user ne nahi bataya woh fill na kare.

**Illustrative example:** “Observations are daily; predict fourteen days; I will upload readings.csv.” These are separate requirements, supported by separate phrases.

**Results and limitations:** V5 achieved 0.999454 non-intent F1 on its 231-scenario synthetic component validation, with 3,664 of 3,666 gold updates recovered. Later turns received correct prior context. On the fresh 31-conversation natural development comparison its F1 was 0.4213 component and 0.4270 rollout. Repeated clause templates, gold context and limited natural variation mean the synthetic score cannot establish real-user reliability. The recorded 6,663 flagged overlap pairs were pending review in the original audit; zero semantic leakage was not claimed. Prior user confirmation of v5 review does not automatically cover new v6/v7 labels or later failure queues.

### V6 specializing in weather and economics

**What it contained:** 2,000 scenarios across twenty subdomains: six weather, twelve economics and two combined. The agreed mixture was 600 weather, 1,200 economics and 200 combined. Examples cover CPI level versus inflation rate, real versus nominal growth, employment, rates, adjusted share prices, FX, commodities, spot crypto versus derivatives, housing, fiscal data, finance and weather-linked energy/agriculture. There were 8,320 extraction/question SFT examples. Entity groups and template families were split before generation. There were 78 positive slots; credential references were excluded from positive supervision because the prompt redacts them.

**Professional explanation:** V6 increased domain vocabulary and contrastive target distinctions while keeping the same 79-slot interface. It was also larger than v5, so the experiment does not isolate domain specificity as the cause of improvement.

**Roman Urdu:** v6 ko weather aur economy ke terms sikhaye. CPI index aur inflation rate, ya BTC spot aur futures, aik cheez nahi hain. Dataset bara bhi hua, is liye score ka farq sirf domain-specific training ki wajah se prove nahi hota.

**Illustrative example:** “Forecast CPI year-over-year inflation, not the CPI index level.” Preserve the stated target; do not silently substitute price levels or infer index units.

**Where it lacked:** V6 remained templated and largely English. Initial slot recall was still limited, some JSON responses failed, and early omissions survived horizon corrections. The public corpus status records machine validation with human review pending; a separate, documented independent adjudication is still needed for publication.

<!-- PAGE -->
## Latest fair comparison and what improved

The fresh comparison used gpt-5-mini-2025-08-07 and stock/v5/v6 Qwen3.5-2B on the same 31 development conversations, with 93 extraction calls per model per track. Weather had seven conversations; inflation, stocks and crypto had eight each. The frozen parser, labels and component contexts matched. There were 744 scored calls in total.

| Model | Component F1 | Rollout F1 | Rollout precision | Rollout recall |
| --- | ---: | ---: | ---: | ---: |
| Stock 2B | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v5 | 0.4213 | 0.4270 | 0.5429 | 0.3519 |
| v6 | 0.4818 | 0.4986 | 0.6312 | 0.4120 |
| OpenAI F | 0.4844 | 0.5086 | 0.4332 | 0.6157 |

**Professional explanation:** V6 is more conservative than OpenAI: fewer incorrect exact tuples, but more omitted gold facts. The two systems have close F1 estimates on this cohort. For v6 minus OpenAI, the paired 95% intervals are [-0.0912, +0.0895] component and [-0.0836, +0.0639] rollout. Both contain zero. This establishes neither superiority nor equivalence.

**Roman Urdu:** V6 kam extra ya ghalat updates karta hai, magar kuch sahi information bhi chhor deta hai. OpenAI zyada information pakarta hai aur extra updates bhi deta hai. Scores qareeb hain, lekin is chhote test se winner decide nahi hota.

| Diagnostic | v5 | v6 | OpenAI F |
| --- | ---: | ---: | ---: |
| JSON valid per rollout | 93/93 | 91/93 | 93/93 |
| Parser admitted per rollout | 93/93 | 90/93 | 93/93 |
| Valid unknown turns with no extra updates | 23/31 | 27/31 | 0/31 |
| Fully exact conversations | 0/31 | 0/31 | 0/31 |

**Example:** The model initially misses the unit. Later it correctly changes horizon from 7 to 8 days. The horizon correction succeeds, yet accumulated state is still missing the unit. V6 correctly changed rollout horizons and preserved other prior slots in 31/31 corrections; complete post-correction state remained wrong in 0/31 exact matches for both adapters.

The unknown rule requires omission, so an explicit “unknown” marker can fail without fabricating a real-world value. Stock's zero operational F1 was dominated by output-contract failures. Hosted and local runtimes differ; median extraction latency was roughly 2.7 seconds hosted versus 5.3 seconds v6, which is descriptive rather than a controlled compute-efficiency claim.

**Limits:** agent-authored and agent-reviewed posthoc development cases, only ten positive non-intent slots, no STT or live generated dialogue, no independent unseen-user validation. These are not final-test paper claims.

<!-- PAGE -->
## JSON Pydantic and the six approaches

**Professional explanation:** JSON is the output representation. JSON Schema describes allowed fields and types. Pydantic validates and creates Python objects. These work together. They are not competing alternatives. A schema-valid answer can still contain the wrong requirement.

**Roman Urdu:** JSON data ka format hai. Schema batati hai kin fields aur types ki ijazat hai. Pydantic Python mein unko check karta hai. Hum in sab ko mil kar use karte hain. Sahi format mein ghalat value phir bhi aa sakti hai.

**Example:** {"forecast_horizon":{"periods":70,"unit":"day"}} may be structurally acceptable but is wrong for a seven-day request. Our actual wire format stores candidate_value inside a string envelope; the parser decodes that content exactly once. V7 will emit JSON-encoded values consistently, including strings, while retaining the frozen provider-independent admission policy.

### A Prompted JSON plus Pydantic

**Professional:** Make one extraction request, parse and validate it, and count an unusable response as a failure. This is the reference output-method arm.

**Roman Urdu:** Model ko aik chance do. Jawab ka format aur schema check karo. Ghalat ho to failure record karo; chupke se answer theek nahi karo.

**Example and test:** “Observations are daily; forecast the next seven days.” Check whether both frequency and horizon are extracted, without invented source details. Measure exact F1, first-pass admission and latency.

### B Validation guided retry

**Professional:** Allow one further generation after a structural validation failure, providing the validation error. Count both attempts, cost and latency. A valid-but-wrong value does not automatically trigger this retry.

**Roman Urdu:** Agar required field missing ho ya structure ghalat ho to error bata kar aik aur chance dein. Agar 7 ki jagah 70 sahi structure mein aaye to validator ko meaning ki ghalti khud nahi pata chalegi.

**Example and test:** A response omits correction_detected; retry once. Measure recovered admission and whether the extra generation improves semantic extraction enough to justify its extra calls.

<!-- PAGE -->
## Constrained decoding and slot wise questions

### C Local JSON Schema constrained decoding

**Professional:** Use XGrammar during token generation to restrict the structure the local model can produce. Pydantic still validates the result afterwards. This is a local generation-method arm.

**Roman Urdu:** Answer bante waqt allowed structure se bahar ke tokens rok diye jate hain. Brackets aur fields sahi hone ka chance barhta hai, magar model ko sahi fact samajhna phir bhi zaroori hai.

**Example and test:** Prevent an undeclared debug field or native object in candidate_value. Then check whether correct horizon, unit and target recall improves or declines. A structurally valid eight-day horizon still fails if the user asked for seven.

**Result status:** XGrammar was not run as an OpenAI arm. OpenAI F is its hosted schema-constrained counterpart. A separate older local decoder diagnostic exists, but it used a different cohort/policy; its scores are not substituted into the hosted A-F table or the fresh four-model comparison.

### D Slot wise question answering

**Professional:** Use a control request and separate probes for the non-intent slots, then aggregate their results. The planned full design can approach 79 calls per extraction turn. The recorded pilot sometimes probed fewer slots after short-circuit conditions; report actual calls.

**Roman Urdu:** Har slot ke liye alag poochte hain: horizon diya? frequency di? source diya? Is se coverage check hoti hai, lekin calls, time aur cost bohat barh sakte hain.

**Example and test:** Probe observation frequency, forecast horizon and source independently. Keep absent information absent. Measure coverage, probe compliance, aggregate F1 and total cost.

**Measured scope:** D completed one weather conversation with three turns on each track, not the complete 32-conversation study. Component F1 was 0.0000 with 159 API calls; rollout F1 was 0.0000 with 237 API calls. Costs were approximately $0.1239 and $0.1714. The strict aggregator failed the whole extraction if a required probe failed. This tiny pilot and weak probe contracts do not establish that slot-wise QA is inherently inferior.

**Next requirement:** Before a larger D experiment, freeze explicit probe schemas and separately test strict versus partial-result aggregation. Do not retrofit the pilot scores after changing the contract.

<!-- PAGE -->
## Rules and OpenAI Structured Outputs

### E Deterministic Python rules

**Professional:** Extract explicit phrases using conservative regex and dictionaries, without a language model. The rules are cheap and reproducible but have limited wording and slot coverage.

**Roman Urdu:** Fixed Python rules se “next seven days” jaisi baat pakarte hain. Simple wording mil jaye to kaam ho jata hai, lekin mukhtalif tareeqay se boli gayi baat miss ho sakti hai.

**Example and test:** A horizon pattern handles “next 7 days.” Test “a week ahead,” a correction to ten days and the phrase “I haven't decided the horizon.” Compare exact recall and false positives.

**Measured results:** Full synthetic v5 validation: F1 0.1763, precision 0.6378, recall 0.1023, zero model calls, 948/948 structurally valid outputs. In the older 32-conversation natural study E scored 0.3030 on both tracks. These are different cohorts, so they are separate measurements. E is a zero-LLM control, not an OpenAI-powered extraction method.

### F OpenAI Structured Outputs

**Professional:** Use pinned gpt-5-mini-2025-08-07 with the declared output schema, then apply the same local validation policy. Generation-time schema enforcement helps structure but does not guarantee correct facts, evidence or unique slot IDs within the updates list.

**Roman Urdu:** OpenAI ko schema ke mutabiq jawab dene ke liye constrain karte hain. Baad mein hum apne Python rules se bhi check karte hain. Schema sahi hone se user ki meaning sahi nikalna automatically guarantee nahi hota.

**Example and test:** “Change only the horizon to eight days; leave everything else alone.” Expect one horizon update with provided status and correction_detected=true. Check unchanged-state preservation and extra updates.

**Fresh measured result:** On the reviewed 31-conversation cohort F1 was 0.4844 component and 0.5086 rollout. JSON and declared schema compliance were 93/93 per track; component admission was 91/93 due to two duplicate-slot responses, while rollout admission was 93/93. All 31 unknown turns violated the omission policy. Exact complete conversations were 0/31.

### J JSON mode as an additional control

J requests JSON syntax without the same strict schema restriction. It is an extra arm, not one of the original six A-F approaches. Test whether valid JSON still misses required fields or uses wrong types. Older natural F1 was 0.1379 component and 0.1875 rollout under the original wire policy.

<!-- PAGE -->
## The older OpenAI method experiment

This October 6 study used the same pinned OpenAI model for A/B/J/F, with E as rules and D as a limited pilot. Main natural comparisons had 32 conversations and 96 logical extraction calls per method per track. The study's original parser required every candidate_value string's contents to encode JSON, even for ordinary text. Keep these original recorded scores separate from the fresh 31-case policy study.

| Method | Component F1 | Rollout F1 | Component schema valid | Component wire valid |
| --- | ---: | ---: | ---: | ---: |
| A prompted JSON | 0.0801 | 0.2040 | 65.6% | 50.0% |
| B one retry | 0.1456 | 0.2096 | 90.6% | 58.3% |
| J JSON mode | 0.1379 | 0.1875 | 64.6% | 54.2% |
| F Structured Outputs | 0.0826 | 0.1181 | 100.0% | 46.9% |
| E Python rules | 0.3030 | 0.3030 | 100.0% | 100.0% |

C has no hosted XGrammar score. D is excluded from this table because its one-conversation pilot is not the same full cohort. A/B/J/F also ran a 16-scenario engineered schema-coverage diagnostic with component F1 0.4174/0.4275/0.3776/0.5379 respectively; E scored 0.1284 there. Structural percentages for that diagnostic include question calls; F1 concerns extraction. Do not mix its denominator with the natural study.

### Why the parser mattered

**Professional:** The exported schema permits a string envelope. The pinned Pydantic decoder preserves ordinary text that is not JSON. An additional research guard instead rejected such text. That guard was not an inherent Pydantic requirement or a failure of OpenAI's declared schema compliance.

**Roman Urdu:** OpenAI temperature ko plain string bana sakta tha. Schema aur Pydantic usko qabool karte thay, magar hamara extra checker reject karta tha. Is liye purane score mein model ki meaning aur hamare checker ki sakhti dono ka asar tha.

**Example:** candidate_value containing plain temperature can pass declared schema and Pydantic. JSON-quoted temperature also passes the extra encoding rule. V7 uses the consistently encoded form to avoid this mismatch.

| Recorded component responses | Original F1 | Posthoc admission audit F1 |
| --- | ---: | ---: |
| A | 0.0801 | 0.1394 |
| B | 0.1456 | 0.4547 |
| J | 0.1379 | 0.1815 |
| F | 0.0826 | 0.5253 |

This posthoc audit reused cached component responses; it is not fresh inference or a new rollout. B retained its original retry decisions. Use the fresh reviewed F comparison for current model claims. The older study cost approximately $1.45 for 1,667 paid responses; the later 186 fresh F calls cost about $0.16.

<!-- PAGE -->
## Flaws and requirements for v7

### Extract all explicitly stated facts

**Finding:** V6 rollout recall was 0.4120. The initial request remains an important source of omissions. **Requirement:** train balanced multi-slot initial messages, incremental additions and independently rendered variants, with a source fact ledger. Do not increase abstention so much that the model learns to return empty updates for ordinary requests.

**Roman Urdu and example:** User agar target, unit, frequency, horizon aur filename sab bataye to sab nikalna hai. Sirf horizon pakarna kaafi nahi. Har label ke saath uska asli phrase save karna hai.

### Preserve domain distinctions and temporal meaning

**Finding:** Weather/economic vocabulary and overlapping schema concepts still create errors. **Requirement:** include contrast families for CPI level versus inflation, spot versus derivatives, adjusted versus raw prices, observation frequency versus prediction cadence versus output granularity, and calendar versus trading periods. Keep every contrast family in one split.

**Roman Urdu and example:** “Data hourly hai, report daily chahiye aur model weekly retrain ho.” Yeh teen alag requirements hain. “Every hour” akela likha ho to uska matlab guess nahi karna.

### Teach omission and status without guessing

**Finding:** V6 handled 27/31 unknown turns safely, yet still had four failures; hosted unknown markers violated the current policy. **Requirement:** pair unknown, explicit negative, affirmation and actual-value cases. Replacement corrections are provided; explicit affirmation of an existing value is confirmed. Unset unknowns supply no update. Deleting a known value needs a separate protocol and is excluded from v7 until defined.

**Roman Urdu and example:** “Timezone nahi pata” par timezone fill nahi karna. “UTC hai” par UTC fill karna. “Haan, UTC sahi hai” par confirmation record karna. “10 din kar dein” correction hai, confirmation nahi.

### Repair state only with fresh evidence

**Finding:** Exact complete conversations were 0/31 for v5, v6 and OpenAI. **Requirement:** include long trajectories and explicit restatements after seeded prior-state mistakes. Evaluate correction value accuracy separately from whole-state accuracy. Some failures require application validation or clarification, not additional training alone.

**Roman Urdu and example:** Agar unit pehle miss hua ho aur user sirf horizon badle, model ko unit khud nahi banana. User dobara kahe “unit Celsius hai” tab unit repair karna hai.

### Separate structure from meaning

**Finding:** V6 had two malformed outputs and one duplicate-slot rejection per track. **Requirement:** all gold outputs must pass the frozen admission policy, contain unique slot IDs and use one-pass encoded values. Keep JSON, schema, admission, evidence, semantic and reducer checks separate. Bad assistant outputs are diagnostic fixtures, not SFT targets.

<!-- PAGE -->
## V7 preparation and the research decision

**Candidate design:** retain 30% weather, 60% economics and 10% combined, with 80% English, 10% Roman Urdu and 10% mixed language as confirmed by the user. Keep the same 79-slot schema. Use extraction-only SFT so the new training directly targets this model's production role. Questions remain application-driven. Full required-slot coverage remains necessary alongside common core slots.

**Professional explanation:** V7 should be an auditable candidate built from new fact plans, with explicit domain contrasts, current-turn evidence, correction trajectories, unknown controls and state-repair examples. It should not copy the 31 evaluated conversations into training. Improving a score on known diagnostics is development; publication needs independently annotated unseen conversations.

**Roman Urdu:** v7 mein naye examples banenge, wohi 31 test conversations training mein nahi daalenge. Model ko natural English aur Roman Urdu dono milenge. Pehle labels aur unclear examples insaan review karega, phir training aur same rules ke saath comparison hoga.

### Acceptance and training requirements

1. Freeze facts, source groups, language quotas and split assignments before rendering. Keep alternate phrasing, domain contrasts and shared entities together.
2. Validate all development labels against schema, one-pass parser, evidence spans and a separate reducer replay. Record exact duplicates and lexical/template overlaps honestly; shared synthetic patterns are not independent human language.
3. Review a stratified sample plus every ambiguous/conflicting, encoded-value, multilingual and state-repair family. Record reviewer identity, decisions and artifact hash. Existing v5 review does not approve v7.
4. Run the exact pinned Qwen tokenizer preflight. Set the sequence limit from measured lengths; do not silently truncate. Tokenizer measurement is distinct from structural validation.
5. Train a new Qwen3.5-2B adapter from the pinned stock revision first, retaining LoRA rank 16/alpha 32 and frequent complete checkpoints every 10 optimizer steps. Never overwrite v5/v6 adapters. Keep the current two-epoch recipe as an initial controlled option; select behavior on validation and report the selection rule.
6. Evaluate v6, v7 and hosted F on matched fresh development cases under a separately frozen v7 protocol. Report exact tuple F1, precision/recall, per-slot/language/domain coverage, admission, evidence grounding, unsupported updates and full-state success. Use paired conversation clusters and report uncertainty.
7. Retain a fresh independently annotated evaluation cohort and sealed final labels. Do not use final results to redesign data, prompts, parsing or checkpoints. Compare equal training-example budgets or ablations if claiming benefits from language/domain/error-focused changes.

**What to tell the teacher:** “We established that fine-tuning helps the extraction contract, and v6 is close to a hosted baseline on the current development cohort. The next dataset targets missing facts, ambiguity and state accumulation. We are measuring semantic accuracy separately from valid JSON and will validate on unseen reviewed conversations before making publication claims.”

**Roman Urdu:** “Humne format aur extraction dono improve kiye, lekin abhi complete conversations poori sahi nahi hain. Ab hum kami ko target karke naya dataset banayenge aur unseen reviewed data par check karenge.”

<!-- PAGE -->
## Evidence and reproducibility

Examples throughout this report are illustrative unless explicitly described as recorded observations. Historical handoff claims, recovered measurements, diagnostics and new proposals are identified separately. Dates name checkpoint records rather than a claim that every experiment happened on that date.

### Repository evidence

- corpus/README.md and corpus-v2/README.md: v1/v2 composition and split counts.
- corpus-v3/README.md: extraction-only counts and training contract.
- HANDOFF_AUDIT_2026-09-17.md: missing artifacts, reconstructed v4 scenarios, historical limitations and verified v3 comparison.
- corpus-v5/v5-20260919-r1/statistics.json: v5 mixture, counts and slot coverage.
- research-checkpoints/v5-base-final-validation-20261002/: completed synthetic validation evidence.
- corpus-v6/v6-20261002-r4/statistics.json and V6_WEATHER_ECONOMICS_PLAN.md: v6 taxonomy, source groups and counts.
- research-checkpoints/v6-training-20261003/: v6 completed training record and adapter provenance.
- STRUCTURED_OUTPUT_RESEARCH_CHECKPOINT_2026-09-29.md: original six A-F definitions and rule baseline.
- research-checkpoints/openai-output-methods-20261006/: original OpenAI method study, raw evidence, parser audit and limited D pilot.
- evaluation/comparison-policy-20261006/: frozen one-pass parser, annotation rubric and reviewed development cohort.
- research-checkpoints/policy-openai-slm-matched-20261008/: fresh four-model results, three paired comparisons and human review queue.

### Published checkpoints

Fresh matched checkpoint: https://github.com/AbdullahUsman0/SLM-FineTuning-Testing/tree/afa3a2b1b256709ef15d15dcd4014b50a569b4c8/research-checkpoints/policy-openai-slm-matched-20261008

Remote GPU result source: b809dda515e4484bdb036b4db9caffd294a1b264. Frozen comparison policy SHA-256: 558581a35a0abc592c59d6f30bc7a2a3c0542368bfee668ac3018800b702189e.

The original training dependency is fpy commit 04d52c015d1e3ecdefe92b87116f209361509b4b. V5/v6 use Qwen3.5-2B revision 15852e8c16360a2fea060d615a32b45270f8a8fc. Exact code, data, policy, tokenizer, adapter and runtime hashes should accompany every later run.

**Publication limit:** Requirements extraction, dialogue state tracking, schema validation and constrained output generation are established techniques. Our potential contribution is their measured combination for a local forecasting-requirements assistant with many lifecycle slots, evidence grounding and correction behavior. A targeted search is not proof of novelty, and these development results do not yet establish a final publication benchmark.
