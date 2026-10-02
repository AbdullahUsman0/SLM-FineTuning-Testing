"""Deterministic weather/economics requirements corpus; no model or live-data calls.

v6 retains the pinned v5 wire contract and 79 slots. Values originate in synthetic
fact tables; linguistic variants and entity groups never cross frozen splits.
Final examples are written only below the ignored sealed directory.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from local_slm_lab import corpus_v5 as base
from local_slm_lab.corpus import PROJECT_ROOT, _update
from local_slm_lab.component_eval import state_with_context
from forecasting_assistant.application.clarification import select_next_slot
from forecasting_assistant.application.state_reducer import apply_extraction
from forecasting_assistant.domain.models import ExtractorResult

VERSION = "v6-20261002-r1"
SEED = "v6-weather-economics-1"
TEMPLATES = {
    "train": ("direct", "planning_note", "email", "requirements_meeting"),
    "validation": ("analyst_ticket", "followup_brief"),
    "final": ("heldout_handover", "heldout_procurement"),
}
# Four target forms per subdomain. These are requested variables, never current
# measured values or provider claims. All named scopes/instruments are fictional.
PROFILES = (
    ("temperature", "weather", "weather analyst", "calendar", "day", (
        ("daily maximum air temperature", "degrees Celsius", "tmax_c"),
        ("daily minimum air temperature", "degrees Celsius", "tmin_c"),
        ("daily mean air temperature", "degrees Fahrenheit", "tmean_f"),
        ("hourly air temperature", "degrees Celsius", "temp_c"))),
    ("precipitation", "weather", "water resources planner", "calendar", "day", (
        ("daily accumulated rainfall", "millimetres", "rain_mm"),
        ("hourly rainfall accumulation", "millimetres", "hourly_rain_mm"),
        ("daily snowfall depth", "centimetres", "snow_cm"),
        ("weekly accumulated precipitation", "millimetres", "weekly_precip_mm"))),
    ("wind", "weather", "wind operations analyst", "calendar", "hour", (
        ("hourly mean wind speed", "metres per second", "wind_ms"),
        ("hourly maximum gust speed", "metres per second", "gust_ms"),
        ("daily mean wind speed", "kilometres per hour", "wind_kmh"),
        ("hourly wind direction", "degrees clockwise from north", "wind_direction"))),
    ("humidity_pressure", "weather", "meteorological analyst", "calendar", "hour", (
        ("hourly relative humidity", "percent", "rh_pct"),
        ("hourly dew point temperature", "degrees Celsius", "dewpoint_c"),
        ("hourly sea-level pressure", "hectopascals", "pressure_hpa"),
        ("daily mean relative humidity", "percent", "daily_rh_pct"))),
    ("solar_cloud", "weather", "solar resource analyst", "calendar", "hour", (
        ("hourly global horizontal irradiance", "watts per square metre", "ghi_wm2"),
        ("hourly cloud cover", "percent", "cloud_pct"),
        ("daily sunshine duration", "hours", "sunshine_hours"),
        ("hourly direct normal irradiance", "watts per square metre", "dni_wm2"))),
    ("weather_extremes", "weather", "resilience planning analyst", "calendar", "day", (
        ("daily heatwave indicator defined as maximum temperature above 35 Celsius", "0 or 1", "heatwave_flag"),
        ("daily heavy-rain indicator defined as rainfall above 50 millimetres", "0 or 1", "heavy_rain_flag"),
        ("weekly count of frost days defined as minimum temperature below 0 Celsius", "days", "frost_days"),
        ("daily high-wind indicator defined as gust speed above 25 metres per second", "0 or 1", "gust_flag"))),
    ("inflation_prices", "economics", "inflation research analyst", "calendar", "month", (
        ("headline CPI index level, not its growth rate", "index points, base 2020=100", "cpi_index"),
        ("headline CPI year-over-year inflation rate, not the index level", "percent year over year", "cpi_yoy"),
        ("seasonally adjusted monthly core CPI change", "percent month over month", "core_cpi_mom_sa"),
        ("producer price index level, not consumer prices", "index points, base 2020=100", "ppi_index"))),
    ("output_growth", "economics", "macroeconomic research analyst", "calendar", "quarter", (
        ("real GDP quarterly growth, not nominal GDP", "percent quarter over quarter", "real_gdp_growth"),
        ("nominal GDP level in current prices", "millions of USD", "nominal_gdp"),
        ("industrial production index level", "index points, base 2020=100", "industrial_index"),
        ("real household consumption growth", "percent year over year", "consumption_growth"))),
    ("labor_employment", "economics", "labor market analyst", "calendar", "month", (
        ("unemployment rate", "percent of labor force", "unemployment_pct"),
        ("monthly payroll employment change", "thousands of jobs", "payroll_change"),
        ("labor force participation rate", "percent of working-age population", "participation_pct"),
        ("average hourly earnings growth", "percent year over year", "earnings_growth"))),
    ("rates_fixed_income", "economics", "monetary and fixed-income analyst", "business_day", "day", (
        ("policy interest rate level", "percent per year", "policy_rate"),
        ("investment-grade credit spread", "basis points", "credit_spread_bp"),
        ("bank lending growth", "percent year over year", "lending_growth"),
        ("mortgage interest rate level", "percent per year", "mortgage_rate"),
        ("10-year government bond yield level", "percent per year", "yield_10y"),
        ("10-year minus 2-year yield spread", "basis points", "curve_spread_bp"),
        ("corporate bond clean price excluding accrued interest", "percent of par", "clean_price"),
        ("bond total return", "percent", "bond_return_pct"))),
    ("equities_funds", "economics", "equity research analyst", "trading", "day", (
        ("split-and-dividend-adjusted closing share price", "USD per share", "adjusted_close"),
        ("raw closing share price without corporate-action adjustment", "USD per share", "raw_close"),
        ("daily total return including dividends", "percent", "total_return_pct"),
        ("daily ETF trading volume", "shares", "etf_volume"))),
    ("foreign_exchange", "economics", "currency research analyst", "business_day", "day", (
        ("EUR/USD spot rate quoted as USD per EUR", "USD per EUR", "eurusd_spot"),
        ("USD/JPY spot rate quoted as JPY per USD", "JPY per USD", "usdjpy_spot"),
        ("trade-weighted real effective exchange-rate index", "index points, base 2020=100", "reer_index"),
        ("daily EUR/USD log return", "percent", "eurusd_log_return"))),
    ("commodities", "economics", "commodity research analyst", "trading", "day", (
        ("spot crude-oil price, not a futures contract", "USD per barrel", "oil_spot"),
        ("spot gold price", "USD per troy ounce", "gold_spot"),
        ("wheat cash price", "USD per metric tonne", "wheat_cash"),
        ("natural-gas front-month settlement price", "USD per MMBtu", "gas_settlement"))),
    ("crypto_spot", "economics", "digital-asset research analyst", "calendar", "hour", (
        ("BTC/USD spot closing price on a 24/7 UTC series", "USD per BTC", "btc_usd_close"),
        ("ETH/USD spot traded volume in base asset units", "ETH", "eth_volume"),
        ("BTC/USD hourly log return", "percent", "btc_log_return"),
        ("digital-asset market capitalization", "millions of USD", "crypto_market_cap"))),
    ("crypto_derivatives", "economics", "derivatives research analyst", "calendar", "hour", (
        ("BTC perpetual-swap funding rate per eight-hour funding interval", "percent per 8 hours", "btc_funding_pct"),
        ("ETH perpetual-futures open interest", "contracts", "eth_open_interest"),
        ("BTC annualized realized volatility from past returns", "percent per year", "btc_realized_vol"),
        ("BTC futures-minus-spot basis", "basis points", "btc_basis_bp"))),
    ("housing_property", "economics", "housing research analyst", "calendar", "month", (
        ("residential house-price index level", "index points, base 2020=100", "house_price_index"),
        ("monthly residential rent", "USD per dwelling", "monthly_rent"),
        ("housing starts count", "dwellings", "housing_starts"),
        ("housing vacancy rate", "percent", "vacancy_pct"))),
    ("trade_fiscal", "economics", "trade and fiscal analyst", "calendar", "month", (
        ("goods trade balance, exports minus imports", "millions of USD", "trade_balance"),
        ("government tax revenue", "millions of USD", "tax_revenue"),
        ("government fiscal balance, revenue minus expenditure", "millions of USD", "fiscal_balance"),
        ("goods export volume index", "index points, base 2020=100", "export_volume"))),
    ("corporate_finance", "economics", "financial planning analyst", "calendar", "month", (
        ("net operating cash flow, not revenue", "thousands of USD", "operating_cash_flow"),
        ("recognized sales revenue, not cash collections", "thousands of USD", "recognized_revenue"),
        ("operating expenditure", "thousands of USD", "operating_expense"),
        ("working-capital change", "thousands of USD", "working_capital_change"))),
    ("weather_energy", "combined", "energy planning analyst", "calendar", "hour", (
        ("electricity demand with weather covariates", "megawatts", "load_mw"),
        ("electricity spot price with temperature covariates", "USD per MWh", "power_price"),
        ("natural-gas demand with heating-degree-day covariates", "MMBtu", "gas_demand"),
        ("solar generation with irradiance covariates", "megawatts", "solar_mw"))),
    ("weather_agriculture", "combined", "agricultural market analyst", "calendar", "week", (
        ("wheat cash price with precipitation covariates", "USD per metric tonne", "wheat_weather_price"),
        ("crop shipment demand with rainfall covariates", "metric tonnes", "crop_shipments"),
        ("irrigation expenditure with temperature covariates", "thousands of USD", "irrigation_cost"),
        ("crop-yield index with soil-moisture covariates", "index points, base 2020=100", "yield_index"))),
)
PROFILE = {p[0]: p for p in PROFILES}


def category_for(index, count):
    position = index * 100 // count
    for limit, category in ((50, "medium"), (68, "dense"), (78, "correction"),
                            (84, "confirmed_conflict"), (92, "selected_short"), (100, "abstention")):
        if position < limit:
            return category


def freeze_plan(per_domain=100, version=VERSION):
    if per_domain != 20 and (per_domain < 100 or per_domain % 100):
        raise ValueError("use a multiple of 100 for exact group quotas, or 20 for miniature fixtures")
    assignments = []
    for domain, family, *_ in PROFILES:
        # Five scenarios share an entity; allocate entities, not utterances.
        groups = sorted(range(per_domain // 5), key=lambda g: base.digest([SEED, domain, g]))
        n_train, n_val = len(groups) * 7 // 10, len(groups) * 15 // 100
        # Multiples of 100 yield exact 70/15/15; miniature 20 uses 50/25/25.
        if per_domain < 100:
            n_train, n_val = max(1, len(groups) - 2), 1
        split_by_group = {g: "train" if j < n_train else "validation" if j < n_train+n_val else "final"
                          for j, g in enumerate(groups)}
        for i in range(per_domain):
            split = split_by_group[i // 5]
            pools = TEMPLATES[split]
            assignments.append({"scenario_id": f"{version}/{domain}/{i:03d}", "domain": domain,
                "domain_family": family, "index": i, "entity_group": f"{domain}/entity-{i//5:03d}",
                "split": split, "category": category_for(i, per_domain),
                "template_family": pools[i % len(pools)]})
    return {"version": version, "schema_version": "1.0.0", "fpy_commit": base.FPY_COMMIT,
        "prompt_version": base.PROMPT_VERSION, "seed": SEED, "per_domain": per_domain,
        "status": "entity_groups_and_template_families_frozen_before_rendering",
        "mixture": {"weather": .30, "economics": .60, "combined": .10},
        "assignments": assignments, "template_pools": TEMPLATES}


def source_values(row, schema=None):
    schema = schema or base.load_schema()
    domain, i = row["domain"], row["index"]
    _, family, role, calendar, default_unit, targets = PROFILE[domain]
    description, unit, column = targets[row.get("target_variant", i % len(targets))]
    entity = f"{('Alder','Birch','Cedar','Juniper','Willow')[i//5 % 5]} synthetic {domain.replace('_',' ')} scope {i//5+1}"
    # Start with complete canonical type coverage; replace all operational values.
    values = base.source_values("weather", i, schema)
    freq_unit = ("hour" if description.startswith("hourly") else "day" if description.startswith("daily")
                 else "week" if description.startswith("weekly") else "month" if "industrial production" in description else default_unit)
    negative = any(term in description for term in ("temperature", "dew point", "growth", "change", "return", "spread", "balance", "cash flow", "funding rate", "basis", "spot price with"))
    if domain == "precipitation" or domain == "wind":
        negative = False
    bounds = {"min": -100 if negative else 0, "max": 1000000}
    if unit == "percent" and ("humidity" in description or "cloud cover" in description):
        bounds = {"min": 0, "max": 100}
    if unit == "0 or 1":
        bounds = {"min": 0, "max": 1}
    if "wind direction" in description:
        bounds = {"min": 0, "max": 360}
    values.update({
        "target_description": f"{description} for {entity}", "target_column": column,
        "target_unit": unit, "target_bounds": bounds, "allow_negative_values": negative,
        "problem_statement": f"forecast {description} for {entity} without assuming missing specifications",
        "business_goal": f"prepare the {entity} {('capacity buffer','refresh schedule','stress scenario','planning baseline','resource envelope')[i%5]} outlook",
        "stakeholder_role": role, "decision_supported": f"set the {entity} scenario planning allowance",
        "success_criteria": f"MAE no more than {i%5+1} {unit} for {entity}",
        "frequency": {"periods": 1, "unit": freq_unit},
        "forecast_horizon": {"periods": (6 if freq_unit in ('month','quarter') else 24 if freq_unit=='hour' else 14)+i%5, "unit": freq_unit},
        "calendar_type": calendar, "business_days_only": calendar in {"business_day", "trading"},
        "timezone": "UTC", "time_column": "observation_time", "source_mode": ("upload","api","database","catalog")[i%4],
        "file_format": ("csv","json","parquet","xlsx")[i%4],
        "source_provider": "Synthetic Weather Laboratory" if family=="weather" else "Synthetic Economic Research Laboratory",
        "sheet_or_table": f"{domain}_observations", "series_id_columns": ["scope_id"],
        "hierarchy_columns": ["region_id", "scope_id"], "aggregation_level": "scope",
        "scope_filters": [{"column": "scope_id", "operator": "eq", "value": entity}],
        "geography": [f"Fictional {entity} region"], "dataset_type": "single_series",
        "aggregation_method": "mean" if family=="weather" else "last",
        "forecast_start": "2026-10-01T00:00:00+00:00", "data_cutoff": "2026-09-30T00:00:00+00:00",
        "history_start": "2020-01-01T00:00:00+00:00", "history_end": "2026-09-30T00:00:00+00:00",
        "expected_history_length": {"periods": 81, "unit": "month"},
        "minimum_training_points": 24 if freq_unit in {'month','quarter'} else 365,
        "known_regime_changes": [{"date": "2025-01-01", "description": "synthetic measurement methodology change"}],
        "known_seasonality": bool(i%2), "seasonal_periods": [24,168] if freq_unit=='hour' else [12] if freq_unit=='month' else [4] if freq_unit=='quarter' else [7,365],
        "holidays": ["synthetic market closure"] if calendar=='trading' else ["synthetic reporting holiday"],
        "special_events": [{"date": "2026-10-05", "name": "synthetic weather alert" if family=='weather' else "synthetic scheduled data release"}],
        "past_covariates": ["observed_temperature"] if family=='combined' else ["lagged_target"],
        "known_future_covariates": ["announced_calendar_event"], "static_features": ["scope_elevation"] if family=='weather' else ["scope_sector"],
        "covariate_availability": [{"name":"announced_calendar_event", "available":"announced before cutoff"}],
        "external_covariate_sources": [{"name":"observed_temperature", "source":"synthetic_station_archive"}] if family=='combined' else [{"name":"announced_calendar_event", "source":"synthetic_calendar"}],
        "scenario_forecasts": [{"name":"stress", "description":"synthetic extreme-weather scenario" if family=='combined' else "synthetic planning sensitivity"}],
        "output_granularity": f"{freq_unit}ly per scope" if freq_unit in {'hour','month'} else f"one row per {freq_unit} per scope",
        "privacy_constraints": ["synthetic data only", "no external redistribution"],
        "license": "synthetic examples released under CC0-1.0; no provider data redistributed",
        "destination": f"https://planning.example/{domain}/{i//5+1}",
        "refresh_frequency": {"periods":1,"unit":freq_unit}, "prediction_frequency": {"periods":1,"unit":freq_unit},
        "retraining_frequency": {"periods":1,"unit":"month"}, "lead_time": {"periods":1,"unit":freq_unit},
        "test_window": {"periods":3 if freq_unit=='month' else 7,"unit":freq_unit},
        "contains_sensitive_data": bool(i%2), "human_approval_required": bool(i%2), "provenance_required": bool(i%2),
    })
    values["source_reference"] = (f"synthetic_{domain}_{i//5+1}.{values['file_format']}" if values["source_mode"]=="upload"
        else f"https://data.example/{domain}/{i//5+1}" if values["source_mode"]=="api"
        else f"synthetic_{values['source_mode']}:{domain}:{i//5+1}")
    values["authentication_reference"] = "secret://synthetic/not-for-supervision"
    return values


NATURAL = {
    "target_description": ("Forecast {v}", "The quantity I need predicted is {v}", "Make predictions for {v}"),
    "target_column": ("Read the target from the {v} column", "In my table, {v} is the target field", "Use field {v} as the prediction target"),
    "target_unit": ("Report the target in {v}", "The target is measured in {v}", "Keep the target's units as {v}"),
    "frequency": ("Each observation covers {v}", "The sampling interval is {v}", "There is one observation every {v}"),
    "forecast_horizon": ("Predict the next {v}", "I need an outlook covering {v}", "Extend the forecast {v} into the future"),
    "source_reference": ("Use my data source {v}", "The source is {v}", "Read the observations from {v}"),
    "calendar_type": ("Use a {v} calendar", "The time axis follows the {v} calendar", "Apply calendar type {v}"),
    "data_cutoff": ("Use only data available through {v}", "The as-of cutoff is {v}", "Exclude observations later than {v}"),
    "past_covariates": ("Only historical observations of {v} are available", "The past-only regressors are {v}", "Treat {v} as observed-past covariates"),
    "known_future_covariates": ("These future regressors are already known: {v}", "I can supply {v} ahead of time", "Use {v} as known-future covariates"),
    "primary_metric": ("Judge forecast accuracy with {v}", "My primary error measure is {v}", "Rank candidates using {v}"),
    "forecast_type": ("I want {v} forecasts", "The forecast output should be {v}", "Give the predictions in {v} form"),
    "output_format": ("Export results as {v}", "Return the forecast in {v} format", "Save the predictions as {v}"),
}
PREFIX = {
    "direct": ("", "", ""),
    "planning_note": ("Planning note: ", "The plan requires: ", "Record this in the plan: "),
    "email": ("Hello, ", "My request is as follows: ", "For my forecast request, "),
    "requirements_meeting": ("In the meeting I requested: ", "My agreed requirement is: ", "Please capture my request: "),
    "analyst_ticket": ("Analyst ticket: ", "Ticket acceptance detail: ", "This ticket specifies: "),
    "followup_brief": ("The follow-up brief says: ", "As the brief records, ", "From the follow-up requirements: "),
    "heldout_handover": ("The incoming analyst states: ", "In the handover, the analyst requested: ", "The handover instruction reads: "),
    "heldout_procurement": ("The procurement brief requires: ", "According to the procurement requirement, ", "The requested deliverable specifies: "),
}


def render_turn(facts, *, variant, family, category, selected_slot=None, boundary="unknown", domain):
    parts, spans = [], []
    if category == "correction":
        parts.append(("Correction: replace the previously confirmed requirement.", "I am changing my earlier confirmed specification.", "Please overwrite the old setting with this correction.")[variant])
    elif category == "prior_confirmation":
        parts.append("I confirm these forecasting requirements.")
    elif category == "abstention":
        topic = domain.replace("_", " ")
        options = {
            "unknown": (f"For the {topic} project, I have not decided the frequency, units, horizon, or source.",
                        "Those forecasting details are unknown; ask me before filling them.",
                        "I cannot specify the remaining forecasting settings yet."),
            "unsupported": ("Write a poem about this topic instead of forecasting.", "Create a poem; do not create a forecast.", "I need a poem rather than a forecasting task."),
            "not_forecasting": (f"Explain what {topic} means; no forecast is requested.", "Define the terminology only; do not predict a time series.", "I only want a glossary explanation, not predictions."),
            "privacy": ("I cannot provide the access policy or credentials; do not invent them.", "Access details are unknown. Ask the owner for an authorized reference.", "I have no approved access reference or privacy policy yet."),
        }
        parts.append(PREFIX[family][variant] + options[boundary][variant])
    ordered = list(facts)
    if variant == 1:
        ordered.reverse()
    elif variant == 2 and ordered:
        ordered = ordered[1:]+ordered[:1]
    for fact in ordered:
        slot, value = fact["slot_id"], fact["value"]
        if category == "selected_short":
            answer = ("yes" if value else "no") if isinstance(value,bool) else base.surface(value)
            text = (answer, "My answer is "+answer, "Please use "+answer)[variant]
        elif slot == "intent":
            text = ("Please build a time-series forecast", "I want to create a forecast", "Set up a forecasting task")[variant]
        elif slot in base.BOOL_CLAUSES:
            text = base.BOOL_CLAUSES[slot][int(value)]
        else:
            fmt = NATURAL.get(slot, (base.CLAUSES[slot],)*3)[variant]
            # Bare string values are natural exact text; list/object values retain
            # explicit JSON so the requested structure is unambiguous.
            text = fmt.format(v=value if isinstance(value,str) else base.surface(value))
        text = PREFIX[family][variant] + text + "."
        start = len(" ".join(parts)) + bool(parts)
        parts.append(text)
        spans.append({"source_fact_id":fact["source_fact_id"], "slot_id":slot,
                      "start":start, "end":start+len(text), "text":text})
    return " ".join(parts), spans


def make_turn(row, facts, context, variant, category, turn_id, parent, selected=None, boundary="unknown"):
    message, spans = render_turn(facts, variant=variant, family=row["template_family"], category=category,
                                selected_slot=selected, boundary=boundary, domain=row["domain"])
    by_id = {f["source_fact_id"]:f for f in facts}
    intent = boundary if category=="abstention" and boundary in {"unsupported","not_forecasting"} else "create_forecast"
    return {"message":message, "gold_extraction":{"intent":intent,"intent_confidence":1.0,
        "updates":[_update(s["slot_id"],by_id[s["source_fact_id"]]["value"],s["text"]) for s in spans],
        "correction_detected":category=="correction","unsupported_claims":[]},
        "context_slots":context, "source_fact_ids":[f["source_fact_id"] for f in facts], "evidence_spans":spans,
        "turn_id":turn_id, "parent_id":parent, "variant":variant, "render_category":category,
        "selected_slot":selected, "boundary":boundary, "template_family":row["template_family"],
        "origin":"fictional_synthetic_fact_table", "language":"en", "reviewer_status":base.REVIEW_STATUS}


CORE = ("target_description","target_unit","frequency","forecast_horizon","source_reference","target_column","calendar_type")
CORRECTIONS = ("forecast_horizon","target_unit","target_description","calendar_type","frequency","data_cutoff")


def build_scenario(row, version, schema=None):
    schema = schema or base.load_schema()
    sid,i,category = row["scenario_id"],row["index"],row["category"]
    values = source_values(row,schema)
    all_slots = [s.slot_id for s in schema.slots if s.slot_id!="intent"]
    eligible = [s for s in all_slots if s!="authentication_reference"]
    context,prior,selected = {},[],None
    if category in {"medium","dense"}:
        n = 3+i%6 if category=="medium" else 9+i%6
        # A domain-rich core alternates with full lifecycle coverage. Dense
        # requests always specify target, units, frequency, and horizon together.
        slots = list(CORE[:4])+["business_goal"] if category=="dense" else ["target_description",CORE[1+i%6],"business_goal"]
        if row["domain_family"]=="combined":
            slots.extend(["past_covariates","external_covariate_sources"])
        offset=(i*11+list(PROFILE).index(row["domain"])*17)%len(eligible)
        for slot in eligible[offset:]+eligible[:offset]:
            if len(slots)>=n:
                break
            if slot not in slots:
                slots.append(slot)
        if "hierarchy_columns" in slots:
            values["dataset_type"]="hierarchical"
        elif "series_id_columns" in slots:
            values["dataset_type"]="panel"
        if "seasonal_periods" in slots:
            values["known_seasonality"]=True
        if "quantiles" in slots or "prediction_interval_levels" in slots:
            values["forecast_type"]="probabilistic"
        slots.insert(0,"intent")
    elif category in {"correction","confirmed_conflict"}:
        corrected=CORRECTIONS[i%len(CORRECTIONS)]
        # Unit changes use explicit alternative scales rather than a fabricated
        # generic unit. Domains without an unambiguous alternative correct the
        # source reference instead.
        unit_changes={"degrees Celsius":"degrees Fahrenheit","degrees Fahrenheit":"degrees Celsius",
            "millimetres":"centimetres","centimetres":"millimetres","metres per second":"kilometres per hour",
            "kilometres per hour":"metres per second","hectopascals":"kilopascals"}
        if corrected=="target_unit" and values[corrected] not in unit_changes:
            corrected="source_reference"
        prior_slots=list(dict.fromkeys(["intent","target_description","business_goal",corrected]))
        prior=[base.atom(sid,"t1",s,values[s]) for s in prior_slots]
        context={f["slot_id"]:f["value"] for f in prior}
        if corrected in {"forecast_horizon","frequency"}:
            values[corrected]={**values[corrected],"periods":values[corrected]["periods"]+2}
        elif corrected=="target_unit":
            values[corrected]=unit_changes[values[corrected]]
        elif corrected=="target_description":
            values[corrected]=source_values({**row,"target_variant":(i+1)%len(PROFILE[row['domain']][-1])},schema)[corrected]
        elif corrected=="calendar_type":
            values[corrected]="custom" if values[corrected]!="custom" else "calendar"
        elif corrected=="source_reference":
            values[corrected]=f"https://replacement.example/{row['domain']}/{i//5+1}"
        else:
            values[corrected]="2026-09-15T00:00:00+00:00"
        slots=[corrected]
    elif category=="selected_short":
        selected=("target_column","time_column","forecast_horizon","frequency","file_format","source_reference","target_unit","contains_sensitive_data")[i%8]
        if selected=="file_format":
            values["source_mode"]="upload"
            values["source_reference"]=f"synthetic_{row['domain']}_{i//5+1}.{values['file_format']}"
        context={"intent":values["intent"],"target_description":values["target_description"],"business_goal":values["business_goal"]}
        if selected!="source_reference":
            context["source_reference"]=values["source_reference"]
        for _ in schema.slots:
            choice=select_next_slot(schema,state_with_context(schema,context))
            if choice is not None and choice.slot_id==selected:
                break
            if choice is None:
                raise ValueError("short answer cannot be reached")
            context[choice.slot_id]=values[choice.slot_id]
        else:
            raise ValueError("short-answer selector did not converge")
        slots=[selected]
    else:
        slots=[]
        context={"intent":"create_forecast","target_description":values["target_description"],"business_goal":values["business_goal"]}
    main_id="t2" if prior else "t1"
    facts=[base.atom(sid,main_id,s,values[s]) for s in slots]
    context_facts=[base.atom(sid,"context",s,v,"confirmed_context") for s,v in context.items()]
    turns=[]
    if prior:
        turns.append(make_turn(row,prior,{},0,"prior_confirmation","t1-v0",None))
        turns[-1]["confirmation_event"]="user confirms stated t1 values before t2"
    for variant in range(3):
        turns.append(make_turn(row,facts,context,variant,category,f"{main_id}-v{variant}",
            f"{sid}/t1-v0" if prior else sid,selected,("unknown","unsupported","not_forecasting","privacy")[i%4]))
    last=turns[-1]
    reduced=apply_extraction(base.turn_state(last,schema),ExtractorResult.model_validate(last["gold_extraction"]),schema,2 if prior else 1,last["message"])
    gold={s:json.loads(base.dumps(v.value)) for s,v in reduced.slots.items() if v.value is not None}
    if category=="confirmed_conflict":
        question_slot=slots[0]
    elif last["gold_extraction"]["intent"]!="create_forecast":
        question_slot="intent"
    else:
        question_slot=next(s for s in base.QUESTION_SLOTS if s not in gold)
    mentioned={f["slot_id"] for f in facts+prior}
    return {"scenario_id":sid,"source_fact_id":f"{sid}/fact-table","corpus_version":version,
        "schema_version":schema.version,"split":row["split"],"domain":row["domain"],
        "domain_family":row["domain_family"],"entity_group":row["entity_group"],"category":category,
        "template_family":row["template_family"],"language":"en","origin":"fictional_synthetic_fact_table",
        "reviewer_status":base.REVIEW_STATUS,"parent_id":None,"turns":turns,"source_facts":prior+facts+context_facts,
        "gold_intent":last["gold_extraction"]["intent"],"gold_final_slots":gold,
        "gold_final_statuses":{s:v.status.value for s,v in reduced.slots.items() if v.value is not None},
        "expected_question_slot":question_slot,"ideal_question":(
            "Which rule determines the valid dates or time periods for this series?" if question_slot=="calendar_type"
            else schema.get(question_slot).static_question),
        "must_not_infer":sorted(set(all_slots)-mentioned),
        "behavior_tags":[category,"literal_evidence","domain_specific","no_live_observations","no_invention"],
        "question_context_confirmation_event":"user confirms nonconflicting supplied values before question fixture",
        "context_provenance":"explicit t1 confirmation" if prior else "synthetic previously confirmed facts"}


def sft_examples(scenario,schema=None):
    for example in base.sft_examples(scenario,schema):
        example["metadata"].update(domain_family=scenario["domain_family"],entity_group=scenario["entity_group"],
                                   corpus_version=scenario["corpus_version"])
        yield example


def validate_scenario(scenario,schema=None):
    if scenario["domain"] not in PROFILE or scenario["domain_family"]!=PROFILE[scenario["domain"]][1]:
        raise ValueError("domain taxonomy mismatch")
    base.validate_scenario(scenario,schema,renderer=partial(render_turn,domain=scenario["domain"]))


def validate_collection(scenarios,plan,schema=None):
    schema=schema or base.load_schema()
    frozen={r["scenario_id"]:r for r in plan["assignments"]}
    ids=set(); exact={}; normalized={}; entities={}; families={}
    for s in scenarios:
        if s["scenario_id"] in ids:
            raise ValueError("duplicate scenario")
        ids.add(s["scenario_id"])
        row=frozen.get(s["scenario_id"])
        if row is None or any(s[k]!=row[k] for k in ("split","domain","domain_family","entity_group","category","template_family")):
            raise ValueError("frozen assignment changed")
        if s["template_family"] not in TEMPLATES[s["split"]]:
            raise ValueError("heldout template leaked")
        for table,key in ((entities,s["entity_group"]),(families,s["template_family"])):
            if key in table and table[key]!=s["split"]:
                raise ValueError("entity or template crosses splits")
            table[key]=s["split"]
        validate_scenario(s,schema)
        for example in sft_examples(s,schema):
            text=base.dumps(example["messages"][:2])
            for table,key in ((exact,base.digest(text)),(normalized,base.digest(base.normalized_input(text)))):
                if key in table:
                    raise ValueError("duplicate exact/normalized model input")
                table[key]=s["scenario_id"]
    return {"exact_model_input_duplicates":0,"normalized_model_input_duplicates":0,
            "entity_groups_crossing_splits":0,"template_families_crossing_splits":0,"checked_examples":len(exact)}


def write_jsonl_gz(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw_digest=hashlib.sha256(); count=0; raw_bytes=0
    with path.open("xb") as raw, gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as stream:
        for row in rows:
            line=(base.dumps(row)+"\n").encode("utf8")
            stream.write(line); raw_digest.update(line); raw_bytes+=len(line); count+=1
    return {**base.file_record(path),"rows":count,"uncompressed_bytes":raw_bytes,"uncompressed_sha256":raw_digest.hexdigest()}


def read_jsonl_gz(path):
    with gzip.open(path,"rt",encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def write_artifacts(output,sealed_output,*,per_domain=100,version=VERSION,check_dependency=True):
    output,sealed_output=output.resolve(),sealed_output.resolve()
    if output.exists() or sealed_output.exists():
        raise FileExistsError("immutable revision already exists; choose a new revision")
    if output==sealed_output or output.is_relative_to(sealed_output) or sealed_output.is_relative_to(output):
        raise ValueError("public and sealed directories must be separate")
    if check_dependency:
        permitted=(PROJECT_ROOT/"training-runs/v6-sealed").resolve()
        if not sealed_output.is_relative_to(permitted):
            raise ValueError("sealed output must be under training-runs/v6-sealed")
        if subprocess.run(["git","-C",str(PROJECT_ROOT),"check-ignore","--quiet",str(sealed_output/"splits/final.jsonl.gz")]).returncode:
            raise ValueError("final output is not ignored")
    dependency=base.dependency_record() if check_dependency else {"commit":base.FPY_COMMIT,"miniature_fixture":True}
    schema=base.load_schema(); plan=freeze_plan(per_domain,version)
    output.mkdir(parents=True); sealed_output.mkdir(parents=True)
    base.write_json(output/"split-freeze.json",plan)
    base.write_json(output/"build-status.json",{"status":"building","version":version})
    try:
        cases=[build_scenario(row,version,schema) for row in plan["assignments"]]
        dedup=validate_collection(cases,plan,schema)
        stats=base.statistics_report(cases,schema)
        stats.update(domain_family_scenarios=dict(Counter(s["domain_family"] for s in cases)),
                     source_category_counts=dict(Counter(s["category"] for s in cases)))
        overlap,pairs=base.overlap_report(cases)
        decisions=base.review_decisions(cases)
        # All development scenarios enter the domain review queue. No automated
        # status can be mistaken for user review or resolve a fuzzy overlap flag.
        for decision in decisions:
            decision["human_review_required"]=True
            decision["human_review_reasons"].append("domain_semantic_review_before_training")
        files,sealed_files={},{}
        for split in ("train","validation","final"):
            root=sealed_output if split=="final" else output
            table=sealed_files if split=="final" else files
            subset=[s for s in cases if s["split"]==split]
            table[f"splits/{split}.jsonl.gz"]=write_jsonl_gz(root/"splits"/f"{split}.jsonl.gz",subset)
            table[f"sft/{split}.jsonl.gz"]=write_jsonl_gz(root/"sft"/f"{split}.jsonl.gz",(e for s in subset for e in sft_examples(s,schema)))
        validation=[s for s in cases if s["split"]=="validation"]
        smoke=[]
        for domain in PROFILE:
            options=[s for s in validation if s["domain"]==domain]
            smoke.extend(sorted(options,key=lambda s:base.digest(["v6-smoke",s["scenario_id"]]))[:1])
        files["splits/smoke.jsonl.gz"]=write_jsonl_gz(output/"splits/smoke.jsonl.gz",smoke)
        base.write_json(output/"schema.json",schema.model_dump(mode="json"))
        base.write_json(output/"dependency.json",dependency)
        base.write_json(output/"statistics.json",stats)
        base.write_json(output/"taxonomy.json",{"profiles":[{"subdomain":d,"family":f,"role":r,"calendar":c,"base_frequency_unit":u,"targets":[{"description":t,"unit":v,"column":k} for t,v,k in ts]} for d,f,r,c,u,ts in PROFILES]})
        base.write_json(output/"overlap-summary.json",overlap)
        files["overlap-review-queue.jsonl.gz"]=write_jsonl_gz(output/"overlap-review-queue.jsonl.gz",pairs)
        # Public review ledger includes IDs/reasons only for final cases, not labels.
        files["review-queue.jsonl.gz"]=write_jsonl_gz(output/"review-queue.jsonl.gz",decisions)
        files["review-sample.jsonl.gz"]=write_jsonl_gz(output/"review-sample.jsonl.gz",
            [min((s for s in cases if s["domain"]==d and s["split"]=="train"),key=lambda s:base.digest(["v6-review",s["scenario_id"]])) for d in PROFILE])
        base.write_json(output/"build-status.json",{"status":"candidate_pending_human_review","version":version})
        for p in output.rglob("*"):
            if p.is_file() and p.relative_to(output).as_posix() not in files:
                files[p.relative_to(output).as_posix()]=base.file_record(p)
        manifest={"corpus_version":version,"schema_version":schema.version,"slot_count":len(schema.slots),
            "prompt_version":base.PROMPT_VERSION,"fpy_commit":base.FPY_COMMIT,"build_status":"candidate_pending_human_review",
            "statistics":stats,"dedup":dedup,"fuzzy_overlap":overlap,"files":files,"sealed_files":sealed_files,
            "builder_files":{name:source_hash(PROJECT_ROOT/name) for name in ("local_slm_lab/corpus_v6.py","local_slm_lab/corpus_v5.py","local_slm_lab/v5_prompts.py","local_slm_lab/slm_prompts.py")},
            "review":{"accepted_human":0,"required_scenarios":len(cases),"pending_overlap_pairs":len(pairs)},
            "seal":"procedural only; deterministic builder is public; requires independent final custodian",
            "distribution":"gzip JSONL with deterministic mtime=0; raw hashes recorded; use prepare-v6-training.py",
            "no_model_calls":True,"no_live_observations":True,"trainable":False}
        base.write_json(output/"manifest.json",manifest)
        base.write_json(output/"manifest.sha256.json",{"manifest.json":base.file_record(output/"manifest.json")})
        return manifest
    except Exception:
        base.write_json(output/"build-status.json",{"status":"failed_do_not_reuse_version","version":version})
        raise


def verify_artifacts(output,sealed_output=None):
    output=output.resolve()
    manifest=json.loads((output/"manifest.json").read_bytes())
    if base.file_record(output/"manifest.json")!=json.loads((output/"manifest.sha256.json").read_bytes())["manifest.json"]:
        raise ValueError("manifest hash mismatch")
    if manifest["build_status"]!="candidate_pending_human_review" or manifest["trainable"] is not False:
        raise ValueError("candidate status or review claim changed")
    dependency=json.loads((output/"dependency.json").read_bytes())
    if not dependency.get("miniature_fixture"):
        base.dependency_record()  # fail clearly if the application pin is wrong
    for name,expected in manifest["builder_files"].items():
        if source_hash(PROJECT_ROOT/name)!=expected:
            raise ValueError("builder/prompt source changed")
    expected=set(manifest["files"])|{"manifest.json","manifest.sha256.json"}
    if {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}!=expected:
        raise ValueError("public artifact inventory mismatch")
    if sealed_output and {p.relative_to(sealed_output.resolve()).as_posix() for p in sealed_output.resolve().rglob("*") if p.is_file()}!=set(manifest["sealed_files"]):
        raise ValueError("sealed artifact inventory mismatch")
    for root,table in [(output,manifest["files"])]+([(sealed_output.resolve(),manifest["sealed_files"])] if sealed_output else []):
        for name,record in table.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or base.file_record(path)!={k:record[k] for k in ("bytes","sha256")}:
                raise ValueError("artifact hash/size mismatch")
            # Only stream sealed bytes for integrity; never deserialize final labels.
            if "uncompressed_sha256" in record:
                h=hashlib.sha256(); size=0; lines=0
                with gzip.open(path,"rb") as stream:
                    for line in stream:
                        h.update(line); size+=len(line); lines+=1
                if (h.hexdigest(),size,lines)!=(record["uncompressed_sha256"],record["uncompressed_bytes"],record["rows"]):
                    raise ValueError("decompressed artifact integrity mismatch")
    cases=[]
    for split in ("train","validation"):
        cases.extend(read_jsonl_gz(output/"splits"/f"{split}.jsonl.gz"))
    plan=json.loads((output/"split-freeze.json").read_bytes())
    expected_ids={r["scenario_id"] for r in plan["assignments"] if r["split"]!="final"}
    if {s["scenario_id"] for s in cases}!=expected_ids:
        raise ValueError("development frozen IDs mismatch")
    dedup=validate_collection(cases,plan)
    for split in ("train","validation"):
        subset=[s for s in cases if s["split"]==split]
        if read_jsonl_gz(output/"sft"/f"{split}.jsonl.gz")!=[e for s in subset for e in sft_examples(s)]:
            raise ValueError("SFT does not match component labels")
    validation={s["scenario_id"]:s for s in cases if s["split"]=="validation"}
    smoke=read_jsonl_gz(output/"splits/smoke.jsonl.gz")
    if any(s!=validation.get(s["scenario_id"]) for s in smoke):
        raise ValueError("smoke differs from validation")
    return {"status":"verified_candidate_development","development_scenarios":len(cases),"dedup":dedup,
            "final_labels_parsed":False,"sealed_hashes_checked":bool(sealed_output),"human_review":"pending"}
