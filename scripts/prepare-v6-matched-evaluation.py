"""Freeze four-domain development inputs and authored conversational diagnostics.

Only opens the committed v6 validation split, never training/final labels.
Natural diagnostics are agent-authored, not independent human-reviewed test data.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_slm_lab.v5_eval import build_call_plan, digest, load_schema, now, sha256

# Hand-authored utterances and explicit annotations are frozen before inference.
# Duration notation: (periods, canonical singular unit).
NATURAL = {
    'weather': [
        ('Could you forecast afternoon air temperature? My readings are daily, in degrees Celsius, and I need the next 5 days. The file is CSV.', 'afternoon air temperature', 'degrees Celsius', (1,'day'), (5,'day'), {'file_format':'csv'}),
        ('Rain is what I care about: please predict accumulated rainfall, measured in millimetres. I have hourly readings and want 36 hours ahead.', 'accumulated rainfall', 'millimetres', (1,'hour'), (36,'hour'), {}),
        ('I need a forecast of peak wind speed in metres per second, every hour, out to 12 hours. I will upload the data as parquet.', 'peak wind speed', 'metres per second', (1,'hour'), (12,'hour'), {'source_mode':'upload','file_format':'parquet'}),
        ('Help me predict relative humidity. It is in percent, observed daily; look ahead 9 days. Read humidity_pct as the target and observation_date as the timestamp.', 'relative humidity', 'percent', (1,'day'), (9,'day'), {'target_column':'humidity_pct','time_column':'observation_date'}),
        ('What I want forecast is cloud cover, in percent. We record it every 3 hours. Please look 18 hours into the future.', 'cloud cover', 'percent', (3,'hour'), (18,'hour'), {}),
        ('Can we forecast sea-level pressure? Values are in hectopascals and arrive every 6 hours. I need a 2 day horizon.', 'sea-level pressure', 'hectopascals', (6,'hour'), (2,'day'), {}),
        ('Please predict overnight minimum temperature in degrees Fahrenheit. Observations are daily. Forecast 8 days ahead, and generate a fresh forecast every 2 days.', 'overnight minimum temperature', 'degrees Fahrenheit', (1,'day'), (8,'day'), {'prediction_frequency':{'periods':2,'unit':'day'}}),
        ('Build a forecast for sunshine duration, measured in hours, with one observation every week. I need the next 4 weeks. The uploaded file is xlsx.', 'sunshine duration', 'hours', (1,'week'), (4,'week'), {'source_mode':'upload','file_format':'xlsx'}),
    ],
    'inflation': [
        ('Please forecast year-over-year CPI inflation, in percent, from monthly observations. I need the coming 5 months, not a forecast of the CPI index.', 'year-over-year CPI inflation', 'percent', (1,'month'), (5,'month'), {}),
        ('Can you predict the headline CPI index level? Units are index points, base 2015=100. It is monthly and I want 8 months ahead.', 'headline CPI index level', 'index points, base 2015=100', (1,'month'), (8,'month'), {}),
        ('I need a forecast of month-over-month core inflation. The unit is percent. Observations arrive monthly; forecast the next 3 months. The file is csv.', 'month-over-month core inflation', 'percent', (1,'month'), (3,'month'), {'file_format':'csv'}),
        ('Help me predict food-price inflation in percent. There is one observation each month. Look 11 months ahead. The target column is food_inflation and the date column is month_end.', 'food-price inflation', 'percent', (1,'month'), (11,'month'), {'target_column':'food_inflation','time_column':'month_end'}),
        ('Please forecast producer price inflation, measured in percent, using quarterly observations. I want the next 2 quarters.', 'producer price inflation', 'percent', (1,'quarter'), (2,'quarter'), {}),
        ('Forecast the consumer price index level in index points. The series has a monthly observation and I need a 1 year horizon. I will upload a parquet file.', 'consumer price index level', 'index points', (1,'month'), (1,'year'), {'source_mode':'upload','file_format':'parquet'}),
        ('I want to forecast annual average inflation in percent. One observation per year; look ahead 3 years. Generate a new forecast every 1 year.', 'annual average inflation', 'percent', (1,'year'), (3,'year'), {'prediction_frequency':{'periods':1,'unit':'year'}}),
        ('Could you predict trimmed-mean inflation? It is monthly, expressed in percent. We need 7 months into the future; the uploaded data will be in xlsx format.', 'trimmed-mean inflation', 'percent', (1,'month'), (7,'month'), {'source_mode':'upload','file_format':'xlsx'}),
    ],
    'stocks': [
        ("Can you forecast NOVA's adjusted closing price, in USD per share? I have daily observations and need the next 10 days.", "NOVA's adjusted closing price", 'USD per share', (1,'day'), (10,'day'), {}),
        ('Please predict raw closing share price, without adjusting corporate actions. Prices are in GBP per share, observed every day. Look ahead 6 days.', 'raw closing share price, without adjusting corporate actions', 'GBP per share', (1,'day'), (6,'day'), {}),
        ('Help me forecast total share return including dividends, measured in percent. Observations are weekly; I need 5 weeks ahead. I will upload CSV data.', 'total share return including dividends', 'percent', (1,'week'), (5,'week'), {'source_mode':'upload','file_format':'csv'}),
        ('Forecast trading volume, measured in shares, from daily observations for the next 9 days. Read the target from volume_shares and the timestamp from trade_date.', 'trading volume', 'shares', (1,'day'), (9,'day'), {'target_column':'volume_shares','time_column':'trade_date'}),
        ('Please predict the dividend yield in percent. This series is monthly, and the horizon should be 4 months.', 'dividend yield', 'percent', (1,'month'), (4,'month'), {}),
        ('I need a forecast of adjusted closing share price. It is in EUR per share, sampled hourly, and I want 16 hours ahead. The file format is parquet.', 'adjusted closing share price', 'EUR per share', (1,'hour'), (16,'hour'), {'file_format':'parquet'}),
        ('Could you forecast daily log share return in percent? There is one observation per day. Predict 12 days ahead but refresh the forecast every 3 days.', 'daily log share return', 'percent', (1,'day'), (12,'day'), {'prediction_frequency':{'periods':3,'unit':'day'}}),
        ('I want to forecast ETF trading volume in shares. I have weekly data and need the next 6 weeks. I will upload the spreadsheet in xlsx format.', 'ETF trading volume', 'shares', (1,'week'), (6,'week'), {'source_mode':'upload','file_format':'xlsx'}),
    ],
    'crypto': [
        ('Can you forecast BTC/USD spot price in USD per BTC? My observations are hourly and I want 20 hours ahead, not a futures price.', 'BTC/USD spot price', 'USD per BTC', (1,'hour'), (20,'hour'), {}),
        ('Please predict ETH trading volume, in ETH, using observations every 2 hours. I need the next 14 hours.', 'ETH trading volume', 'ETH', (2,'hour'), (14,'hour'), {}),
        ('Help me forecast perpetual-swap funding rate. Units are percent per 8 hours and observations come every 8 hours. Look ahead 48 hours. I will upload CSV data.', 'perpetual-swap funding rate', 'percent per 8 hours', (8,'hour'), (48,'hour'), {'source_mode':'upload','file_format':'csv'}),
        ('I need a forecast of futures open interest in contracts. This is hourly data and the horizon is 30 hours. Read open_contracts as the target and event_time as the timestamp.', 'futures open interest', 'contracts', (1,'hour'), (30,'hour'), {'target_column':'open_contracts','time_column':'event_time'}),
        ('Please forecast realized volatility, in percent per year, from daily observations for the next 7 days.', 'realized volatility', 'percent per year', (1,'day'), (7,'day'), {}),
        ('Predict futures-minus-spot basis, measured in basis points. Each observation is 1 hour apart; I need a 2 day horizon. The file will be parquet.', 'futures-minus-spot basis', 'basis points', (1,'hour'), (2,'day'), {'file_format':'parquet'}),
        ('Can you forecast BTC log return in percent? I have one observation every hour. Predict 18 hours ahead, but make a fresh forecast every 6 hours.', 'BTC log return', 'percent', (1,'hour'), (18,'hour'), {'prediction_frequency':{'periods':6,'unit':'hour'}}),
        ('I want a forecast of digital-asset market capitalization in millions of USD. We observe it daily and need 15 days ahead. I will upload JSON data.', 'digital-asset market capitalization', 'millions of USD', (1,'day'), (15,'day'), {'source_mode':'upload','file_format':'json'}),
    ],
}
CORRECTIONS = [
    'Actually, change the forecast horizon to {n} {unit}s instead. Leave the other requirements alone.',
    'Correction: I need {n} {unit}s ahead, replacing the earlier horizon.',
    'I changed my mind about the horizon. Forecast the next {n} {unit}s, please.',
    'That previous forecast horizon is wrong: make it {n} {unit}s.',
    'Please revise the forecast horizon to {n} {unit}s. Everything else stays the same.',
    'Use {n} {unit}s for the forecast horizon instead of my earlier request.',
    'One correction to my specification: the forecast horizon is now {n} {unit}s.',
    'Replace the horizon I gave you with {n} {unit}s.',
]
UNKNOWN = [
    "I do not know the timezone, seasonal periods or prediction interval levels yet. Ask me; do not fill them in.",
    "Timezone is undecided, and I haven't chosen seasonal periods or prediction interval levels. I don't have credentials to share either.",
    "I cannot tell you the timezone yet. Seasonal periods and prediction interval levels are unknown too.",
    "I haven't worked out the timezone or seasonal periods. We have no decision on prediction interval levels yet.",
    "Please leave timezone, seasonal periods and prediction interval levels unset. I do not know their values.",
    "I need your questions about timezone, seasonal periods and prediction interval levels because I don't know them.",
    "No guesswork: I haven't decided the timezone, seasonal periods or prediction interval levels.",
    "Those details are still unknown: timezone, seasonal periods and prediction interval levels. I cannot give you an authentication reference.",
]

def update(slot, value, message):
    return dict(slot_id=slot, candidate_value=json.dumps(value,ensure_ascii=False), status='provided',
                confidence=1.0, evidence_text=message)

def turn(message, facts, correction=False, category='initial'):
    return {'message':message, 'test_kind':category,
            'gold_extraction': {'intent':'create_forecast','intent_confidence':1.0,
                'updates':[update(k,v,message) for k,v in facts.items()],
                'correction_detected':correction,'unsupported_claims':[]}}

def prepare(output):
    if output.exists():
        raise FileExistsError('Frozen study already exists; do not overwrite it')
    source = ROOT / 'corpus-v6/v6-20261002-r4/splits/validation.jsonl.gz'
    checksum_manifest = json.loads((source.parent.parent/'manifest.sha256.json').read_text())
    manifest_path = source.parent.parent/'manifest.json'
    if sha256(manifest_path) != checksum_manifest['manifest.json']['sha256']:
        raise ValueError('Corpus manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    # Independent artifact hash check; support the corpus's recorded file map.
    entries = manifest.get('files', manifest)
    item = entries['splits/validation.jsonl.gz']
    expected = item['sha256'] if isinstance(item,dict) else item
    if sha256(source) != expected:
        raise ValueError('Frozen validation artifact hash mismatch')
    rows = [json.loads(line) for line in gzip.open(source,'rt',encoding='utf-8') if line.strip()]
    weather = {'temperature','precipitation','wind','humidity_pressure','solar_cloud','weather_extremes'}
    domain_map = {d:'weather' for d in weather}
    domain_map.update(inflation_prices='inflation', equities_funds='stocks', crypto_spot='crypto', crypto_derivatives='crypto')
    selected = [dict(row,evaluation_domain=domain_map[row['domain']]) for row in rows if row['domain'] in domain_map]
    assert all(row['split']=='validation' for row in selected) and len(selected)==150
    natural = []
    for domain, examples in NATURAL.items():
        for i,(message,target,unit,frequency,horizon,extras) in enumerate(examples):
            assert target in message and unit in message
            facts = {'target_description':target,'target_unit':unit,
                     'frequency':dict(periods=frequency[0],unit=frequency[1]),
                     'forecast_horizon':dict(periods=horizon[0],unit=horizon[1]), **extras}
            revised = horizon[0]+3
            correction = CORRECTIONS[i].format(n=revised,unit=horizon[1])
            turns = [turn(message,facts),turn(correction,{'forecast_horizon':dict(periods=revised,unit=horizon[1])},True,'correction'),
                     turn(UNKNOWN[i],{},False,'unknown')]
            turns[-1]['must_not_infer'] = ['timezone','seasonal_periods','prediction_interval_levels','authentication_reference']
            sid=f'natural-v6-20261005/{domain}/{i+1:02d}'
            natural.append(dict(scenario_id=sid,cluster_id=sid,category='natural_conversation',
                                evaluation_domain=domain,domain=domain,split='validation',turns=turns,
                                origin='agent_authored_before_predictions',reviewer_status='independent_human_review_pending'))
    schema = load_schema()
    plans = {name:build_call_plan(cases,schema) for name,cases in [('frozen',selected),('natural',natural)]}
    output.mkdir(parents=True)
    for name,cases in [('frozen',selected),('natural',natural)]:
        with (output/(name+'.jsonl')).open('x',encoding='utf-8',newline='\n') as stream:
            for case in cases:
                stream.write(json.dumps(case,ensure_ascii=False,sort_keys=True)+'\n')
    metadata = {'version':'v6-matched-20261005-1','frozen_before_inference_utc':now(),
        'base_model':'Qwen/Qwen3.5-2B','base_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc',
        'source_validation_sha256':sha256(source), 'fpy_revision':'04d52c015d1e3ecdefe92b87116f209361509b4b',
        'source_dataset':'v6-20261002-r4','sealed_final_accessed':False,
        'independent_human_review':'pending; authored natural diagnostics are exploratory',
        'primary_metrics':'exact slot/value/status micro F1 excluding intent; intent separate; raw JSON/schema validity',
        'invented_requirements':'new unmentioned slot updates separate from repeated prior slots and wrong stated values',
        'natural_contexts':['gold_context_component','predicted_state_rollout_fixed_user_script'],
        'decoding':{'do_sample':False,'max_new_tokens':1024,'enable_thinking':False,'dtype':'bfloat16','device':'cuda','seed':42},
        'latency':'serial batch size 1 after unscored warmup; load time separate; no retries or repair; CUDA synchronized',
        'cases':{name:{'sha256':sha256(output/(name+'.jsonl')),'scenarios':len(cases),
                       'calls':len(plans[name]),'domains':dict(Counter(r['evaluation_domain'] for r in cases)),
                       'ordered_keys_sha256':digest([c['key'] for c in plans[name]]),
                       'context_gold_sha256':digest([[c['key'],digest(c['context']),digest(c['expected'])] for c in plans[name]])}
                 for name,cases in [('frozen',selected),('natural',natural)]}}
    (output/'study-manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    prepare(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'evaluation/v6-matched-20261005')
