"""Native-parser, actionable-scoring, replay and amended-budget controls."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from controls import AMENDMENT, EFFECTIVE_CEILING, BudgetStop, Ledger, append, dumps, freeze, install_approved_amendment, read_journal
import pipeline as p
RUN=Path(os.environ['CURRENT_PIPELINE_RUN'])
p.configure(RUN/'fpy',Path(__file__).resolve().parents[2])
from forecasting_assistant.domain.models import ExtractorResult


def obj(value='"temperature"',status='provided'):
    return {'intent':'create_forecast','intent_confidence':1.,'updates':[{'slot_id':'target_description','candidate_value':value,'status':status,'confidence':1.,'evidence_text':'temperature'}],'correction_detected':False,'unsupported_claims':[]}


@pytest.mark.parametrize('value',['plain_column_name','"true"','true','false','123','null','[]','{"periods":1,"unit":"day"}','"\\\"true\\\""'])
def test_parser_and_tuple_values_match_native_once(value):
    raw=dumps(obj(value));native=ExtractorResult.model_validate_json(raw)
    assert p.admit(raw,p.wire_schema())==native
    assert json.loads(p.tuples(obj(value))[0][1])==native.updates[0].candidate_value


@pytest.mark.parametrize('status',['unmentioned','dont_care'])
def test_nonactionable_updates_are_reported_not_false_requirements(status):
    gold=obj();gold['updates']=[]
    row=p.raw_score({'raw_output':dumps(obj('unknown',status))},{'gold_extraction':gold,'forbidden_slots':[]},p.wire_schema())
    assert row['confusion']==[0,0,0] and row['ignored_status_updates']==1
    assert row['unknown_no_updates'] and not row['wire_empty_nonintent_updates']


def test_plain_wire_string_and_json_string_match_without_second_decode():
    row=p.raw_score({'raw_output':dumps(obj('temperature'))},{'gold_extraction':obj(),'forbidden_slots':[]},p.wire_schema())
    assert row['confusion']==[1,0,0]
    row=p.raw_score({'raw_output':dumps(obj('true'))},{'gold_extraction':obj('"true"'),'forbidden_slots':[]},p.wire_schema())
    assert row['confusion']==[0,1,1]


def test_native_duplicates_are_admitted_but_extra_prediction_is_penalized():
    predicted=obj();predicted['updates']*=2
    row=p.raw_score({'raw_output':dumps(predicted)},{'gold_extraction':obj(),'forbidden_slots':[]},p.wire_schema())
    assert row['raw_contract_admitted'] and row['duplicate_slots']
    assert row['confusion']==[1,1,0]


@pytest.mark.parametrize('flag',['capped','refused'])
def test_capped_or_refused_not_credited(flag):
    row=p.raw_score({'raw_output':dumps(obj()),flag:True},{'gold_extraction':obj(),'forbidden_slots':[]},p.wire_schema())
    assert row['confusion']==[0,1,1] and not row['raw_contract_admitted']


def test_wrong_schema_valid_value_not_credited():
    row=p.raw_score({'raw_output':dumps(obj('"wrong"'))},{'gold_extraction':obj(),'forbidden_slots':[]},p.wire_schema())
    assert row['schema_valid'] and row['confusion']==[0,1,1]


@pytest.mark.parametrize('raw',['{','[]','null','"text"'])
def test_malformed_or_nonobject_response_is_a_miss(raw):
    row=p.raw_score({'raw_output':raw},{'gold_extraction':obj(),'forbidden_slots':[]},p.wire_schema())
    assert not row['raw_contract_admitted'] and row['confusion']==[0,0,1]


def seeded(path):
    db=sqlite3.connect(path)
    db.executescript('CREATE TABLE authorization (id TEXT PRIMARY KEY, config TEXT NOT NULL); CREATE TABLE calls (key TEXT PRIMARY KEY,context TEXT NOT NULL,bound INTEGER NOT NULL,actual INTEGER,state TEXT NOT NULL,response_sha TEXT,usage TEXT);')
    original={'scope':'one bounded current-pipeline v8/OpenAI evaluation, including pilots and retries','user_ceiling_nanousd':1_000_000_000,'dispatch_ceiling_nanousd':900_000_000,'model':'gpt-5-mini-2025-08-07','input_rate_nanousd_per_token':250,'output_rate_nanousd_per_token':2000}
    db.execute('INSERT INTO authorization VALUES (?,?)',('20261023',dumps(original)))
    db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?)',('prior','ctx',AMENDMENT['prior_actual_nanousd'],AMENDMENT['prior_actual_nanousd'],'reconciled','sha',dumps({'input_tokens':2588093,'output_tokens':0})))
    db.commit();db.close();return path


def ledger(tmp_path):
    path=seeded(tmp_path/'budget.sqlite3');install_approved_amendment(path);return Ledger(path)


def test_amendment_required_and_prior_spending_never_reset(tmp_path):
    path=seeded(tmp_path/'budget.sqlite3')
    with pytest.raises(BudgetStop):Ledger(path)
    install_approved_amendment(path);b=Ledger(path)
    assert b.summary()['actual_nanousd']==647023250 and b.summary()['authorization_total_USD']==2
    old=b.db.execute('SELECT * FROM calls').fetchall();install_approved_amendment(path)
    assert b.db.execute('SELECT * FROM calls').fetchall()==old
    assert json.loads(b.db.execute('SELECT config FROM authorization').fetchone()[0])['user_ceiling_nanousd']==1_000_000_000


def test_rerun_cap_survives_restart(tmp_path):
    b=ledger(tmp_path);b.reserve('a','ctx',3_500_000)
    b.reconcile('a','ctx',{'input_tokens':3_500_000,'output_tokens':2048},'sha')
    b.db.close();b=Ledger(tmp_path/'budget.sqlite3')
    with pytest.raises(BudgetStop):b.reserve('b','ctx',100_000)


@pytest.mark.parametrize('uncertain',[False,True])
def test_pending_and_uncertain_stop_restart(tmp_path,uncertain):
    b=ledger(tmp_path);b.reserve('a','ctx',100)
    if uncertain:b.uncertain('a')
    b.db.close();b=Ledger(tmp_path/'budget.sqlite3')
    with pytest.raises(BudgetStop):b.reserve('b','ctx',100)


def test_usage_including_reasoning_charged_once_and_no_cache_discount(tmp_path):
    b=ledger(tmp_path);b.reserve('a','ctx',1000)
    b.reconcile('a','ctx',{'input_tokens':1000,'input_tokens_details':{'cached_tokens':900},'output_tokens':200,'output_tokens_details':{'reasoning_tokens':150}},'sha')
    assert b.summary()['actual_nanousd']==647023250+250000+400000
    b.reconcile('a','ctx',{'input_tokens':1000,'output_tokens':200},'sha')
    with pytest.raises(BudgetStop):b.reserve('a','ctx',100)


def test_missing_usage_and_overrun_preserved(tmp_path):
    b=ledger(tmp_path);b.reserve('a','ctx',1)
    with pytest.raises(BudgetStop):b.reconcile('a','ctx',{},'sha')
    with pytest.raises(BudgetStop):b.reconcile('a','ctx',{'input_tokens':100000,'output_tokens':2048},'sha')
    b.db.close()
    with pytest.raises(BudgetStop):Ledger(tmp_path/'budget.sqlite3').reserve('b','ctx',1)


def test_context_and_journal_controls(tmp_path):
    path=tmp_path/'calls.jsonl';append(path,{'key':['a'],'raw':'done'})
    with path.open('ab') as stream:stream.write(b'{')
    assert ('a',) in read_journal(path) and list(tmp_path.glob('*.incomplete-*'))
    append(path,{'key':['a'],'raw':'done'})
    with pytest.raises(ValueError):read_journal(path)
    freeze(tmp_path/'protocol.json',{'a':1})
    with pytest.raises(ValueError):freeze(tmp_path/'protocol.json',{'a':2})


def test_all_reused_slm_inputs_and_trajectories_were_proven():
    proof=json.loads((RUN/'slm-reuse-verification.json').read_bytes())
    assert proof['passed'] and proof['calls']==220 and proof['main_calls']==202
    assert proof['all_inputs_and_pipeline_trajectories_exactly_unchanged']
