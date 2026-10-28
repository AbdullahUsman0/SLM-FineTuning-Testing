"""Offline scientific and cost-safety regression cases; no external requests."""
import json
import os
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from controls import BudgetStop, Ledger, append, dumps, freeze, read_journal, request_input_bound
import pipeline as p

RUN=Path(os.environ['CURRENT_PIPELINE_RUN'])
p.configure(RUN/'fpy',Path(__file__).resolve().parents[2])


def ledger(tmp_path):
    return Ledger(tmp_path/'budget.sqlite3')


def test_budget_exhaustion_and_no_reset_on_reopen(tmp_path):
    b=ledger(tmp_path);b.reserve('first','ctx',3_500_000)
    b.reconcile('first','ctx',{'input_tokens':3_500_000,'output_tokens':2048},'hash')
    b.db.close();again=ledger(tmp_path)
    assert again.summary()['actual_USD']>.87
    with pytest.raises(BudgetStop):again.reserve('second','ctx',100_000)


@pytest.mark.parametrize('uncertain',[False,True])
def test_restart_with_pending_or_uncertain_blocks_dispatch(tmp_path,uncertain):
    b=ledger(tmp_path);b.reserve('first','ctx',100)
    if uncertain:b.uncertain('first')
    b.db.close();again=ledger(tmp_path)
    with pytest.raises(BudgetStop):again.reserve('second','ctx',100)
    assert again.summary()['reserved_or_uncertain_USD']>0


def test_reasoning_is_billed_once_and_cache_discount_not_assumed(tmp_path):
    b=ledger(tmp_path);b.reserve('x','ctx',1000)
    b.reconcile('x','ctx',{'input_tokens':1000,'input_tokens_details':{'cached_tokens':900},'output_tokens':200,'output_tokens_details':{'reasoning_tokens':150}},'hash')
    assert b.summary()['actual_nanousd']==1000*250+200*2000


def test_usage_overrun_remains_halted_after_restart(tmp_path):
    b=ledger(tmp_path);b.reserve('x','ctx',1)
    with pytest.raises(BudgetStop):b.reconcile('x','ctx',{'input_tokens':100_000,'output_tokens':2048},'hash')
    b.db.close()
    with pytest.raises(BudgetStop):ledger(tmp_path).reserve('next','ctx',100)


def test_missing_usage_retains_reservation(tmp_path):
    b=ledger(tmp_path);b.reserve('x','ctx',100)
    with pytest.raises(BudgetStop):b.reconcile('x','ctx',{},'hash')
    with pytest.raises(BudgetStop):b.reserve('next','ctx',100)


def test_completed_response_reconcile_is_idempotent_and_key_not_replayed(tmp_path):
    b=ledger(tmp_path);b.reserve('x','ctx',100)
    usage={'input_tokens':10,'output_tokens':1}
    b.reconcile('x','ctx',usage,'hash');b.reconcile('x','ctx',usage,'hash')
    assert len(b.summary()['calls'])==1
    with pytest.raises(BudgetStop):b.reserve('x','ctx',100)
    with pytest.raises(BudgetStop):b.reconcile('x','different_context',usage,'hash')


def test_journal_preserves_truncated_tail_and_rejects_duplicate_keys(tmp_path):
    path=tmp_path/'calls.jsonl';append(path,{'key':['x'],'raw':'complete'})
    with path.open('ab') as stream:stream.write(b'{"key":')
    assert list(read_journal(path))==[('x',)]
    assert len(list(tmp_path.glob('*.incomplete-*')))==1
    append(path,{'key':['x'],'raw':'duplicate'})
    with pytest.raises(ValueError):read_journal(path)


def test_context_drift(tmp_path):
    path=tmp_path/'protocol.json';freeze(path,{'context':'a'})
    with pytest.raises(ValueError):freeze(path,{'context':'b'})


def fixture(value='forecast temperature'):
    return {'intent':'create_forecast','intent_confidence':1.,'updates':[{'slot_id':'problem_statement','candidate_value':json.dumps(value),'status':'provided','confidence':1.,'evidence_text':'forecast temperature'}],'correction_detected':False,'unsupported_claims':[]}


@pytest.mark.parametrize('flag',['capped','refused'])
def test_capped_refused_well_formed_response_is_not_a_true_positive(flag):
    gold=fixture();turn={'gold_extraction':gold,'forbidden_slots':[]}
    metrics=p.raw_score({'raw_output':dumps(gold),flag:True},turn,p.wire_schema())
    assert metrics['confusion']==[0,1,1] and not metrics['raw_contract_admitted']


@pytest.mark.parametrize('raw',['{"intent":"create_forecast","intent":"unsupported"}', '{bad json}'])
def test_malformed_duplicate_fields_are_not_admitted(raw):
    with pytest.raises(Exception):p.admit(raw,p.wire_schema())


def test_duplicate_slot_is_not_silently_collapsed():
    gold=fixture();gold['updates']*=2
    with pytest.raises(ValueError):p.admit(dumps(gold),p.wire_schema())


def test_literal_json_looking_string_decoded_exactly_once():
    obj=fixture('true');parsed=p.admit(dumps(obj),p.wire_schema())
    assert parsed.updates[0].candidate_value=='true'
    assert p.tuples(obj)[0][1]=='"true"'


def test_schema_valid_but_wrong_value_is_false_positive_and_false_negative():
    gold=fixture();wrong=fixture('invented unrelated target')
    metrics=p.raw_score({'raw_output':dumps(wrong)},{'gold_extraction':gold,'forbidden_slots':[]},p.wire_schema())
    assert metrics['schema_valid'] and metrics['confusion']==[0,1,1]


def test_request_bound_includes_schema_and_unicode_bytes():
    request={'input':'\u0627'*100,'text':{'format':{'schema':p.wire_schema()}}}
    assert request_input_bound(request)>=len(dumps(request).encode('utf-8'))+16384


def test_pipeline_recovery_is_separate_from_raw_credit():
    from forecasting_assistant.domain.schema import load_schema,create_initial_state
    from forecasting_assistant.domain.models import Intent,SlotStatus
    schema=load_schema();state=create_initial_state(schema)
    state.intent=Intent.CREATE_FORECAST
    state.slots['intent'].value='create_forecast';state.slots['intent'].status=SlotStatus.PROVIDED
    empty={'intent':'create_forecast','intent_confidence':1.,'updates':[],'correction_detected':False,'unsupported_claims':[]}
    record={'raw_output':dumps(empty)}
    after,trace=p.run_pipeline(state,'upload',schema,record,p.wire_schema())
    assert trace['recovered_slot_ids']==['source_mode']
    assert after.slots['source_mode'].value=='upload'
    assert not p.tuples(empty)


def test_selection_is_reproducible_balanced_and_excludes_pilots():
    from prepare import select
    cases=[]
    for domain in range(20):
        for language in ('en','ur','mixed'):
            cases.append({'domain':str(domain),'language':language,'scenario_id':str(domain)+'/'+language})
    main,pilots=select(cases);assert (main,pilots)==select(list(reversed(cases)))
    from collections import Counter
    assert Counter(c['language'] for c in main)=={'en':16,'ur':2,'mixed':2}
    assert not {c['scenario_id'] for c in main}&{c['scenario_id'] for c in pilots}


def test_state_datetime_serializes_as_iso_without_false_contract_conflict():
    from forecasting_assistant.domain.schema import load_schema,create_initial_state
    from datetime import datetime,timezone
    state=create_initial_state(load_schema())
    state.slots['forecast_start'].value=datetime(2026,10,1,tzinfo=timezone.utc)
    assert p.snapshot(state)['forecast_start']['value']=='2026-10-01T00:00:00+00:00'


def test_user_adjudication_only_changes_approved_state_status():
    from prepare import adjudicate
    cases=[{'scenario_id':'v7-20261009-r8-dev/labor_employment/0178','turns':[{'turn_id':'t2','gold_extraction':fixture('true'),'context_state':{},'expected_state_after':{'target_column':{'value':'true','status':'provided','confirmed_by_user':False}}},{'turn_id':'t3','gold_extraction':fixture(),'context_state':{'target_column':{'value':'true','status':'provided','confirmed_by_user':False}},'expected_state_after':{'target_column':{'value':'true','status':'provided','confirmed_by_user':False}}}],'gold_final_statuses':{'target_column':'provided'}}]
    revised,changes=adjudicate(cases)
    assert len(changes)==3
    assert revised[0]['turns'][0]['expected_state_after']['target_column']['status']=='invalid'
    assert revised[0]['turns'][1]['context_state']['target_column']['status']=='invalid'
    assert revised[0]['turns'][0]['gold_extraction']==cases[0]['turns'][0]['gold_extraction']
    assert cases[0]['turns'][0]['expected_state_after']['target_column']['status']=='provided'


def test_hosted_saved_response_recovery_never_dispatches_a_second_request(tmp_path):
    from runner import Hosted
    from controls import digest,atomic,MODEL
    from types import SimpleNamespace
    import unittest.mock as mock
    h=Hosted.__new__(Hosted);h.run=tmp_path;h.protocol={'instructions':'Extract only current evidence','response_schema':p.wire_schema()};h.ledger=ledger(tmp_path)
    response={'model':MODEL,'id':'offline-response','status':'completed','output':[],'usage':{'input_tokens':100,'output_tokens':20,'output_tokens_details':{'reasoning_tokens':10}}}
    item=SimpleNamespace(output_text=dumps(fixture()),model_dump=lambda **kwargs:response)
    create=mock.Mock(return_value=item);h.client=SimpleNamespace(responses=SimpleNamespace(create=create))
    key=['pilot','openai','component','offline-fixture','t1']
    first=h.generate(key,'same context');second=h.generate(key,'same context')
    assert first==second and create.call_count==1
    assert len(h.ledger.summary()['calls'])==1
    with pytest.raises(ValueError):h.generate(key,'drifted context')


def test_hosted_timeout_keeps_reservation_and_resume_does_not_retry(tmp_path):
    from runner import Hosted
    from types import SimpleNamespace
    import unittest.mock as mock
    h=Hosted.__new__(Hosted);h.run=tmp_path;h.protocol={'instructions':'Extract','response_schema':p.wire_schema()};h.ledger=ledger(tmp_path)
    create=mock.Mock(side_effect=TimeoutError('sensitive transport detail must not be logged'));h.client=SimpleNamespace(responses=SimpleNamespace(create=create))
    key=['pilot','openai','component','offline-fixture','t1']
    with pytest.raises(BudgetStop):h.generate(key,'context')
    with pytest.raises(BudgetStop):h.generate(key,'context')
    assert create.call_count==1
    assert h.ledger.summary()['reserved_or_uncertain_USD']>0
    assert 'sensitive transport detail' not in (tmp_path/'infrastructure-errors.jsonl').read_text()
