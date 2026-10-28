"""Current application adapter; extraction is always scored BEFORE recovery."""
from __future__ import annotations
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import date, datetime
import json
from pathlib import Path
import sys
from controls import dumps, strict_json


def configure(fpy, repo):
    src = Path(fpy).resolve() / 'src'
    sys.path[:0] = [str(src), str(repo)]
    import forecasting_assistant.domain.models as models
    if not Path(models.__file__).resolve().is_relative_to(src):
        raise RuntimeError('Wrong installed fpy imported')


def wire_schema():
    from forecasting_assistant.domain.models import ExtractorResult
    from openai.lib._pydantic import to_strict_json_schema
    return to_strict_json_schema(ExtractorResult)


def build_instructions(repo):
    from forecasting_assistant.prompts.extractor import build_extractor_instructions
    training=(Path(repo)/'corpus-v8/v8-final-20261018-r1/extractor-instructions.txt').read_text(encoding='utf-8')
    return build_extractor_instructions()+'\n\nExplicit common extraction interface and unchanged annotation policy:\n'+training+'\nThe response_schema describes the OUTPUT only. Priority rows with aspect/dimension/question are INPUT context, not an output format. Do not copy those rows as updates. Return every update with exactly slot_id, candidate_value, status, confidence, evidence_text. Both confidence fields must be between 0 and 1.\n'


def snapshot(state):
    result = {k: {'value': v.value, 'status': v.status.value, 'confirmed_by_user': v.confirmed_by_user}
              for k, v in state.slots.items() if v.value is not None}
    # This is serialization of native date objects, not semantic value repair.
    return json.loads(json.dumps(result,default=lambda v:v.isoformat() if isinstance(v,(date,datetime)) else str(v),allow_nan=False))


def restore(turn, schema):
    from forecasting_assistant.domain.models import DialogueTurn, Intent, SlotState
    from forecasting_assistant.domain.schema import create_initial_state
    state = create_initial_state(schema)
    state.intent = Intent(turn['established_intent'])
    for sid, item in turn['context_state'].items():
        state.slots[sid] = SlotState(slot_id=sid, **item, evidence_text='Recorded synthetic prior', confidence=1.)
    state.turns = [DialogueTurn(turn_number=i, user_message=m) for i, m in enumerate(turn['prior_user_messages'], 1)]
    return state


def intervention(state, change):
    from forecasting_assistant.domain.models import SlotState
    for mutation in (change or {}).get('mutations', []):
        if mutation['kind'] == 'omitted_prior_value':
            state.slots[mutation['slot']] = SlotState(slot_id=mutation['slot'])
        elif mutation['kind'] == 'wrong_prior_value':
            state.slots[mutation['slot']].value = deepcopy(mutation['value'])
        else:
            raise ValueError('Unrecognized explicit synthetic intervention')


def extraction_context(state, message, schema):
    """The engine appends the current user turn BEFORE ranking pending slots."""
    from forecasting_assistant.domain.models import DialogueTurn
    from forecasting_assistant.prompts.extractor import build_extractor_input
    prior=state.model_copy(deep=True)
    for turn in prior.turns:
        turn.assistant_message=None
    prior.turns.append(DialogueTurn(turn_number=len(prior.turns)+1,user_message=message))
    return enrich_context(build_extractor_input(message,prior,schema),schema)


def enrich_context(native_context,schema):
    """Explicit common interface adapter, versioned after the failed native run.

    Retains native current context; exposes the SAME full contract to BOTH models.
    No labels, desired answers, reducer recovery or output repairs are introduced.
    """
    payload=strict_json(native_context)
    payload['slot_definitions']=[{'slot_id':s.slot_id,'description':s.description,'value_type':s.value_type,
                                 **({'allowed_values':list(s.allowed_values)} if s.allowed_values else {})} for s in schema.slots]
    payload['response_schema']=wire_schema()
    payload['established_intent']=payload['intent_policy']['value']
    payload['selected_slot']=(payload.get('pending_slot') or {}).get('slot_id')
    payload['last_assistant_question']=payload.get('latest_assistant_question')
    return dumps(payload)


def tuples(obj):
    """Native one-decode, actionable tuples. No state recovery or synonym repair."""
    from forecasting_assistant.domain.models import _decode_candidate_value
    values = []
    if not isinstance(obj, dict) or not isinstance(obj.get('updates'), list):
        return values
    for update in obj['updates']:
        if not isinstance(update, dict) or not {'slot_id','candidate_value','status'} <= update.keys() or update['slot_id'] == 'intent' or update['status'] in {'unmentioned','dont_care'}:
            continue
        try:
            # Decode the WIRE exactly once, independently of Pydantic dump/revalidation.
            value = _decode_candidate_value(update['candidate_value'])
            values.append((update['slot_id'], dumps(value), update['status']))
        except (ValueError, TypeError):
            values.append((str(update['slot_id']), '<invalid-inner-JSON>', str(update['status'])))
    return values


def admit(raw, wire, capped=False, refused=False):
    from forecasting_assistant.domain.models import ExtractorResult
    if capped or refused:
        raise ValueError('Capped or refused response')
    return ExtractorResult.model_validate_json(raw)


def raw_score(record, turn, wire):
    from jsonschema import Draft202012Validator
    obj = None; json_valid = schema_valid = admitted = False; error = None
    try:
        decoded=json.loads(record['raw_output']);json_valid=True
        obj=decoded if isinstance(decoded,dict) else None
    except Exception as exc:error=type(exc).__name__
    if json_valid:
        try:Draft202012Validator(wire).validate(obj);schema_valid=True
        except Exception:pass
    try:admit(record['raw_output'],wire,record.get('capped',False),record.get('refused',False));admitted=True
    except Exception as exc:error=type(exc).__name__
    gold = Counter(tuples(turn['gold_extraction'])); emitted = Counter(tuples(obj))
    tp = sum((gold & emitted).values()) if admitted else 0
    gold_slots = {x[0] for x in gold}
    return {'confusion': [tp, sum(emitted.values()) - tp, sum(gold.values()) - tp],
            'json_valid': json_valid, 'schema_valid': schema_valid, 'raw_contract_admitted': admitted,
            'error_type': error, 'extra_slot_flags': sum(n for t,n in emitted.items() if t[0] not in gold_slots),
            'forbidden_slot_flags': sum(n for t,n in emitted.items() if t[0] in turn['forbidden_slots']),
            'correct_correction': admitted and bool((obj or {}).get('correction_detected')) and emitted == gold,
            'unknown_no_updates': admitted and not emitted,
            'ignored_status_updates':sum(u.get('slot_id')!='intent' and u.get('status') in {'unmentioned','dont_care'} for u in (obj or {}).get('updates',[]) if isinstance(u,dict)),
            'wire_empty_nonintent_updates':admitted and not any(u.get('slot_id')!='intent' for u in (obj or {}).get('updates',[]) if isinstance(u,dict)),
            'duplicate_slots': isinstance(obj,dict) and isinstance(obj.get('updates'),list) and len([u.get('slot_id') for u in obj['updates'] if isinstance(u,dict)]) != len({u.get('slot_id') for u in obj['updates'] if isinstance(u,dict)})}


class MemoryRepository:
    def __init__(self):
        self.states = {}; self.events = []; self.specifications = {}
    def save_state(self, state):
        self.states[state.dialogue_id] = state.model_copy(deep=True)
    def load_state(self, dialogue_id):
        state = self.states.get(dialogue_id)
        return state.model_copy(deep=True) if state is not None else None
    def append_event(self, dialogue_id, event_type, payload):
        self.events.append({'type': event_type, 'payload': deepcopy(payload)})
    def save_specification(self, spec):
        self.specifications[spec.dialogue_id] = spec


def run_pipeline(state, message, schema, record, wire):
    """Actual current engine, deterministic question renderer, no question API calls.

    Fixed-script question context is explicitly None for BOTH systems. Generated
    static question text is recorded, then removed before the next scripted turn.
    """
    from forecasting_assistant.application.orchestrator import ElicitationEngine
    from forecasting_assistant.prompts.extractor import build_extractor_input
    from forecasting_assistant.prompts.llmrei_long import static_fallback_question
    class RecordedProvider:
        captured = []
        async def extract(self, current, prior):
            self.captured.append(enrich_context(build_extractor_input(current, prior, schema),schema))
            return admit(record['raw_output'], wire, record.get('capped',False), record.get('refused',False))
        async def ask(self, request):
            return static_fallback_question(request)
    state = state.model_copy(deep=True)
    for turn in state.turns:
        turn.assistant_message = None
    repository = MemoryRepository(); repository.save_state(state)
    provider = RecordedProvider(); trace = []
    engine = ElicitationEngine(schema, provider, repository, trace_sink=lambda stage,payload:trace.append({'stage':stage,'payload':payload}))
    output = asyncio.run(engine.handle_user_message(state.dialogue_id, message))
    expected_context = extraction_context(state, message, schema)
    if provider.captured != [expected_context]:
        raise ValueError('Engine bypassed extraction or changed frozen input')
    recovered = [event['payload']['slot_id'] for event in repository.events if event['type']=='deterministic_recovery_applied']
    return output.state, {'recovered_slot_ids': recovered, 'trace': trace, 'events': repository.events, 'static_question': output.assistant_message}


def prf(rows):
    tp, fp, fn = [sum(row['confusion'][i] for row in rows) for i in range(3)]
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
