"""A frozen format-only addition to the prior v6 candidate prompt."""
from local_slm_lab.v6_prompt_revision import build_revised_instructions


SCHEMA_FIX = (
    '\nFinal output contract:\n'
    'The outer object has exactly these five keys: intent, intent_confidence, updates, '
    'correction_detected, unsupported_claims.\n'
    'updates is an array. EVERY object in that array has exactly these five keys: '
    'slot_id, candidate_value, status, confidence, evidence_text. No other keys are allowed.\n'
    'Do not add a type, value_type, properties, or schema key anywhere in an update. '
    'These are data updates, not schema definitions.\n'
    'target_unit, frequency, forecast_horizon, prediction_frequency, file_format and other '
    'slot names are VALUES of slot_id in separate array members. They must NEVER appear '
    'as extra property names inside another update.\n'
    'For several facts in one clause, create several separate five-key update objects. '
    'Give each slot_id at most one object. Give each key exactly one occurrence in its object.\n'
    'candidate_value remains a string containing JSON text, exactly as in the illustration above. '
    'Keep the existing extraction, evidence, correction and unknown-information rules unchanged.\n'
    'Before returning, silently check the five outer keys and five keys of EVERY update. '
    'Return only the data object. Do not print the check or the contract.'
)


def build_schema_fixed_instructions() -> str:
    return build_revised_instructions() + SCHEMA_FIX
