"""Frozen candidate prompt for a development-only v6 comparison."""
import json
from local_slm_lab.v5_prompts import build_v5_extractor_instructions


def build_revised_instructions() -> str:
    example = {
        'intent':'create_forecast','intent_confidence':1.0,
        'updates':[
            {'slot_id':'target_description','candidate_value':json.dumps('water demand'),
             'status':'provided','confidence':1.0,'evidence_text':'water demand'},
            {'slot_id':'target_unit','candidate_value':json.dumps('litres'),
             'status':'provided','confidence':1.0,'evidence_text':'litres'},
            {'slot_id':'frequency','candidate_value':json.dumps({'periods':1,'unit':'week'},separators=(',',':')),
             'status':'provided','confidence':1.0,'evidence_text':'weekly'},
            {'slot_id':'forecast_horizon','candidate_value':json.dumps({'periods':2,'unit':'week'},separators=(',',':')),
             'status':'provided','confidence':1.0,'evidence_text':'next 2 weeks'},
        ],
        'correction_detected':False,'unsupported_claims':[],
    }
    return build_v5_extractor_instructions() + (
        '\nCompleteness and encoding checklist, applied silently before returning JSON:\n'
        '1. Read every clause of current_message. A clause can state several independent facts. '
        'Create a separate update for every explicitly stated requirement; never merge facts into one update.\n'
        '2. Keep the target description separate from its measurement unit. Observation spacing is frequency; '
        'how far ahead to predict is forecast_horizon; how often to refresh a forecast is prediction_frequency. '
        'A target column and timestamp column require separate updates.\n'
        '3. candidate_value must always be a STRING containing valid JSON text. For free text, the inner JSON '
        'is a quoted string. For durations, the inner JSON is an object with numeric periods and a singular unit. '
        'Do not put an object directly in candidate_value and do not encode a duration as plain words.\n'
        '4. Use exactly one update for each slot_id and exactly one occurrence of each JSON key. '
        'Copy allowed enum values exactly, including lowercase file formats. Preserve meaningful target qualifiers.\n'
        '5. A request to replace a previous requirement is an explicit correction even if that requirement '
        'was previously missed. Update the requested slot only and set correction_detected true. '
        'Do not restate unchanged requirements from prior turns.\n'
        '6. Unknown, undecided, not yet chosen, and unavailable credentials give no value. Omit those updates; '
        'never supply defaults, null values, inferred numbers, or dont_care for unknown information. '
        'Retain established intent.\n'
        '7. Check that each value has current-message evidence and that no stated unit, frequency, horizon, '
        'format, source mode, column, or refresh interval was silently dropped. '
        'Only explicitly stated facts qualify; do not invent missing requirements.\n'
        'Illustration, not user data: for "Forecast water demand in litres weekly for the next 2 weeks", '
        'the wire response is:\n' + json.dumps(example,separators=(',',':')) + '\n'
        'Use only facts from the actual current_message, never facts from this illustration. '
        'Return the JSON response only, without a checklist or explanation.'
    )
