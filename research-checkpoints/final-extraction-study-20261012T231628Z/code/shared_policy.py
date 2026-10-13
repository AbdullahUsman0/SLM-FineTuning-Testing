"""New policy version: historical structural admission plus exact evidence gating."""
from dataclasses import dataclass, replace
import runtime as rt

VERSION = 'fresh-exact-evidence-pydantic-once-20261012-1'


@dataclass(frozen=True)
class SharedAdmission:
    structural: object
    evidence_valid: bool
    admitted: bool
    error_code: str | None

    @property
    def predicted(self):
        return self.structural.predicted

    def to_model(self):
        if not self.admitted:
            raise ValueError('No model for rejected admission')
        return self.structural.to_model('extract')

    def audit(self):
        return {**self.structural.to_dict(), 'structural_parser_admitted': self.structural.admitted,
                'evidence_valid': self.evidence_valid, 'admitted': self.admitted, 'error_code': self.error_code}


class SharedPolicy:
    def __init__(self):
        self.reference = rt.ComparisonPolicy()
        self.schema = self.reference.schemas['extract']
        self.version = VERSION
        freeze = rt.CHECKPOINT / 'policy/freeze-manifest.json'
        if freeze.exists():
            manifest = rt.read(freeze)
            if manifest['policy_version'] != VERSION or manifest['reference_policy_fingerprint'] != self.reference.fingerprint:
                raise ValueError('Shared policy identity changed')
            for name, expected in manifest['files_sha256'].items():
                if rt.sha(rt.CHECKPOINT / name) != expected:
                    raise ValueError('Shared policy source/artifact changed: ' + name)

    def parse(self, raw, message, *, response_status='completed', refused=False, hit_generation_limit=False):
        reference = self.reference.parse(raw, 'extract', response_status=response_status,
            refused=refused, hit_generation_limit=hit_generation_limit)
        grounded = bool(reference.admitted and all(update['evidence_text'].strip()
            and update['evidence_text'] in message for update in reference.predicted['updates']))
        return SharedAdmission(reference, grounded, grounded,
            reference.error_code if not reference.admitted else None if grounded else 'exact_current_evidence')

    def advance(self, state, admission, schema, turn, message):
        if not admission.admitted:
            return state.model_copy(deep=True), False, admission.error_code
        try:
            result = rt.apply_extraction(state, admission.to_model(), schema, turn, message)
        except Exception as error:
            return state.model_copy(deep=True), False, type(error).__name__
        return result, True, None

    def emitted(self, raw, admission):
        """Count readable rejected emissions as FP; never silently drop them."""
        if admission.predicted is not None:
            return admission.predicted
        try:
            value = rt.strict_json(raw)
        except (ValueError, TypeError):
            return {'updates': []}
        updates = []
        if not isinstance(value, dict) or not isinstance(value.get('updates'), list):
            return {'updates': []}
        for update in value['updates']:
            if not isinstance(update, dict) or not {'slot_id', 'candidate_value', 'status'} <= update.keys():
                continue
            if not isinstance(update['slot_id'], str) or not isinstance(update['status'], str):
                continue
            candidate = update['candidate_value']
            if isinstance(candidate, str):
                try:
                    candidate = rt.strict_json(candidate)
                except (ValueError, TypeError):
                    pass
            try:
                rt.dumps(candidate)
            except (ValueError, TypeError):
                continue
            updates.append({'slot_id': update['slot_id'], 'candidate_value': candidate, 'status': update['status']})
        return {'updates': updates}


def admit_for_provider(provider, raw, message, **flags):
    if provider not in {'v6', 'v7', 'openai', 'manual', 'oracle'}:
        raise ValueError('Unknown provider')
    return SharedPolicy().parse(raw, message, **flags)
