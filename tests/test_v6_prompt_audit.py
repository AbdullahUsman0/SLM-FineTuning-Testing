import importlib.util
from copy import deepcopy
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('natural_audit',ROOT/'scripts/audit-v6-natural-failures.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.schema=audit.load_schema()
        cases,_=audit.load_cases(ROOT/'evaluation/v6-matched-20261005/natural.jsonl',split='validation')
        self.calls=audit.build_call_plan(cases[:1],self.schema)

    def row(self,call):
        return {'expected':call['expected'],'emitted':deepcopy(call['expected']),
                'prediction_valid':True,'test_kind':'initial','context':call['context']}

    def test_normalization_accepts_existing_enum_case_but_not_omission(self):
        row=self.row(self.calls[0])
        row['emitted']['updates'][-1]['candidate_value']='CSV'
        self.assertEqual(audit.normalized_counts([row],self.schema)['f1'],1)
        row['emitted']['updates'].pop()
        self.assertLess(audit.normalized_counts([row],self.schema)['f1'],1)

    def test_normalization_never_certifies_invalid_wire_content(self):
        row=self.row(self.calls[0]);row['prediction_valid']=False
        self.assertEqual(audit.normalized_counts([row],self.schema)['tp'],0)

    def test_corrected_horizon_can_succeed_without_full_gold_state(self):
        call=self.calls[1];row=self.row(call)
        row.update(test_kind='correction',gold_after=call['gold_after'],predicted=call['expected'])
        row['predicted_after']=deepcopy(call['gold_after'])
        row['predicted_after']['slots']['target_unit']['value']='wrong earlier unit'
        row['context']=deepcopy(call['context'])
        row['context']['state']['slots']['target_unit']['value']='wrong earlier unit'
        result=audit.normalized_metrics([row],self.schema)
        self.assertEqual(result['correction_horizon_after_reducer']['numerator'],1)
        self.assertEqual(result['correction_horizon_and_other_prior_slots_preserved']['numerator'],1)
        self.assertNotEqual(row['predicted_after'],row['gold_after'])

    def test_candidate_prompt_keeps_json_text_wire_contract(self):
        text=audit.build_revised_instructions()
        example=text.split('the wire response is:\n',1)[1].split('\n',1)[0]
        parsed=audit.strict_json_object(example)
        result=audit.validate_raw_output(parsed,audit.ExtractorResult)
        duration=next(u for u in result.updates if u.slot_id=='forecast_horizon')
        self.assertEqual(duration.candidate_value,{'periods':2,'unit':'week'})

if __name__=='__main__':unittest.main()
