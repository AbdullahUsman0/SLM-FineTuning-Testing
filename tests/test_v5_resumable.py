import importlib.util
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class V5ResumableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('test_raw_resume', ROOT / 'scripts/evaluate-v5-resumable.py')
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_final_and_hosted_are_not_accepted(self):
        for option in (['--provider', 'openai'], ['--split', 'final']):
            with self.assertRaises(SystemExit):
                self.runner.parser().parse_args(option)

    def test_interruption_replays_only_unfinished_scenario_and_retains_partial(self):
        m = self.runner
        schema = m.load_schema()
        cases, metadata = m.load_cases(ROOT / 'corpus-v5/v5-20260919-r1/splits/validation.jsonl', split='validation')
        cases = cases[:2]

        class FailingProvider:
            async def extract(self, message, state):
                raise ValueError('mock parse failure')

            async def ask(self, request):
                raise ValueError('mock parse failure')

            def drain_traces(self):
                return []

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args = m.parser().parse_args(['--provider', 'base', '--model', 'model', '--revision', 'a' * 40,
                                          '--run-dir', directory])
            stack.enter_context(patch.object(m, 'load_cases', return_value=(cases, metadata)))
            stack.enter_context(patch.object(m.DURABLE.BASE, 'verify_frozen_study', return_value={'sha256': 'frozen'}))
            stack.enter_context(patch.object(m, 'provenance', return_value={'versions': {}}))
            stack.enter_context(patch.object(m.RAW, 'settings_for', return_value={'provider': 'base'}))
            constructor = stack.enter_context(patch.object(m.RAW, 'create_provider', return_value=(FailingProvider(), {})))
            stack.enter_context(patch.object(m, 'runtime_metadata', return_value={'actual_device': 'cpu'}))
            stack.enter_context(patch.object(m.RAW, 'model_artifacts', return_value={'snapshot_sha256': 'hash'}))
            original = m.evaluate
            calls = []

            async def interrupted(provider, scenarios, **kwargs):
                calls.append(scenarios[0]['scenario_id'])
                if len(calls) == 3:  # warmup, completed first scenario, interrupted second
                    raise KeyboardInterrupt()
                return await original(provider, scenarios, **kwargs)

            with patch.object(m, 'evaluate', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    m.run(args)
            self.assertEqual(len(list((Path(directory) / 'scenarios').glob('*.json'))), 1)
            calls.clear()

            async def resumed(provider, scenarios, **kwargs):
                calls.append(scenarios[0]['scenario_id'])
                return await original(provider, scenarios, **kwargs)

            with patch.object(m, 'evaluate', side_effect=resumed):
                report = m.run(args)
            self.assertEqual(calls, [cases[0]['scenario_id'], cases[1]['scenario_id']])
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['scenario_count'], 2)
            self.assertEqual(report['metrics']['provider_success']['rate'], 0)
            self.assertEqual(len(list((Path(directory) / 'interrupted').glob('*.json'))), 1)
            self.assertEqual(len(report['records']), len(m.build_call_plan(cases, schema)))
            constructor.reset_mock()
            self.assertEqual(m.run(args)['records'], report['records'])
            constructor.assert_not_called()
            with patch.object(m, 'provenance', return_value={'versions': {'changed': '1'}}):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    m.run(args)
            constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
