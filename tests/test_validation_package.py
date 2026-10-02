import asyncio
import copy
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ValidationPackageTests(unittest.TestCase):
    def test_lossless_reports_and_matched_runtime_gate(self):
        spec = importlib.util.spec_from_file_location('package_test', ROOT / 'scripts/package-base-final-validation.py')
        package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package)
        from local_slm_lab.v5_eval import evaluate, load_cases, load_schema, sha256

        class Provider:
            async def extract(self, message, state):
                raise ValueError('test failure')

            async def ask(self, request):
                raise ValueError('test failure')

        cases, input_metadata = load_cases(ROOT / 'corpus-v5/v5-20260919-r1/splits/validation.jsonl', split='validation')
        settings = {'provider': 'base', 'model': 'model', 'revision': 'a' * 40, 'dtype': 'bfloat16',
                    'device': 'cuda', 'prompt_sha256': 'prompt', 'decoding': {}, 'max_new_tokens': 1024,
                    'scorer_sha256': 'scorer', 'dependency_sha256': 'dependency'}
        metadata = {'input': input_metadata, 'settings': settings, 'runtime': {'actual_device': 'cuda:0'}}
        base = asyncio.run(evaluate(Provider(), cases, schema=load_schema(), metadata=metadata))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / 'training'
            adapter = training / 'full/best-adapter'
            adapter.mkdir(parents=True)
            for name in ('adapter_model.safetensors', 'adapter_config.json'):
                (adapter / name).write_text('test only', encoding='utf-8')
            package.write_json(training / 'completion-audit.json', {'status': 'passed'})
            (training / 'TRAINING_COMPLETION_SUMMARY.md').write_text('Test summary', encoding='utf-8')
            candidate = copy.deepcopy(base)
            candidate['metadata']['settings']['provider'] = 'lora'
            candidate['metadata']['settings']['adapter_sha256'] = {p.name: sha256(p) for p in adapter.iterdir()}
            for name, report in (('base', base), ('lora', candidate)):
                (root / name).mkdir()
                package.write_json(root / name / 'combined-report.json', report)
            comparison = package.package(root, training, root / 'published')
            self.assertEqual(comparison['runtime_differences'], [])
            self.assertEqual(comparison['nonintent']['delta'], 0)
            with gzip.open(root / 'published/lora-validation.json.gz', 'rt', encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['records'], candidate['records'])
            hashes = json.loads((root / 'published/artifact-hashes.json').read_text(encoding='utf-8'))
            self.assertTrue(all(sha256(root / 'published' / name) == info['sha256'] for name, info in hashes.items()))
            candidate['metadata']['runtime']['actual_device'] = 'cpu'
            package.write_json(root / 'lora/combined-report.json', candidate)
            with self.assertRaisesRegex(ValueError, 'runtime'):
                package.package(root, training, root / 'refused')
            self.assertFalse((root / 'refused').exists())


if __name__ == '__main__':
    unittest.main()
