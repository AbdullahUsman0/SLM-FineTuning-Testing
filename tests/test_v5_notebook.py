import ast
import json
import unittest
from pathlib import Path


class V5NotebookTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks/Qwen_2B_LoRA_v5_Grounded_Colab.ipynb'
        self.notebook = json.loads(path.read_text(encoding='utf-8'))
        self.code = {cell['id']: ''.join(cell['source']) for cell in self.notebook['cells'] if cell['cell_type'] == 'code'}

    def test_all_cells_compile_without_stored_outputs(self):
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_pilot_serialization_preserves_embedded_prompt_newlines(self):
        tree = ast.parse(self.code['verify-and-pilot-inputs'])
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'data' for target in node.targets))
        row = {'prompt': [{'role': 'user', 'content': 'Line one\nLine two'}], 'completion': []}
        scope = {'json': json, 'rows': [row], 'count': 1}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'serialization', 'exec'), scope)
        self.assertEqual(len(scope['data'].splitlines()), 1)
        self.assertEqual(json.loads(scope['data']), row)

    def test_training_cells_require_drive_and_pinned_model(self):
        for stage in ['pilot', 'full-train-reviewed']:
            code = self.code[stage]
            self.assertIn('drive.mount', code)
            self.assertIn("'--revision'", code)
            self.assertIn("'--resume-from-checkpoint', 'auto'", code)
            self.assertNotIn('--allow-truncation', code)
        self.assertIn('human-review-approval.json', self.code['full-train-reviewed'])
        self.assertIn('corpus_manifest_sha256', self.code['full-train-reviewed'])

    def test_preflight_uses_public_corpus_verifier_cli(self):
        code = self.code['verify-and-pilot-inputs']
        self.assertIn("'--output'", code)
        self.assertNotIn("'--corpus'", code)
        self.assertNotIn("'--sealed-output'", code)

    def test_api_secret_is_ephemeral_and_cost_gated(self):
        code = self.code['openai-validation-opt-in']
        self.assertIn("userdata.get('OPENAI_API_KEY')", code)
        self.assertIn("os.environ.pop('OPENAI_API_KEY', None)", code)
        self.assertIn('V5_OPENAI_COST_APPROVED', code)
        self.assertNotIn('.env', code.replace('os.environ', 'environment'))


if __name__ == '__main__':
    unittest.main()
