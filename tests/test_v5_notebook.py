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

    def test_preflight_counts_tokens_from_transformers_mapping_results(self):
        code = self.code['verify-and-pilot-inputs']
        self.assertIn('isinstance(tokenized, Mapping)', code)
        self.assertIn("tokenized = tokenized['input_ids']", code)
        self.assertIn('isinstance(tokenized[0], list)', code)
        self.assertIn('lengths.append(len(tokenized))', code)

    def test_api_secret_is_ephemeral_and_cost_gated(self):
        code = self.code['openai-validation-opt-in']
        self.assertIn("userdata.get('OPENAI_API_KEY')", code)
        self.assertIn("os.environ.pop('OPENAI_API_KEY', None)", code)
        self.assertIn('V5_OPENAI_COST_APPROVED', code)
        self.assertNotIn('.env', code.replace('os.environ', 'environment'))


class V5PortableNotebookTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks/Qwen_2B_LoRA_v5_Portable_10Step_Colab.ipynb'
        self.notebook = json.loads(path.read_text(encoding='utf-8'))
        self.code = {cell['id']: ''.join(cell['source']) for cell in self.notebook['cells'] if cell['cell_type'] == 'code'}

    def test_all_cells_compile_without_stored_outputs(self):
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_training_saves_every_ten_steps_and_guards_resume_floor(self):
        config = self.code['portable-config']
        train = self.code['portable-train']
        self.assertIn('CHECKPOINT_INTERVAL = 10', config)
        self.assertIn('MINIMUM_RESUME_STEP = 0', config)
        self.assertIn("'--save-steps', str(CHECKPOINT_INTERVAL)", train)
        self.assertIn("'--minimum-resume-step', str(MINIMUM_RESUME_STEP)", train)
        self.assertIn("'--resume-from-checkpoint', 'auto'", train)
        self.assertNotIn('output.mkdir', train)

    def test_transfer_bundle_is_verified_before_import(self):
        code = self.code['portable-import']
        self.assertIn("member.issym() or member.islnk()", code)
        self.assertIn("str(target).startswith(str(staging.resolve()) + os.sep)", code)
        self.assertIn("transfer-manifest.json", code)
        self.assertIn("sha256(path) == expected['sha256']", code)
        self.assertIn("Import destination must be empty", code)

    def test_export_includes_exact_resume_state_and_receipt_hash(self):
        code = self.code['portable-export']
        for filename in ['optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth', 'checkpoint-complete.json']:
            self.assertIn(filename, code)
        self.assertIn("'sha256': sha256(bundle)", code)
        self.assertIn("transfer-manifest.json", code)

    def test_notebook_pins_resume_guard_commit_and_avoids_final_set(self):
        all_code = '\n'.join(self.code.values())
        self.assertIn('7469e8b516474d684f5859f1e9a6a09078af944b', self.code['portable-config'])
        self.assertNotIn('splits/final.jsonl', all_code)
        self.assertNotIn('sft/final.jsonl', all_code)


if __name__ == '__main__':
    unittest.main()
