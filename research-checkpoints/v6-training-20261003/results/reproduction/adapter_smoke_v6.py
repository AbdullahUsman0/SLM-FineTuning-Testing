"""One synthetic weather extraction to check the saved specialist can load."""
import asyncio
import json
from pathlib import Path
import sys

run = Path(sys.argv[1]).resolve()
config = json.loads((run / 'launch-config.json').read_text(encoding='utf-8'))
sys.path.insert(0, config['repository'])
from local_slm_lab.peft_provider import PeftInferenceConfig
from local_slm_lab.v5_provider import V5TransformersProvider
from forecasting_assistant.domain.schema import load_schema, create_initial_state

schema = load_schema()
provider = V5TransformersProvider(PeftInferenceConfig(base_model=config['model'], revision=config['model_revision'],
    variant='custom', adapter_path=run / 'full/best-adapter', device='cuda', dtype='bfloat16', max_new_tokens=1024), schema)
message = 'I want to forecast daily maximum temperature in Celsius for the next 7 days using a CSV file.'
result = asyncio.run(provider.extract(message, create_initial_state(schema)))
report = {'status': 'passed', 'purpose': 'Single synthetic loading/generation check, not behavioral validation',
          'input': message, 'output': result.model_dump(mode='json'), 'device': str(provider._input_device)}
(run / 'adapter-smoke.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
