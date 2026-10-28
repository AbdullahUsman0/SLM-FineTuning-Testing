"""Verify an extracted handoff bundle using only the Python standard library."""
from pathlib import Path
import hashlib,json

root=Path(__file__).resolve().parent
manifest=json.loads((root/'BUNDLE_MANIFEST.json').read_bytes())
for name,digest in manifest['files_sha256'].items():
    p=(root/name).resolve()
    if not p.is_relative_to(root) or not p.is_file():raise SystemExit('Missing or unsafe member: '+name)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise SystemExit('SHA-256 mismatch: '+name)
budget=json.loads((root/'API_BUDGET_AUTHORIZATION.json').read_bytes())
assert budget['total_user_ceiling_usd']==1.0 and budget['internal_dispatch_ceiling_usd']==0.9
assert budget['scope']=='one bounded current-pipeline v8/OpenAI evaluation, including pilots and retries'
print(json.dumps({'status':'verified','files':len(manifest['files_sha256']),'total_budget_usd':1.0,'runner_implemented':False}))
