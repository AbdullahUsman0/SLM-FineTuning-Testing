"""Credential scan with one byte-pinned, audited encrypted-payload false positive.

The raw SDK response is retained unchanged. This exception never admits a match
of the configured user key, any other file, or any other JSON field.
"""
import hashlib
import json
import re

PATTERN = re.compile(rb'(?i)(?<![A-Za-z0-9])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,})')
OPAQUE_RESPONSE_SHA256 = '4c3e5e6c1394dc0b3b7d27e930637503fbd6cd5ff403db78d9adca2221054f3b'


def scan_payload(body, secret):
    if isinstance(secret, str):
        secret = secret.encode()
    if secret and secret in body:
        raise ValueError('Configured credential found; content withheld')
    if not PATTERN.search(body):
        return None
    if hashlib.sha256(body).hexdigest() != OPAQUE_RESPONSE_SHA256:
        raise ValueError('Unadjudicated credential-shaped content; content withheld')
    obj = json.loads(body)
    output = obj['response']['output'][0]
    assert output['type'] == 'reasoning'
    encrypted = output['encrypted_content']
    assert isinstance(encrypted, str) and PATTERN.search(encrypted.encode())
    ciphertext_sha = hashlib.sha256(encrypted.encode()).hexdigest()
    output['encrypted_content'] = '[OPAQUE ENCRYPTED REASONING FIELD OMITTED ONLY FROM SCANNER VIEW]'
    if PATTERN.search(json.dumps(obj, ensure_ascii=False).encode()):
        raise ValueError('Credential-shaped content outside audited opaque field')
    return {'exception': 'opaque encrypted-reasoning bytes matched a generic token pattern',
            'raw_response_sha256': OPAQUE_RESPONSE_SHA256,
            'json_path': 'response.output[0].encrypted_content',
            'ciphertext_sha256': ciphertext_sha,
            'configured_credential_match': False,
            'raw_response_modified': False,
            'other_files_and_fields_exempted': False}
