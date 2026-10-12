"""Complete publication with a versioned packaging-only import repair."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
import zipfile

RUN = Path(r'D:\SLM\FYP-model-runs\qwen35-2b-lora-v7-r8-pilot-20261010T002204Z').resolve()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN / 'continuation-code'))
from evaluate_r8 import atomic, lock, now, sha, verify_freeze


def main():
    with lock(RUN / 'continuation.lock'):
        verify_freeze(RUN)
        original = RUN / 'continuation-code/finish_r8.py'
        corrected = HERE / 'finish_r8_packaging_fixed.py'
        original_bytes = original.read_bytes()
        separator = b'\r\n' if b'import json\r\n' in original_bytes else b'\n'
        assert corrected.read_bytes() == original_bytes.replace(b'import json' + separator,
            b'import hashlib' + separator + b'import json' + separator, 1)
        spec = importlib.util.spec_from_file_location('finish_r8_packaging_fixed', corrected)
        finisher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(finisher)

        # The failed path was ZIP member hashing. Exercise it before publishing.
        tests = HERE / 'zip-check-test'
        tests.mkdir(exist_ok=False)
        payload = tests / 'payload.json'
        payload.write_bytes(b'{"test":"zip member verification"}\n')
        tested_zip = tests / 'verified.zip'
        checksum = finisher.zip_checked(tested_zip, {'payload.json': payload})
        assert checksum == sha(tested_zip)
        assert tested_zip.with_name(tested_zip.name + '.sha256').read_text().split()[0] == checksum
        try:
            finisher.zip_checked(tested_zip, {'payload.json': payload})
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing archive was overwritten')
        assert sha(tested_zip) == checksum
        print('Packaging regression checks passed: CRC, inventory, member SHA, archive SHA, no overwrite.', flush=True)

        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        preserved = RUN / 'packaging-recovery' / stamp
        preserved.mkdir(parents=True, exist_ok=False)
        out = RUN / 'continuation-results'
        old_adapter_zip = RUN / (RUN.name + '-adapter.zip')
        assert out.is_dir() and old_adapter_zip.is_file()
        assert not (RUN / (RUN.name + '-evaluation.zip')).exists()
        archived = []
        for source in (out, old_adapter_zip):
            destination = preserved / source.name
            # These are named paths; both resolved targets must stay inside this run.
            assert source.resolve().is_relative_to(RUN) and destination.resolve().is_relative_to(RUN)
            assert not destination.exists()
            archived.append({'source': str(source), 'preserved_as': str(destination)})
            source.rename(destination)
        shutil.copyfile(RUN / 'continuation-status.json', preserved / 'failed-continuation-status.json')
        previous_results = preserved / 'continuation-results'
        receipt = {'status': 'packaging_only_repair', 'started_at': now(),
            'failure': "NameError: name 'hashlib' is not defined in ZIP member verification",
            'repair': 'Versioned copy of finish_r8.py adds only import hashlib. Frozen source remains byte-identical.',
            'frozen_packager_sha256': sha(original), 'corrected_packager_sha256': sha(corrected),
            'publication_wrapper_sha256': sha(__file__),
            'protocol_fingerprint': json.loads((RUN / 'evaluation-freeze.json').read_bytes())['fingerprint'],
            'preserved_previous_artifacts': archived,
            'regression_checks': ['ZIP CRC', 'member inventory', 'member SHA-256', 'archive SHA-256', 'existing archive overwrite rejected'],
            'evaluation_calls_repeated': 0, 'training_restarted': False, 'paid_API_calls': 0,
            'sealed_final_accessed': False,
            'latency_note': 'V7 was resumed twice after unexpected PC shutdowns; three GPU sessions and three unscored warmups. Resume receipts are included under evaluation/v7 in the result archive.'}
        atomic(preserved / 'repair-receipt.json', receipt)

        actual_zip_checked = finisher.zip_checked

        def checked_packaging(path, files):
            if path.name.endswith('-adapter.zip'):
                # The unmodified scorer regenerated these files. Require exact reproduction.
                for name in ('summary.csv', 'stratified-summary.json', 'paired-comparisons.json', 'next-round-decision.json'):
                    if (out / name).read_bytes() != (previous_results / name).read_bytes():
                        raise ValueError('Packaging recovery changed research results: ' + name)
                receipt['scored_summaries_and_intervals_byte_identical'] = True
                atomic(out / 'packaging-recovery.json', receipt)
                shutil.copyfile(corrected, out / corrected.name)
                shutil.copyfile(Path(__file__), out / Path(__file__).name)
                readme = out / 'README.md'
                with readme.open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write('\nPublication recovery: a missing standard-library import in the ZIP member verifier was repaired in a versioned copy of the packager. Frozen evaluation sources, labels, raw outputs, scores and bootstrap intervals remain unchanged; original failed artifacts were preserved locally. See packaging-recovery.json. V7 spans three GPU sessions after two interruption recoveries, with three unscored warmups; latency comparisons are descriptive and do not establish a controlled advantage.\n')
            return actual_zip_checked(path, files)

        finisher.zip_checked = checked_packaging
        atomic(RUN / 'continuation-status.json', {'status': 'running', 'phase': 'packaging_recovery_verification_and_publication',
               'pid': os.getpid(), 'updated_at': now(), 'scored_calls': 6240})
        try:
            result = finisher.finish(RUN)
            result['packaging_recovery_receipt'] = str(out / 'packaging-recovery.json')
            result['adapter_zip_sha256'] = sha(RUN / (RUN.name + '-adapter.zip'))
            atomic(RUN / 'continuation-completion.json', result)
            atomic(RUN / 'continuation-status.json', dict(result, phase='complete'))
            print(json.dumps(result, indent=2), flush=True)
        except BaseException as error:
            atomic(RUN / 'continuation-status.json', {'status': 'failed', 'phase': 'packaging_recovery',
                   'error_type': type(error).__name__, 'error': str(error), 'updated_at': now(),
                   'preserved_prior_artifacts': str(preserved), 'training_restarted': False})
            raise


if __name__ == '__main__':
    main()
