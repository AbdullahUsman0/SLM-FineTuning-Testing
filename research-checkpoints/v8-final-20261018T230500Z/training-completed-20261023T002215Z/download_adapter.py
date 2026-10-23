"""Download the public v8 release, verify it, and extract into a NEW directory."""
import argparse
import hashlib
import json
import stat
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def download(url, path):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != 'https' or parsed.netloc != 'github.com':
        raise ValueError('Expected the public GitHub release URL')
    request = urllib.request.Request(url, headers={'User-Agent': 'FYP-verified-adapter-download'})
    with urllib.request.urlopen(request, timeout=90) as response, path.open('xb') as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='NEW directory; existing paths are refused')
    parser.add_argument('--receipt', type=Path, default=Path(__file__).with_name('ADAPTER_RELEASE.json'))
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    if receipt['publication_status'] != 'published_prerelease_download_verified':
        raise RuntimeError('This receipt does not yet confirm a verified publication; pull the latest branch')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    archive = root / receipt['zip_name']
    checksum = root / receipt['checksum_name']
    download(receipt['zip_download_url'], archive)
    download(receipt['checksum_download_url'], checksum)
    if sha(archive) != receipt['zip_sha256'] or sha(checksum) != receipt['checksum_sha256']:
        raise RuntimeError('STOP: downloaded asset SHA-256 mismatch; nothing extracted')
    fields = checksum.read_text(encoding='ascii').strip().split(maxsplit=1)
    if fields != [receipt['zip_sha256'], receipt['zip_name']]:
        raise RuntimeError('STOP: checksum asset references unexpected bytes/name')
    extract = root / 'extracted'
    extract.mkdir(exist_ok=False)
    expected = receipt['members_sha256']
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        if (set(names) != set(expected) | {'ARCHIVE_MANIFEST.json'} or
                len(names) != len(set(name.casefold() for name in names)) or package.testzip() is not None):
            raise RuntimeError('STOP: archive inventory/CRC mismatch')
        embedded = json.loads(package.read('ARCHIVE_MANIFEST.json'))['files_sha256']
        if embedded != expected:
            raise RuntimeError('STOP: embedded manifest differs from the committed receipt')
        for name, digest in expected.items():
            if hashlib.sha256(package.read(name)).hexdigest() != digest:
                raise RuntimeError('STOP: archive member hash mismatch: ' + name)
        for info in package.infolist():
            parts = PurePosixPath(info.filename)
            if (parts.is_absolute() or '..' in parts.parts or ':' in info.filename or '\\' in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise RuntimeError('Unsafe archive path/type')
            target = (extract / Path(*parts.parts)).resolve()
            if not target.is_relative_to(extract):
                raise RuntimeError('Archive path escapes extraction directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(package.read(info.filename))
    for name, expected_hash in receipt['adapter_files_sha256'].items():
        if sha(extract / 'best-adapter' / name) != expected_hash:
            raise RuntimeError('STOP: extracted adapter differs from training completion')
    print(json.dumps({'status': 'download_and_extraction_verified', 'zip_sha256': sha(archive),
        'verified_members': len(expected), 'adapter': str(extract / 'best-adapter'),
        'completion': str(extract / 'training-completion.json'), 'output': str(root)}))


if __name__ == '__main__':
    main()
