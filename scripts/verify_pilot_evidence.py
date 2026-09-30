"""Verify public Pilot bytes and immutable source snapshots without ML dependencies."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('tactile-history-control', 'active-tactile-insertion')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    total = 0
    for group in GROUPS:
        for manifest_path in sorted((ROOT / 'docs/research-results' / group).glob('*/EXPORT_MANIFEST.json')):
            run = manifest_path.parent
            export = json.loads(manifest_path.read_text())
            for relative, record in export['files'].items():
                path = run / relative
                assert path.resolve().is_relative_to(run.resolve()), relative
                assert path.stat().st_size == record['bytes'], path
                assert digest(path) == record['sha256'], path
                total += 1
            original = json.loads((run / 'manifest.json').read_text())
            for relative, expected in original['source'].items():
                assert digest(run / 'source' / relative) == expected, relative
            assert digest(run / 'config.yaml') == original['config_sha256'], run
            print(f'{group}/{run.name}: {len(export["files"])} public files, original source/config verified')
    assert total, 'No exported Pilot evidence found'
    print(f'PASS: {total} public files; omitted local files are not required for this check.')


if __name__ == '__main__':
    main()
