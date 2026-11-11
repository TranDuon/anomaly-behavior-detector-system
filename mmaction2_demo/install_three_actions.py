"""Verify the separate three-label pilot package before installation."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from action_engine import ROOT
from three_actions_engine import LABELS, PARENT_SHA, load_three_actions


def install(archive, root=ROOT):
    root = Path(root)
    (root / 'results').mkdir(exist_ok=True)
    names = {'three_actions.pt', 'three_actions.json', 'report.json', 'protocol.json'}
    with tempfile.TemporaryDirectory(dir=root / 'results') as directory:
        staging = Path(directory)
        with zipfile.ZipFile(archive) as package:
            if len(package.namelist()) != len(names) or set(package.namelist()) != names:
                raise ValueError('Unexpected three-action package members')
            for name in names:
                if package.getinfo(name).file_size > 32 * 1024 * 1024:
                    raise ValueError('Oversized package member')
                (staging / name).write_bytes(package.read(name))
        metadata, _ = load_three_actions(staging)
        report = json.loads((staging / 'report.json').read_text())
        protocol = json.loads((staging / 'protocol.json').read_text())
        if (report.get('status') != 'completed' or not metadata.get('fingerprint')
                or report.get('fingerprint') != metadata['fingerprint']
                or protocol != metadata['fingerprint']
                or protocol.get('parent_checkpoint_sha256') != PARENT_SHA
                or set(report.get('test', {})) != set(LABELS)
                or set(report.get('development', {})) != set(LABELS)):
            raise ValueError('Incomplete three-action report or provenance')
        targets = {name: root / ('weights' if name.startswith('three_actions.') else 'results') /
                   (name if name.startswith('three_actions.') else 'three_actions_' + name) for name in names}
        backup = root / 'results' / ('three_actions_backup_' + str(time.time_ns()))
        existing = {name: target for name, target in targets.items() if target.is_file()}
        if existing:
            backup.mkdir()
            for name, target in existing.items():
                shutil.copyfile(target, backup / name)
        try:
            for name, target in targets.items():
                target.parent.mkdir(exist_ok=True)
                temporary = target.with_suffix(target.suffix + '.tmp')
                shutil.copyfile(staging / name, temporary)
                temporary.replace(target)
        except OSError:
            for name, target in targets.items():
                if name in existing:
                    shutil.copyfile(backup / name, target)
                else:
                    target.unlink(missing_ok=True)
            raise
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    arguments = parser.parse_args()
    print('Three-action pilot installed:', install(arguments.archive)['checkpoint_sha256'])
