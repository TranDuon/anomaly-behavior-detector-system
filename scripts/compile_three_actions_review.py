"""Compile manually inspected public contact-sheet windows into train input."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from mmaction2_demo.train_three_actions import LABELS, audit


ROOT = Path(__file__).resolve().parents[1]


def compile_review(pool_dir):
    pool_dir = Path(pool_dir)
    pool = json.loads((pool_dir/'pool.json').read_text())
    review_path = ROOT/'benchmarks/three_actions_visual_review_v1.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    sources = []
    for item in pool:
        if item['id'] not in review['sources']:
            raise ValueError('Not visually reviewed: ' + item['id'])
        windows = []
        for annotation in review['sources'][item['id']]:
            for anchor in annotation['anchors']:
                if not 0 <= anchor < item['duration']:
                    raise ValueError('Anchor outside source duration')
                windows.append({'anchor_seconds': anchor, 'targets': dict(zip(LABELS, annotation['targets'])),
                                'notes': annotation['notes']})
        sources.append({'video': 'three_actions_v1/videos/' + item['id'] + '.mp4',
                        'sha256': item['sha256'], 'group': item['group'], 'identity_group': item.get('subject_group', item['group']),
                        'split': item['split'], 'reviewer': review['reviewer'], 'review_method': review['method'],
                        'public_source_url': item['public_source_url'], 'download_url': item['download_url'],
                        'blob_url': item['blob_url'], 'size': item['size'], 'git_blob_sha1': item['git_blob_sha1'],
                        'contact_sheet_sha256': hashlib.sha256((pool_dir/(item['id']+'.jpg')).read_bytes()).hexdigest(),
                        'windows': windows})
    manifest = {'schema_version': 1, 'labels': list(LABELS), 'sources': sources,
                'review_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
                'limitations': ['AI-assisted review, not independent human gold.',
                                'AIRTLab actors recur across recordings; only recording-level split verified.',
                                'GMDCSA subjects 1-2 train, 3 development, 4 test.',
                                'Some test clips previously inspected for debugging; no claim of blind acceptance accuracy.']}
    _, report = audit(manifest, ROOT, verify_files=False)
    if report['status'] != 'ready':
        raise ValueError(json.dumps(report))
    (pool_dir/'reviewed_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (pool_dir/'review_coverage.json').write_text(json.dumps(report, indent=2))
    bundle = pool_dir.parent/'three_actions_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(pool_dir/'reviewed_manifest.json', 'reviewed_manifest.json')
        archive.write(review_path, 'visual_review.json')
        for name in ['train_three_actions.py', 'action_engine.py', 'person_recovery.py']:
            archive.write(ROOT/'mmaction2_demo'/name, name)
    print(json.dumps({'sources': len(sources), 'windows': report['windows'], 'coverage': report['coverage'],
                      'bundle': str(bundle), 'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir', type=Path, required=True)
    compile_review(parser.parse_args().pool_dir)
