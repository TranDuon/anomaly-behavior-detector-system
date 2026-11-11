"""Prepare score-blinded public review sheets; keep no extra local videos."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np


def request(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'three-action-review'}), timeout=60) as response:
                return response.read()
        except (OSError, http.client.HTTPException):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def selection():
    rows = []
    for purpose, folder, groups in [
        ('grapple_review', 'violent', {'train': [80, 81, 89], 'development': [38, 83], 'test': [37, 87]}),
        ('strike_review', 'violent', {'train': [7, 46, 77], 'development': [85, 107], 'test': [109, 101]}),
        ('hug_review', 'non-violent', {'train': [5, 6, 25], 'development': [12, 37], 'test': [38, 49]}),
    ]:
        for split, numbers in groups.items():
            for number in numbers:
                rows.append({'dataset': 'airtlab', 'group': f'airtlab/{folder}/{number}', 'split': split,
                             'path': f'violence-detection-dataset/{folder}/cam2/{number}.mp4', 'purpose': purpose})
    for subject, fall_ids in [(1, [1, 3]), (2, [1, 2]), (3, [3, 4]), (4, [1, 2])]:
        split = 'train' if subject <= 2 else 'development' if subject == 3 else 'test'
        for category, numbers in [('Fall', fall_ids), ('ADL', [1, 2])]:
            for number in numbers:
                rows.append({'dataset': 'gmdcsa24', 'group': f'gmdcsa24/subject_{subject}/{category}/{number:02}',
                             'subject_group': f'gmdcsa24/subject_{subject}', 'split': split,
                             'path': f'Subject {subject}/{category}/{number:02}.mp4', 'purpose': category.lower() + '_review'})
    for index, row in enumerate(rows):
        row['id'] = f'T{index + 1:03}'
    return rows


def prepare(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    repositories = {'airtlab': ('airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos', 'HEAD'),
                    'gmdcsa24': ('ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos', 'v2.1')}
    trees = {}
    commits = {}
    for name, (repo, revision) in repositories.items():
        commits[name] = json.loads(request(f'https://api.github.com/repos/{repo}/commits/{revision}'))['sha']
        tree = json.loads(request(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1'))
        if tree.get('truncated'):
            raise ValueError('Truncated publisher tree')
        trees[name] = tree
    rows = selection()
    for row in rows:
        repo, _ = repositories[row['dataset']]
        tree = trees[row['dataset']]
        revision = commits[row['dataset']]
        entry = next(item for item in tree['tree'] if item['path'] == row['path'])
        if entry['size'] > 25 * 1024**2:
            raise ValueError('Unexpectedly large review clip')
        row.update(git_blob_sha1=entry['sha'], size=entry['size'], revision=revision,
                   public_source_url=f'https://github.com/{repo}/blob/{revision}/{urllib.parse.quote(row["path"])}',
                   download_url=f'https://raw.githubusercontent.com/{repo}/{revision}/{urllib.parse.quote(row["path"])}',
                   blob_url=entry['url'])

    def process(row):
        saved = destination / (row['id'] + '.json')
        if saved.exists():
            previous = json.loads(saved.read_text())
            if previous['git_blob_sha1'] == row['git_blob_sha1'] and (destination / (row['id'] + '.jpg')).exists():
                return {**previous, 'revision': row['revision'], 'public_source_url': row['public_source_url'],
                        'download_url': row['download_url']}
        try:
            payload = (base64.b64decode(json.loads(request(row['blob_url']))['content'])
                       if row['dataset'] == 'gmdcsa24' else request(row['download_url']))
        except (OSError, http.client.HTTPException):
            payload = base64.b64decode(json.loads(request(row['blob_url']))['content'])
        if len(payload) != row['size'] or hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest() != row['git_blob_sha1']:
            raise ValueError('Publisher file changed: ' + row['id'])
        row['sha256'] = hashlib.sha256(payload).hexdigest()
        temporary = destination / (row['id'] + '.mp4')
        temporary.write_bytes(payload)
        capture = cv2.VideoCapture(str(temporary))
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count < 3:
            raise ValueError('Invalid public video')
        row['duration'] = count / fps
        row['fps'] = fps
        frames = []
        for seconds in np.arange(0, row['duration'], .5):
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (200, 112))
            cv2.putText(frame, f'{row["id"]} {seconds:.1f}s', (4, 17), 0, .5, (0, 255, 255), 1)
            frames.append(frame)
        capture.release()
        while len(frames) % 8:
            frames.append(np.zeros((112, 200, 3), np.uint8))
        sheet = np.vstack([np.hstack(frames[index:index + 8]) for index in range(0, len(frames), 8)])
        cv2.imwrite(str(destination / (row['id'] + '.jpg')), sheet)
        if not temporary.resolve().is_relative_to(destination):
            raise ValueError('Unexpected temporary path')
        temporary.unlink()
        saved.write_text(json.dumps(row, indent=2), encoding='utf-8')
        print(row['id'], row['purpose'], round(row['duration'], 2), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        prepared = list(pool.map(process, rows))
    (destination / 'pool.json').write_text(json.dumps(prepared, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    prepare(parser.parse_args().destination)
