"""Build the pinned Colab notebook from the verified public-data bootstrap."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "colab/mmaction_general_fall_finetune.ipynb"
TARGET = ROOT / "colab/mmaction_action_transition_v2.ipynb"


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(revision):
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ValueError("Use a full, immutable Git commit hash")
    previous = json.loads(SOURCE.read_text(encoding="utf-8"))
    setup = """from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, urllib.request, zipfile
from google.colab import drive

RUN_TAG = 'action_transition_v2_01'
EPOCHS = 40
TRAIN_MINUTES = 90
assert RUN_TAG and all(c.isalnum() or c in '_-' for c in RUN_TAG)
subprocess.run(['nvidia-smi'], check=True)
drive.mount('/content/drive', timeout_ms=180000)
DRIVE = Path('/content/drive/MyDrive/SENTINEL')
RUN = DRIVE / 'artifacts/mmaction_action_transition_v2' / RUN_TAG
RUN.mkdir(parents=True, exist_ok=True)
WORK = Path('/content/action_transition_v2'); WORK.mkdir(exist_ok=True)
CODE = WORK / 'code'; CODE.mkdir(exist_ok=True)
DEMO = CODE / 'mmaction2_demo'
WEIGHTS = WORK / 'weights'; WEIGHTS.mkdir(exist_ok=True)
CACHE = RUN / 'feature_cache'
print('Output on Drive:', RUN)
"""
    source = f"""# Immutable project source; reruns use the same trainer and preprocessing.
SOURCE_REVISION = '{revision}'
source_zip = WORK / 'source.zip'
revision_marker = WORK / 'source_revision.txt'
if (DEMO/'train_action_transition_v2.py').is_file() and (
        not revision_marker.is_file() or revision_marker.read_text().strip() != SOURCE_REVISION):
    raise RuntimeError('Colab has code from another revision; restart runtime before rerunning')
if not (DEMO / 'train_action_transition_v2.py').is_file():
    url = 'https://codeload.github.com/TranDuon/anomaly-behavior-detector-system/zip/' + SOURCE_REVISION
    with urllib.request.urlopen(url, timeout=180) as response, source_zip.open('wb') as target:
        shutil.copyfileobj(response, target)
    with zipfile.ZipFile(source_zip) as package:
        for item in package.infolist():
            target = (WORK / item.filename).resolve()
            assert target.is_relative_to(WORK.resolve()), item.filename
        package.extractall(WORK)
    source = WORK / ('anomaly-behavior-detector-system-' + SOURCE_REVISION)
    shutil.copytree(source, CODE, dirs_exist_ok=True)
    revision_marker.write_text(SOURCE_REVISION)
manifest_path = CODE / 'benchmarks/gmdcsa24_v2_1_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
assert manifest['publisher_tree_sha'] == '5abac7693229900cf80f722e878fbb119211fc1c'
print('Pinned source:', SOURCE_REVISION)
"""
    old_parent = "".join(previous["cells"][3]["source"])
    helper = old_parent[:old_parent.index("parent_candidates =")]
    parent = helper + """
PARENT_SHA = 'ddb09180284c4093189c53b7decbfcc5b51fd2d9b7700536d104524bceb5c344'
PARENT_FOLDER_ID = '1CPnlYVWz_xJLWqyQbM6oULojAo0vYTDw'
parent_zip = DRIVE / 'artifacts/mmaction_interaction_fall_wrestle/fall_wrestle_pilot_01/fall_wrestle_parent_verified.zip'
if not parent_zip.is_file():
    parent_zip = shared_drive_file(PARENT_FOLDER_ID, 'fall_wrestle_parent_verified.zip',
                                   WORK/'fall_wrestle_parent_verified.zip')
with zipfile.ZipFile(parent_zip) as package:
    assert {'interaction.pt', 'interaction.json'} <= set(package.namelist())
    metadata = json.loads(package.read('interaction.json'))
    payload = package.read('interaction.pt')
    assert hashlib.sha256(payload).hexdigest() == metadata['checkpoint_sha256'] == PARENT_SHA
    (WEIGHTS/'interaction_fall_wrestle.pt').write_bytes(payload)
    (WEIGHTS/'interaction_fall_wrestle.json').write_bytes(package.read('interaction.json'))
print('Verified parent:', PARENT_SHA)
"""
    review = """# Prefer your reviewed ZIP in MyDrive; otherwise use the shared test-only pack.
review_zip = DRIVE / 'artifacts/action_review_v1/action_review_v1.zip'
review_root = WORK / 'review'
review_args = []
if not review_zip.is_file():
    review_zip = shared_drive_file('1RDprs7P9ag0Hm_ZMPmIdfbTP87OldfEk',
                                   'action_review_v1.zip', WORK/'action_review_v1.zip')
if review_zip.is_file():
    review_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(review_zip) as package:
        for item in package.infolist():
            assert (review_root/item.filename).resolve().is_relative_to(review_root.resolve())
        package.extractall(review_root)
    review_manifest = review_root/'review.json'
    assert review_manifest.is_file()
    review_args = ['--review-manifest', review_manifest, '--review-root', review_root]
    print('Reviewed intervals loaded:', review_manifest)
else:
    print('No reviewed pack: public early-fall experiment only; no post-fight training claims.')
"""
    train = """base = [PY, DEMO/'train_action_transition_v2.py', '--manifest', manifest_path,
        '--fall-root', FALL_ROOT, '--scfd-root', SCFD_ROOT,
        '--weights', WEIGHTS, '--cache', CACHE, '--output', RUN,
        '--device', 'cuda:0', '--epochs', EPOCHS, '--minutes', TRAIN_MINUTES] + review_args
run(base + ['--stage', 'features'], 'extract_features', timeout=8*3600)
run(base + ['--stage', 'train'], 'fine_tune', timeout=(TRAIN_MINUTES+10)*60)
"""
    result = """report_path = RUN/'report.json'
assert report_path.is_file(), 'Training did not finish; inspect logs on Drive'
result = json.loads(report_path.read_text(encoding='utf-8'))
assert result['status'] == 'completed'
package = RUN/'mmaction_action_transition_v2_pilot.zip'
assert package.is_file()
print('PACKAGE:', package)
print('REPORT:', report_path)
print('Parent vs new test with detected people:', json.dumps(result['test_with_person'], indent=2))
print('Reviewed test actions:', json.dumps(result['test_reviewed_by_action'], indent=2))
print('This is a pilot. Install only after checking held-out temporal metrics.')
"""
    title = """# MMAction: phát hiện ngã sớm và kiểm tra chuyển trạng thái

Notebook này warm-start từ checkpoint `fall_wrestle_pilot_01`, giữ nguyên trọng số AVA và bộ phát hiện người. Chọn **Colab GPU** rồi chạy các cell theo thứ tự. Dữ liệu công khai cho phép thử cải thiện ngã ngay. Gói review chia sẻ hiện có một video vật thể thao giữ riêng để test; hai video còn lại chưa có mốc nhãn. Để dạy model ngừng báo đỏ khi xô xát kết thúc, gắn nhãn các đoạn `after_fight` trong gói review và đưa ZIP vào `MyDrive/SENTINEL/artifacts/action_review_v1/action_review_v1.zip`, sau đó đổi `RUN_TAG`.

Checkpoint mới chỉ là pilot. Báo cáo so sánh checkpoint cha và mới trên test giữ riêng; không tự cài vào ứng dụng. Cache đặc trưng và trạng thái train lưu trên Drive để chạy tiếp. Không dùng video vật thể thao làm nhãn xô xát dương.
"""
    cells = [cell("markdown", title), cell("code", setup), cell("code", source),
             cell("code", parent), previous["cells"][4], previous["cells"][5],
             previous["cells"][6], cell("code", review), cell("code", train),
             cell("code", result)]
    for index, item in enumerate(cells):
        item["id"] = f"action-step-{index:02d}"
    notebook = {"cells": cells, "metadata": previous["metadata"],
                "nbformat": 4, "nbformat_minor": 5}
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    return TARGET


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    print(build(parser.parse_args().revision))
