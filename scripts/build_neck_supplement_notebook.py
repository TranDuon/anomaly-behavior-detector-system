"""Build the resumable Colab notebook for the additive neck pilot."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "colab/mmaction_release_neck_v4.ipynb"
TARGET = ROOT / "colab/mmaction_neck_supplement_v5.ipynb"


def set_source(cell, value):
    cell["source"] = value.splitlines(keepends=True)


def build(revision):
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("A pinned 40-character Git revision is required")
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    set_source(cells[0], """# MMAction2 neck supplement v5 · Colab GPU

Train one **neck assault** score from AIRTLab choke clips and hard negatives. The
original `mmaction_fall_wrestle` checkpoint remains frozen. The final alert is
`parent alert OR neck alert`, so the new head cannot turn off an existing fight
alert. The notebook measures fight recall against the parent on held-out SCFD
clips and saves all states to Drive after each epoch.

The pilot is trained from a small neck set. It is **not approved for camera use**
until additional neck examples, including CHAD clips reviewed by a human, and
real camera tests pass. This notebook creates a new ZIP; it does not install it.
""")
    set_source(cells[1], """from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, urllib.request, zipfile
from google.colab import drive, files

RUN_TAG = 'neck_supplement_v5_01'
EPOCHS = 40
TRAIN_MINUTES = 90
try:
    drive.mount('/content/drive', timeout_ms=180000)
except Exception as exc:
    raise RuntimeError('Drive mount failed; authorize Google Drive and rerun cell 1') from exc
my_drive_candidates = [Path('/content/drive/MyDrive'), Path('/content/drive/My Drive')]
mounted_roots = [root for root in my_drive_candidates if root.is_dir()]
if not mounted_roots:
    raise RuntimeError('MyDrive is not visible under /content/drive')
RUN = None
write_errors = []
for root in mounted_roots:
    candidate = root / 'SENTINEL/artifacts/mmaction_neck_supplement_v5' / RUN_TAG
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate/'colab_write_probe.txt'
        probe.write_text(RUN_TAG+': Drive write preflight OK', encoding='utf-8')
        if not probe.read_text(encoding='utf-8').endswith('Drive write preflight OK'):
            raise OSError('Drive write probe did not round-trip')
        RUN = candidate
        break
    except OSError as exc:
        write_errors.append(str(candidate)+': '+str(exc))
if RUN is None:
    raise RuntimeError('No writable Drive path: '+ '; '.join(write_errors))
subprocess.run(['nvidia-smi'], check=True)
WORK = Path('/content/neck_supplement_v5'); WORK.mkdir(exist_ok=True)
CODE = WORK/'code'; CODE.mkdir(exist_ok=True)
DEMO = CODE/'mmaction2_demo'
WEIGHTS = WORK/'weights'; WEIGHTS.mkdir(exist_ok=True)
CACHE = RUN/'feature_cache'
print('Run directory:', RUN)
""")
    source = "".join(cells[2]["source"])
    source = source.replace("e9ba6e83254fa8c9d8101ce1f688ab62de29da55", revision)
    source = source.replace("train_release_neck_v4.py", "train_neck_supplement_v5.py")
    set_source(cells[2], source)
    set_source(cells[8], """base = [PY, DEMO/'train_neck_supplement_v5.py', '--manifest', manifest_path,
        '--fall-root', FALL_ROOT, '--scfd-root', SCFD_ROOT,
        '--airtlab-root', AIRTLAB_ROOT,
        '--review-manifest', review_manifest, '--review-root', review_root,
        '--weights', WEIGHTS, '--cache', CACHE, '--output', RUN,
        '--device', 'cuda:0', '--epochs', EPOCHS, '--minutes', TRAIN_MINUTES]
run(base + ['--stage', 'features'], 'extract_features', timeout=8*3600)
run(base + ['--stage', 'train'], 'train_neck_supplement', timeout=(TRAIN_MINUTES+10)*60)
""")
    set_source(cells[9], """report_path = RUN/'report.json'
assert report_path.is_file(), 'Training did not finish; inspect logs in '+str(RUN)
result = json.loads(report_path.read_text(encoding='utf-8'))
assert result['status'] == 'completed'
assert result['test']['fight_recall_nonregression']
package = RUN/'mmaction_neck_supplement_v5_pilot.zip'
assert package.is_file() and package.stat().st_size > 1000000
with zipfile.ZipFile(package) as saved:
    assert {'neck_supplement.pt','neck_supplement.json','report.json','protocol.json'} == set(saved.namelist())
    assert saved.testzip() is None, 'Pilot ZIP is damaged'
print('Held-out test:', json.dumps(result['test'], indent=2))
print('Pilot ZIP SHA-256:', sha256(package))
print('Flushing files to Google Drive...')
drive.flush_and_unmount(timeout_ms=600000)
DRIVE_FLUSHED = True
print('Saved and flushed to mounted Drive path:', RUN)
""")
    notebook["metadata"].setdefault("colab", {})["name"] = TARGET.name
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return TARGET


if __name__ == "__main__":
    print(build(sys.argv[1]))
