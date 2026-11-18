"""Build the pinned Colab notebook for the handgun hand-crop classifier (extensions/handgun).

    python scripts/build_handgun_notebook.py <source_commit_sha>

Same pipeline as the sharp-object hand_04 notebook (scripts/build_sharp_hand_notebook.py), run with
WEAPON_CLASS_MAP=datasets/handgun/class_map.json so every data script reads the handgun classes:
public pool (Granada pistols + Sohas look-alikes, Zenodo 'gun', Open Images 'Handgun', Simuletic ATM,
COCO / Charades people holding everyday things) -> optional self-filmed toy-gun frames -> pose hand
crops -> ResNet50 from ImageNet -> threshold -> handgun_<RUN_TAG>.zip for install_handgun.py.
The commit must already be pushed to origin (the notebook downloads that exact revision).
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "colab" / "handgun_hand_v1.ipynb"
_spec = importlib.util.spec_from_file_location("sharp_notebook", HERE / "build_sharp_hand_notebook.py")
sharp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sharp)


def cell(kind, value, index):
    made = sharp.cell(kind, value, index)
    made["id"] = f"handgun-{index:02d}"
    return made


def replace_all(text, pairs):
    for old, new in pairs:
        if old not in text:
            raise ValueError(f"template changed, cannot find: {old!r}")
        text = text.replace(old, new)
    return text


INTRO = """
# Người cầm súng ngắn · handgun hand_01: pose → crop quanh tay → ResNet

Cùng cách làm với nhánh vật sắc nhọn (hand_04): YOLO **pose** tìm người và khuỷu tay / cổ tay, cắt ô vuông quanh
**mỗi bàn tay**, rồi **ResNet50** phân loại ô đó: tay *có* hay *không* cầm súng ngắn. Súng ngắn nhỏ và nằm gọn trong
lòng bàn tay, nên phóng to vùng tay lên 224 px giúp thấy rõ hơn hẳn so với detector chạy trên cả khung hình.

**Dữ liệu công khai** (tất cả qua `extensions/sharp_object/collect.py` với `WEAPON_CLASS_MAP=datasets/handgun/class_map.json`):

| Nguồn | Dương (súng) | Âm |
|---|---|---|
| Granada *Pistol detection* | 3000 ảnh, nhãn `pistol` | |
| Granada *Sohas* (tay cầm đồ vật) | `pistol` | tay cầm dao, điện thoại, ví, tiền, thẻ |
| Zenodo Dangerous Items | `gun` | ảnh chỉ có dao / mã tấu / gậy (ảnh có `rifle` bị bỏ) |
| Open Images | `Handgun` | điện thoại, remote, máy sấy, chuột, dập ghim, máy ảnh, đèn pin, máy tính bỏ túi |
| Simuletic ATM (giả lập CCTV) | `gun` | cảnh cướp bằng dao |
| COCO `coco_hold`, Charades `charades_neg` | | người cầm điện thoại / cốc / sách / chổi… trong nhà |

**Dữ liệu tự quay (không bắt buộc):** zip CVAT *YOLO 1.1* (hoặc thư mục Ultralytics) có nhãn `handgun`, đặt trong
`MyDrive/SENTINEL/datasets/handgun_self/labelled/`. Clip mp4 tự quay (mọi clip đều có người cầm súng) đặt trong
`handgun_self/clips/` để đo độ ổn định ở bước 6. Chỉ dùng súng đồ chơi / mô hình hợp pháp.

**Cách chạy:**
1. *Runtime → Change runtime type*: **L4** là đủ.
2. *Runtime → Run all*, cho phép mount Google Drive.
3. Phiên **tự ngắt** khi xong hoặc khi lỗi (traceback ghi ra `ERROR.txt` trên Drive).

| Bước | Việc làm | Thời gian ước tính (L4) |
|---|---|---|
| 1 | Pool công khai: lần đầu tải (~1.5 GB Zenodo + Granada + Open Images…) rồi cache `pool_v1.tar` lên Drive | 40–80 phút lần đầu, 2–5 phút các lần sau |
| 2 | Ghép dữ liệu tự quay (nếu có), chia train/val/test theo nhóm nguồn | 2–5 phút |
| 3 | Pose → crop quanh tay → gán nhãn từ box súng, ảnh xem trước | 15–40 phút |
| 4 | Fine-tune ResNet50 (ImageNet), cân bằng lớp, dừng sớm theo AP val | 30–90 phút |
| 5 | Đánh giá, chọn ngưỡng (báo nhầm ≤ 3% trên ảnh không có súng của val), đóng gói `handgun_<RUN_TAG>.zip` | ~5 phút |
| 6 | Độ ổn định trên clip tự quay (nếu có) và báo nhầm trên clip Charades test | 5–15 phút |

**Kết quả trên Drive:** `MyDrive/SENTINEL/artifacts/handgun/<RUN_TAG>/`. Cài trên máy bằng
`extensions\\env\\Scripts\\python.exe -m extensions.handgun.install_handgun handgun_<RUN_TAG>.zip`.

**Lưu ý:** ảnh cận cảnh (Sohas, Granada) thường không thấy cả người nên pose không tìm được tay; ở tập train các ảnh
đó cho crop "giả" quanh box súng (trọng số ×0.5), val/test chỉ dùng crop quanh tay thật.
`hand_coverage_of_weapon_images` ở bước 3 cho biết tỉ lệ ảnh có súng mà pose tìm được bàn tay cầm nó.
"""

SETUP = replace_all(sharp.SETUP, [
    ("VN_WEIGHT = 6                 # trọng số lấy mẫu crop VN",
     "VN_WEIGHT = 4                 # trọng số lấy mẫu crop tự quay (nhóm 'vn' của build_dataset)"),
    ("PUBLIC_POOL_VERSION = 'v1'    # dùng lại pool công khai sharp_02 đã cache trên Drive",
     "PUBLIC_POOL_VERSION = 'v1'    # pool công khai súng ngắn, cache trên Drive sau lần đầu"),
    ("SOURCE_REVISION = '{revision}'",
     "SOURCE_REVISION = '{revision}'\n"
     "os.environ['WEAPON_CLASS_MAP'] = 'datasets/handgun/class_map.json'  # mọi script dữ liệu đọc lớp súng"),
    ("VN_DIR = SENTINEL / 'datasets' / 'sharp_object_vn'", "VN_DIR = SENTINEL / 'datasets' / 'handgun_self'"),
    ("PUBLIC_CACHE = SENTINEL / 'datasets' / 'sharp_object_public'",
     "PUBLIC_CACHE = SENTINEL / 'datasets' / 'handgun_public'"),
    ("OUT = SENTINEL / 'artifacts' / 'sharp_object' / RUN_TAG", "OUT = SENTINEL / 'artifacts' / 'handgun' / RUN_TAG"),
    ("(VN_DIR / 'labelled').mkdir(parents=True, exist_ok=True)",
     "(VN_DIR / 'labelled').mkdir(parents=True, exist_ok=True)\n(VN_DIR / 'clips').mkdir(parents=True, exist_ok=True)"),
    ("WORK = Path('/content/sharp_hand_work')", "WORK = Path('/content/handgun_work')"),
    ("glob('sharp_object_*.zip')", "glob('handgun_*.zip')"),
    ("print('Dữ liệu VN đọc từ:', VN_DIR)", "print('Dữ liệu tự quay đọc từ:', VN_DIR)"),
])

POOL_MD = """
## Bước 1 · Pool dữ liệu công khai

Nếu `handgun_public/pool_v1.tar` đã có trên Drive thì chỉ giải nén. Chưa có thì tải từng nguồn (nguồn nào lỗi thì bỏ
qua và in cảnh báo) rồi cache lại lên Drive cho các lần sau.
"""

POOL_CODE = """
if PUBLIC_CACHE.is_file() and not REFRESH_PUBLIC_POOL:
    print('Dùng pool công khai đã cache:', PUBLIC_CACHE, '(%.2f GB)' % (PUBLIC_CACHE.stat().st_size / 1e9))
    shutil.rmtree(POOL, ignore_errors=True)
    with tarfile.open(PUBLIC_CACHE) as bundle:
        bundle.extractall(WORK)
else:
    failed = []
    for source, extra in POOL_SOURCES:
        try:
            py('extensions.sharp_object.collect', 'labeled', '--root', POOL, '--work', SCRATCH,
               '--sources', source, *extra, log='collect.log')
        except RuntimeError as error:
            failed.append(source)
            print('CẢNH BÁO: bỏ qua nguồn', source, '-', error)
        shutil.rmtree(SCRATCH, ignore_errors=True)
        SCRATCH.mkdir(parents=True, exist_ok=True)
    assert (POOL / 'manifest.csv').is_file(), 'Không tải được nguồn công khai nào'
    PUBLIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    partial = PUBLIC_CACHE.with_suffix('.partial')
    with tarfile.open(partial, 'w') as bundle:
        bundle.add(POOL, arcname='pool')
    partial.replace(PUBLIC_CACHE)
    print('Đã cache pool lên Drive:', PUBLIC_CACHE, '| nguồn lỗi:', failed or 'không')
print(json.dumps(json.loads((POOL / 'stats.json').read_text()), indent=1, ensure_ascii=False))
"""

SELF_MD = """
## Bước 2 · Ghép dữ liệu tự quay và chia tập

Mọi zip / thư mục trong `handgun_self/labelled/` (cũ trước, mới sau). Không có thì chỉ dùng dữ liệu công khai.
Nhóm tự quay được chia val/test riêng (`crop_vn_*`, `alert_vn_*` ở bước 5).
"""

SELF_CODE = """
inputs = sorted((p for p in (VN_DIR / 'labelled').iterdir() if p.suffix.lower() == '.zip' or p.is_dir()),
                key=lambda p: p.stat().st_mtime)
print('Đầu vào tự quay:', [p.name for p in inputs] or 'không có')
local_inputs = []
for index, path in enumerate(inputs):
    target = WORK / 'vn_inputs' / f'{index:02d}_{path.name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, target, dirs_exist_ok=True)
    else:
        shutil.copy(path, target)
    local_inputs.append(target)
py('extensions.sharp_object.build_dataset', '--pool', POOL, *(['--vn', *local_inputs] if local_inputs else []),
   '--out', DATA, log='build_dataset.log')
shutil.copy(DATA / 'summary.json', OUT / 'dataset_summary.json')
summary = json.loads((DATA / 'summary.json').read_text())
for split in ('train', 'val', 'test'):
    print(split, '| ảnh:', summary['images'][split], '| tự quay:', summary[split]['vn_images'],
          '| ảnh âm:', summary[split]['negatives'])
for warning in summary['warnings']:
    print('CẢNH BÁO:', warning)
"""

CROPS_MD = replace_all(sharp.CROPS_MD, [("tỉ lệ ảnh có vũ khí", "tỉ lệ ảnh có súng")])

TRAIN_MD = """
## Bước 4 · Fine-tune ResNet

- ResNet `ARCH` pre-train ImageNet, một logit "tay cầm súng ngắn"; AdamW, cosine, AMP; tăng cường crop/scale, lật,
  xoay, màu, mờ.
- Lấy mẫu cân bằng hai lớp; crop tự quay ×`VN_WEIGHT`, crop giả ×0.5, crop âm COCO / Charades ×`HARD_NEGATIVE_WEIGHT`.
- Dừng sớm sau 6 epoch không tăng AP trên crop tay thật của val. Bật `SYNC_DURING_TRAIN` để train tiếp được nếu bị ngắt.
"""

TRAIN_CODE = sharp.TRAIN_CODE

EVAL_MD = """
## Bước 5 · Đánh giá, chọn ngưỡng, đóng gói

- AP / AUC trên crop tay thật của val và test (toàn bộ và phần tự quay).
- Cảnh báo theo ảnh: ảnh cảnh báo khi một crop tay ≥ ngưỡng. Ngưỡng thấp nhất sao cho báo nhầm trên ảnh **không có súng**
  của val ≤ 3%. `alert_hard_negative_*`: báo nhầm trên ảnh COCO / Charades người cầm đồ vật thường.
"""

EVAL_CODE = replace_all(sharp.EVAL_CODE, [
    ("'--run-tag', RUN_TAG, log='export.log')", "'--run-tag', RUN_TAG, '--name', 'handgun', log='export.log')"),
    ("print('Bước tiếp theo: tải sharp_object_' + RUN_TAG + '.zip về máy, cài bằng install_sharp_object.py rồi thử trên camera.')",
     "print('Bước tiếp theo: tải handgun_' + RUN_TAG + '.zip về máy, cài bằng extensions.handgun.install_handgun rồi thử trên camera.')"),
])

CLIPS_MD = """
## Bước 6 · Clip: độ ổn định (tự quay) và báo nhầm (Charades test)

- Clip trong `handgun_self/clips/` (mọi clip có người cầm súng): `alert_share` càng cao càng tốt, `drops` (điểm tụt dưới
  ngưỡng ngay sau một bước trên ngưỡng) càng ít càng tốt.
- `NEG_CLIPS` clip Charades **test** (người cầm chổi / điện thoại / cốc… trong nhà, không dùng train): `alert_share` càng
  thấp càng tốt.
"""

CLIPS_CODE = """
clips_dir = VN_DIR / 'clips'
package = LOCAL_OUT / 'train' / 'package'
if any(clips_dir.glob('*.mp4')):
    py('extensions.sharp_object.eval_hand_clips', '--package', package, '--clips', clips_dir,
       '--out', LOCAL_OUT / 'train' / 'clips_eval.json', log='clips_eval.log')
    shutil.copy(LOCAL_OUT / 'train' / 'clips_eval.json', OUT / 'clips_eval.json')
    own = json.loads((LOCAL_OUT / 'train' / 'clips_eval.json').read_text())
    print('CLIP TỰ QUAY (càng cao càng tốt) - tỉ lệ bước vượt ngưỡng:', {k: v['alert_share'] for k, v in own['overall'].items()})
else:
    print('Không có clip tự quay trong', clips_dir)
if NEG_CLIPS:
    neg_dir = WORK / 'neg_clips'
    shutil.rmtree(neg_dir, ignore_errors=True)
    py('extensions.sharp_object.collect', 'charades_test', '--root', neg_dir, '--work', SCRATCH, '--limit', NEG_CLIPS,
       log='neg_clips.log')
    py('extensions.sharp_object.eval_hand_clips', '--package', package, '--clips', neg_dir,
       '--out', LOCAL_OUT / 'train' / 'clips_neg_eval.json', log='clips_eval.log')
    shutil.copy(LOCAL_OUT / 'train' / 'clips_neg_eval.json', OUT / 'clips_neg_eval.json')
    neg = json.loads((LOCAL_OUT / 'train' / 'clips_neg_eval.json').read_text())
    print('BẢN ÂM (càng thấp càng tốt) - tỉ lệ bước vượt ngưỡng:', {k: v['alert_share'] for k, v in neg['overall'].items()})
sync_to_drive()
"""

VARIANT = """RUN_TAG = 'hand_01'
INIT_FROM = None              # train từ ImageNet
LR = 1e-4
HARD_NEGATIVE_WEIGHT = 3.0    # crop âm COCO / Charades (người cầm đồ vật thường)
NEG_CLIPS = 24                # clip Charades test để đo báo nhầm theo thời gian (0 = bỏ qua)
POOL_SOURCES = [              # (nguồn, tham số thêm) cho collect.py; chỉ chạy khi chưa có cache trên Drive
    ('granada', []),
    ('zenodo', []),
    ('simuletic', []),
    ('openimages', ['--oi-per-class', 2000, '--oi-negatives-per-class', 300]),
    ('coco_hold', ['--coco-neg-limit', 3000]),
    ('charades_neg', ['--charades-videos', 600]),
]"""


def build(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("give the full 40-character commit SHA that is pushed to origin")
    setup = SETUP.replace("{revision}", revision).replace("{variant}", VARIANT)
    setup = setup.replace("{{", "{").replace("}}", "}")  # same template escaping as the sharp builder
    cells = [("markdown", INTRO), ("code", setup), ("code", sharp.SOURCE), ("code", sharp.INSTALL),
             ("markdown", POOL_MD), ("code", POOL_CODE), ("markdown", SELF_MD), ("code", SELF_CODE),
             ("markdown", CROPS_MD), ("code", sharp.CROPS_CODE), ("markdown", TRAIN_MD), ("code", TRAIN_CODE),
             ("markdown", EVAL_MD), ("code", EVAL_CODE), ("markdown", CLIPS_MD), ("code", CLIPS_CODE),
             ("markdown", "## Tự ngắt phiên\n\nKết quả đã nằm trên Drive nên ngắt phiên để không tốn compute unit.\n"),
             ("code", "disconnect('XONG. Kết quả ở ' + str(OUT))\n")]
    notebook = {
        "cells": [cell(kind, value, index) for index, (kind, value) in enumerate(cells)],
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print("wrote", build(sys.argv[1]))
