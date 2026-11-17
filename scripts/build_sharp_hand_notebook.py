"""Build the pinned Colab notebook for the hand-crop sharp-object classifier.

    python scripts/build_sharp_hand_notebook.py <source_commit_sha> [hand_01|hand_02|hand_03]

hand_01: first run from ImageNet. hand_02: fine-tune from hand_01 with the newest reviewed VN zip
(self-filmed frames forced into train) and a clip stability check against hand_01.
hand_03: fine-tune from hand_02 with COCO hard negatives (people holding bats / rackets / umbrellas /
skis; pool v2 = v1 + coco_neg) and any vn_negatives_*.zip, compared with hand_02.
"""

import json
import sys
from pathlib import Path
from textwrap import dedent

COLAB = Path(__file__).resolve().parents[1] / "colab"
OUTPUTS = {"hand_01": COLAB / "sharp_object_hand_v1.ipynb", "hand_02": COLAB / "sharp_object_hand_v2.ipynb",
           "hand_03": COLAB / "sharp_object_hand_v3.ipynb"}


def cell(kind, value, index):
    return {
        "cell_type": kind,
        "metadata": {},
        "source": dedent(value).lstrip("\n").splitlines(keepends=True),
        "id": f"sharp-hand-{index:02d}",
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


INTRO = """
# Người cầm vật sắc nhọn · hand_01: pose → crop quanh tay → ResNet

Hướng mới thay cho detector YOLO toàn khung: YOLO **pose** tìm người và vị trí khuỷu tay / cổ tay,
cắt một ô vuông quanh **mỗi bàn tay** (cạnh = 0.45 × chiều cao người, đẩy qua cổ tay theo hướng cẳng tay để
lưỡi/cán dao nằm trong ô), rồi **ResNet50** phân loại ô đó: tay *có* hay *không* cầm vật sắc nhọn.
Dao nhỏ trong khung hình lớn được "phóng to" ở 224 px, và chỉ dao **đang cầm** mới được xét.

**Tận dụng từ sharp_02:** pool dữ liệu công khai (đọc lại bản cache `pool_v1.tar` trên Drive, không tải lại),
dữ liệu VN đã duyệt, cách chia tập theo nhóm nguồn, quy tắc chọn ngưỡng (báo nhầm ≤ 3% trên ảnh không có vũ khí của val)
và EventGate ở runtime. Nhãn crop lấy từ box vũ khí có sẵn, không cần gán nhãn thêm.

**Cách chạy:**
1. *Runtime → Change runtime type*: **L4** là đủ (A100 nhanh hơn nhưng tốn unit hơn).
2. *Runtime → Run all*, cho phép mount Google Drive.
3. Phiên **tự ngắt** khi xong hoặc khi lỗi (traceback ghi ra `ERROR.txt` trên Drive).

**Các bước:**

| Bước | Việc làm | Thời gian ước tính (L4) |
|---|---|---|
| 1 | Pool công khai: giải nén bản cache trên Drive (lần đầu mới phải tải, 30–60 phút) | 2–5 phút |
| 2 | Ghép dữ liệu VN đã duyệt, chia train/val/test (như sharp_02) | 2–5 phút |
| 3 | Chạy pose trên mọi ảnh, cắt crop quanh tay, gán nhãn từ box vũ khí, in độ phủ và ảnh xem trước | 15–40 phút |
| 4 | Fine-tune ResNet50 (ImageNet), cân bằng lớp, crop VN ×6, tối đa `EPOCHS` epoch, dừng sớm theo AP val | 30–90 phút |
| 5 | Đánh giá AP crop và cảnh báo theo ảnh, chọn ngưỡng, đóng gói `sharp_object_<RUN_TAG>.zip` | ~5 phút |

**Nhãn crop:** dương nếu crop chạm box vũ khí đủ nhiều (≥ 25% box hoặc ≥ 25% crop), âm nếu không chạm box nào,
lưng chừng thì bỏ. Ảnh cận cảnh không thấy người: ở tập train cắt thêm crop "giả" quanh box vũ khí (trọng số ×0.5);
val/test chỉ dùng crop quanh tay thật nên điểm phản ánh đúng đường chạy runtime.

**Kết quả trên Drive:** `MyDrive/SENTINEL/artifacts/sharp_object/<RUN_TAG>/`. Zip gồm classifier (fp16), file pose,
spec (`"method": "hand_crop"`) và `evaluation.json`; cài bằng `install_sharp_object.py` như bản detector.

**Lưu ý:** val/test VN còn ít ảnh nên điểm phần VN chỉ để tham khảo. `hand_coverage_of_weapon_images` ở bước 3
cho biết bao nhiêu ảnh có vũ khí mà pose tìm được bàn tay đang cầm: con số thấp nghĩa là nhiều ảnh công khai là ảnh cận cảnh.
"""

SETUP = """
from pathlib import Path
import json, os, shutil, subprocess, sys, tarfile, time, traceback, urllib.request, zipfile
from google.colab import drive

ARCH = 'resnet50'             # resnet18 / resnet34 nhẹ hơn cho laptop yếu
POSE = 'yolo11s-pose.pt'      # model pose dùng để cắt crop và chạy ở runtime
EPOCHS = 25
BATCH = 96
WORKERS = 4                   # crop nhỏ (256 px) nên 4 worker không tốn nhiều RAM
VN_WEIGHT = 6                 # trọng số lấy mẫu crop VN
PUBLIC_POOL_VERSION = 'v1'    # dùng lại pool công khai sharp_02 đã cache trên Drive
REFRESH_PUBLIC_POOL = False
AUTO_DISCONNECT = True        # tự ngắt phiên khi xong hoặc khi lỗi
SYNC_DURING_TRAIN = False     # True: chép last.pt lên Drive mỗi SYNC_MINUTES phút để train tiếp nếu bị ngắt
SYNC_MINUTES = 5              # chu kỳ in RAM/GPU (và chép Drive nếu bật)
SOURCE_REVISION = '{revision}'
{variant}

assert RUN_TAG and all(c.isalnum() or c in '_-' for c in RUN_TAG)
subprocess.run(['nvidia-smi'], check=True)
drive.mount('/content/drive', timeout_ms=180000)
roots = [p for p in (Path('/content/drive/MyDrive'), Path('/content/drive/My Drive')) if p.is_dir()]
assert roots, 'Không thấy MyDrive sau khi mount Google Drive'
SENTINEL = roots[0] / 'SENTINEL'
VN_DIR = SENTINEL / 'datasets' / 'sharp_object_vn'
PUBLIC_CACHE = SENTINEL / 'datasets' / 'sharp_object_public' / f'pool_{{PUBLIC_POOL_VERSION}}.tar'
OUT = SENTINEL / 'artifacts' / 'sharp_object' / RUN_TAG
OUT.mkdir(parents=True, exist_ok=True)
(VN_DIR / 'labelled').mkdir(parents=True, exist_ok=True)
(OUT / 'drive_write_probe.txt').write_text('write OK', encoding='utf-8')
WORK = Path('/content/sharp_hand_work')
SRC, POOL, DATA, CROPS, SCRATCH = WORK / 'src', WORK / 'pool', WORK / 'dataset', WORK / 'crops', WORK / 'scratch'
LOCAL_OUT, LOGS = WORK / 'out', WORK / 'logs'
for path in (WORK, SCRATCH, LOCAL_OUT, LOGS):
    path.mkdir(parents=True, exist_ok=True)
STARTED = time.time()
SYNC_FILES = ('train/last.pt', 'train/best.pt', 'train/history.json', 'train/TRAIN_DONE', 'train/evaluation.json')


def sync_to_drive():
    \"\"\"Chép checkpoint, kết quả và log từ ổ phiên lên Drive (ghi file tạm rồi đổi tên).\"\"\"
    import threading
    with globals().setdefault('_SYNC_LOCK', threading.Lock()):
        items = [(LOCAL_OUT / rel, OUT / rel) for rel in SYNC_FILES]
        items += [(p, OUT / p.name) for p in LOGS.glob('*.log')]
        items += [(p, OUT / p.name) for p in (LOCAL_OUT / 'train').glob('sharp_object_*.zip')]
        for source, target in items:
            if not source.is_file():
                continue
            if target.is_file() and target.stat().st_size == source.stat().st_size \\
                    and target.stat().st_mtime >= source.stat().st_mtime:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.partial')
            shutil.copy2(source, partial)
            partial.replace(target)


def disconnect(reason):
    print(reason, flush=True)
    try:
        sync_to_drive()
    except Exception as error:
        print('Không chép được lên Drive:', error, flush=True)
    if AUTO_DISCONNECT:
        print('Phiên Colab sẽ tự ngắt sau 20 giây (đặt AUTO_DISCONNECT = False để giữ lại).', flush=True)
        time.sleep(20)
        drive.flush_and_unmount()
        from google.colab import runtime
        runtime.unassign()


def on_error(shell, etype, evalue, tb, tb_offset=None):
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
    text = ''.join(traceback.format_exception(etype, evalue, tb))
    (OUT / 'ERROR.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S ') + text, encoding='utf-8')
    disconnect(f'LỖI, xem {{OUT / "ERROR.txt"}} và log trên Drive.')


get_ipython().set_custom_exc((Exception,), on_error)
(OUT / 'ERROR.txt').unlink(missing_ok=True)
print('Kết quả sẽ lưu tại:', OUT)
print('Dữ liệu VN đọc từ:', VN_DIR)
"""

SOURCE = """
# Mã nguồn được ghim theo commit để kết quả tái lập được.
marker = SRC / '.revision'
if not marker.is_file() or marker.read_text().strip() != SOURCE_REVISION:
    shutil.rmtree(SRC, ignore_errors=True)
    archive = WORK / 'source.zip'
    url = 'https://codeload.github.com/TranDuon/anomaly-behavior-detector-system/zip/' + SOURCE_REVISION
    with urllib.request.urlopen(url, timeout=300) as response, archive.open('wb') as output:
        shutil.copyfileobj(response, output)
    with zipfile.ZipFile(archive) as package:
        top = package.namelist()[0].split('/')[0]
        package.extractall(WORK)
    (WORK / top).rename(SRC)
    marker.write_text(SOURCE_REVISION)
print('Mã nguồn:', SOURCE_REVISION)


def is_progress(line):
    return '%|' in line or ('it/s]' in line) or ('s/it]' in line)


def run(args, log_name, cwd=SRC):
    \"\"\"Chạy lệnh; log đầy đủ ghi ở ổ phiên, notebook chỉ hiện dòng tóm tắt.\"\"\"
    started = time.time()
    env = {**os.environ, 'TQDM_MININTERVAL': '30'}
    with (LOGS / log_name).open('a', encoding='utf-8') as log:
        log.write(f'\\n$ {" ".join(map(str, args))}\\n')
        process = subprocess.Popen(list(map(str, args)), cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in process.stdout:
            log.write(line)
            if not is_progress(line):
                print(line, end='')
        code = process.wait()
    print(f'--- {log_name}: exit {code}, {(time.time() - started) / 60:.1f} phút')
    if code:
        raise RuntimeError(f'{log_name} lỗi (exit {code}), xem {OUT / log_name}')


def py(module, *args, log):
    run([sys.executable, '-u', '-m', module, *args], log)
"""

INSTALL = """
run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.183', 'hf_xet'], 'install.log')
import torch, torchvision, ultralytics
assert torch.cuda.is_available(), 'Chọn GPU (L4/A100) trong Runtime → Change runtime type'
print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| torchvision', torchvision.__version__,
      '| ultralytics', ultralytics.__version__)
print('Ổ đĩa trống: %.0f GB' % (shutil.disk_usage('/content').free / 1e9))
"""

POOL_MD = """
## Bước 1 · Pool dữ liệu công khai (dùng lại cache của sharp_02)

Nếu `pool_v1.tar` đã có trên Drive thì chỉ giải nén. Chưa có thì tải như sharp_02 rồi cache lại.
"""

POOL_CODE = """
BASE_CACHE = PUBLIC_CACHE.with_name(f'pool_{BASE_POOL_VERSION}.tar') if BASE_POOL_VERSION else None
if PUBLIC_CACHE.is_file() and not REFRESH_PUBLIC_POOL:
    print('Dùng pool công khai đã cache:', PUBLIC_CACHE, '(%.2f GB)' % (PUBLIC_CACHE.stat().st_size / 1e9))
    shutil.rmtree(POOL, ignore_errors=True)
    with tarfile.open(PUBLIC_CACHE) as bundle:
        bundle.extractall(WORK)
elif BASE_CACHE is not None and BASE_CACHE.is_file() and not REFRESH_PUBLIC_POOL:
    print('Mở pool gốc', BASE_CACHE.name, 'rồi thêm', EXTRA_POOL_SOURCES)
    shutil.rmtree(POOL, ignore_errors=True)
    with tarfile.open(BASE_CACHE) as bundle:
        bundle.extractall(WORK)
    for source in EXTRA_POOL_SOURCES:
        py('extensions.sharp_object.collect', 'labeled', '--root', POOL, '--work', SCRATCH, '--sources', source,
           log='collect.log')
    partial = PUBLIC_CACHE.with_suffix('.partial')
    with tarfile.open(partial, 'w') as bundle:
        bundle.add(POOL, arcname='pool')
    partial.replace(PUBLIC_CACHE)
    print('Đã cache pool mới lên Drive:', PUBLIC_CACHE)
else:
    failed = []
    for source, extra in (('zenodo', []), ('granada', []), ('simuletic', []),
                          ('coco', ['--coco-limit', 800]),
                          ('openimages', ['--oi-per-class', 1500, '--oi-negatives-per-class', 300]),
                          *((name, []) for name in EXTRA_POOL_SOURCES)):
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

VN_MD = """
## Bước 2 · Ghép dữ liệu VN đã duyệt và chia tập (giống sharp_02)

1. `vn_weapons_all_260_training_inputs_009.zip` (phần `supervised/`)
2. mọi zip và thư mục trong `labelled/`, mới nhất cuối cùng
"""

VN_CODE = """
vn_inputs = [VN_DIR / 'vn_weapons_all_260_training_inputs_009.zip']
vn_inputs = [p for p in vn_inputs if p.exists()]
reviewed = []
if VN_REVIEWED_GLOB:
    reviewed = sorted(VN_DIR.glob(VN_REVIEWED_GLOB))
    assert reviewed, f'Không thấy {VN_REVIEWED_GLOB} trong {VN_DIR}'
    vn_inputs.append(reviewed[-1])  # bản duyệt mới nhất đã gồm các bản trước
if VN_NEGATIVES_GLOB:
    vn_inputs += sorted(VN_DIR.glob(VN_NEGATIVES_GLOB))  # make_negative_set.py: khung không có vũ khí
vn_inputs += sorted((p for p in (VN_DIR / 'labelled').iterdir() if p.suffix.lower() == '.zip' or p.is_dir()),
                    key=lambda p: p.stat().st_mtime)
print('Đầu vào VN:')
for path in vn_inputs:
    print('  ', path.relative_to(VN_DIR))
local_inputs = []
for index, path in enumerate(vn_inputs):
    target = WORK / 'vn_inputs' / f'{index:02d}_{path.name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, target, dirs_exist_ok=True)
    else:
        shutil.copy(path, target)
    local_inputs.append(target)
py('extensions.sharp_object.build_dataset', '--pool', POOL, '--vn', *local_inputs, '--out', DATA,
   *(['--vn-train-groups', *VN_TRAIN_GROUPS] if VN_TRAIN_GROUPS else []), log='build_dataset.log')
shutil.copy(DATA / 'summary.json', OUT / 'dataset_summary.json')
summary = json.loads((DATA / 'summary.json').read_text())
for split in ('train', 'val', 'test'):
    print(split, '| ảnh:', summary['images'][split], '| VN:', summary[split]['vn_images'],
          '| ảnh âm:', summary[split]['negatives'])
for warning in summary['warnings']:
    print('CẢNH BÁO:', warning)
"""

CROPS_MD = """
## Bước 3 · Pose → crop quanh tay → gán nhãn

- Pose `POSE` chạy ở 960 px (lớn hơn runtime để bắt được người nhỏ trong ảnh công khai), tối đa 8 người mỗi ảnh.
- Mỗi bàn tay thấy được (độ tin cổ tay ≥ 0.3) cho một crop, lưu 256 px.
- `hand_coverage_of_weapon_images`: tỉ lệ ảnh có vũ khí mà pose tìm được bàn tay cầm nó.
- Ảnh xem trước: hàng trên (vạch đỏ) là crop dương, hàng dưới (vạch xanh) là crop âm. Nên nhìn qua để chắc crop đúng chỗ.
"""

CROPS_CODE = """
from IPython.display import Image, display

py('extensions.sharp_object.build_hand_crops', '--data', DATA, '--out', CROPS, '--pose', POSE, '--batch', 32,
   log='hand_crops.log')
crops_summary = json.loads((CROPS / 'summary.json').read_text())
shutil.copy(CROPS / 'summary.json', OUT / 'crops_summary.json')
for split, stat in crops_summary['splits'].items():
    print(split, {k: stat.get(k) for k in ('images', 'images_with_people', 'weapon_images',
                                           'hand_coverage_of_weapon_images', 'hand_pos', 'hand_neg',
                                           'pseudo_pos', 'pseudo_neg', 'ambiguous', 'vn_hand_pos')})
for split in ('train', 'val'):
    preview = CROPS / f'preview_{split}.jpg'
    if preview.is_file():
        shutil.copy(preview, OUT / preview.name)
        print('Xem trước', split)
        display(Image(filename=str(preview)))
"""

TRAIN_MD = """
## Bước 4 · Fine-tune ResNet

- ResNet `ARCH` pre-train ImageNet, một logit "tay cầm vật sắc nhọn"; AdamW (lr 1e-4 backbone, 1e-3 đầu ra), cosine,
  AMP; tăng cường: crop/scale, lật, xoay ±20°, màu, mờ.
- Lấy mẫu cân bằng hai lớp; crop VN ×`VN_WEIGHT`, crop giả ×0.5.
- Dừng sớm sau 6 epoch không tăng AP trên crop tay thật của val. `last.pt` ghi mỗi epoch ở ổ phiên;
  *Run all* lại sau khi bị ngắt sẽ train tiếp nếu `last.pt` có trên Drive (bật `SYNC_DURING_TRAIN`).
"""

TRAIN_CODE = """
import threading, torch

INIT_LOCAL = WORK / 'init.pt'
if INIT_FROM:
    assert INIT_FROM.is_file(), f'Không thấy checkpoint khởi đầu {INIT_FROM}'
    shutil.copy(INIT_FROM, INIT_LOCAL)
    print('Train tiếp từ:', INIT_FROM)
drive_last, local_last = OUT / 'train' / 'last.pt', LOCAL_OUT / 'train' / 'last.pt'
if drive_last.is_file() and not local_last.is_file():
    print('Chép run dở từ Drive về ổ phiên:', OUT / 'train')
    shutil.copytree(OUT / 'train', LOCAL_OUT / 'train', dirs_exist_ok=True)
stop_sync = threading.Event()


def sync_loop():
    while not stop_sync.wait(SYNC_MINUTES * 60):
        try:
            sync_to_drive()
        except Exception as error:
            print('Chép lên Drive lỗi (sẽ thử lại):', error, flush=True)


def monitor_loop():
    import psutil
    while not stop_sync.wait(SYNC_MINUTES * 60):
        ram = psutil.virtual_memory()
        gpu = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        print(f'[{time.strftime("%H:%M")}] RAM {ram.used / 1e9:.1f}/{ram.total / 1e9:.1f} GB ({ram.percent:.0f}%) | '
              f'ổ trống {shutil.disk_usage("/content").free / 1e9:.0f} GB | '
              f'GPU dùng {(gpu[1] - gpu[0]) / 1e9:.1f}/{gpu[1] / 1e9:.1f} GB', flush=True)


threading.Thread(target=monitor_loop, daemon=True).start()
if SYNC_DURING_TRAIN:
    threading.Thread(target=sync_loop, daemon=True).start()
common = ['--crops', CROPS, '--output', LOCAL_OUT / 'train', '--arch', ARCH, '--batch', BATCH, '--workers', WORKERS]
try:
    py('extensions.sharp_object.train_hand_classifier', '--stage', 'train', *common, '--epochs', EPOCHS,
       '--vn-weight', VN_WEIGHT, '--lr', LR, '--hard-negative-weight', HARD_NEGATIVE_WEIGHT, *(['--init', INIT_LOCAL] if INIT_FROM else []), log='train.log')
finally:
    stop_sync.set()
    sync_to_drive()
"""

EVAL_MD = """
## Bước 5 · Đánh giá, chọn ngưỡng, đóng gói

- AP / AUC trên crop tay thật của val và test (toàn bộ và phần VN).
- Cảnh báo theo ảnh: ảnh cảnh báo khi một crop tay ≥ ngưỡng. Ngưỡng thấp nhất sao cho báo nhầm trên ảnh
  **không có vũ khí** của val ≤ 3% (cùng quy tắc sharp_02). Ảnh có vũ khí nhưng pose không thấy tay cầm thì không tính.
"""

EVAL_CODE = """
py('extensions.sharp_object.train_hand_classifier', '--stage', 'evaluate', *common, log='evaluate.log')
py('extensions.sharp_object.train_hand_classifier', '--stage', 'export', *common, '--pose', POSE,
   '--run-tag', RUN_TAG, log='export.log')
sync_to_drive()
report = json.loads((LOCAL_OUT / 'train' / 'evaluation.json').read_text())
fmt = lambda d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}
for key in ('crop_all_val', 'crop_all_test', 'crop_vn_val', 'crop_vn_test'):
    print(f'{key:16s}', fmt(report[key]))
print('\\nNgưỡng cảnh báo:', report['alert_threshold'], '-', report['threshold_rule'])
for key in ('alert_val', 'alert_test', 'alert_vn_test', 'alert_hard_negative_val', 'alert_hard_negative_test'):
    if key in report:
        print(f'{key:24s}', fmt(report[key]))
print('Báo nhầm theo crop (test):', report['crop_false_alarm_test'])
print('\\nĐường cong ngưỡng (test):')
for row in report['alert_curve']:
    print('  ', row['threshold'], fmt(row['test']))
print('\\nFile trên Drive:')
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.suffix in {'.zip', '.json', '.pt', '.log', '.jpg', '.txt'}:
        print(f'  {str(path.relative_to(OUT)):40s} {path.stat().st_size / 1e6:8.2f} MB')
(OUT / 'DONE.txt').write_text(f'xong {time.strftime("%Y-%m-%d %H:%M:%S")}, {(time.time() - STARTED) / 3600:.1f} giờ',
                              encoding='utf-8')
print('\\nTổng thời gian: %.1f giờ' % ((time.time() - STARTED) / 3600))
print('Bước tiếp theo: tải sharp_object_' + RUN_TAG + '.zip về máy, cài bằng install_sharp_object.py rồi thử trên camera.')
"""

CLIPS_MD = """
## Bước 6 · Độ ổn định theo thời gian trên clip tự quay

Chấm các clip trong `source_clips/` của zip VN mới nhất ở 4 khung/giây bằng bản mới và hand_01.
`alert_share` = tỉ lệ bước có điểm ≥ ngưỡng (clip nào cũng có người cầm dao nên càng cao càng tốt);
`drops` = số lần điểm tụt dưới ngưỡng ngay sau một bước trên ngưỡng (khung chập chờn), càng ít càng tốt.
Clip cùng phòng với khung hình đã train nên chỉ đo độ ổn định, không phải độ tổng quát.
"""

CLIPS_CODE = """
clips_dir, previous_dir = WORK / 'clips', WORK / 'previous'
for path in (clips_dir, previous_dir):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
with zipfile.ZipFile(reviewed[-1]) as bundle:
    for name in bundle.namelist():
        if '/source_clips/' in '/' + name and name.lower().endswith('.mp4'):
            (clips_dir / Path(name).name).write_bytes(bundle.read(name))
print('Clip:', len(list(clips_dir.iterdir())))
extra = []
if PREVIOUS_ZIP.is_file():
    with zipfile.ZipFile(PREVIOUS_ZIP) as bundle:
        bundle.extractall(previous_dir)
    extra = ['--extra-package', previous_dir]
if any(clips_dir.iterdir()):
    py('extensions.sharp_object.eval_hand_clips', '--package', LOCAL_OUT / 'train' / 'package', *extra,
       '--clips', clips_dir, '--out', LOCAL_OUT / 'train' / 'clips_eval.json', log='clips_eval.log')
    shutil.copy(LOCAL_OUT / 'train' / 'clips_eval.json', OUT / 'clips_eval.json')
    sync_to_drive()
"""

VARIANTS = {
    "hand_01": """RUN_TAG = 'hand_01'
INIT_FROM = None              # train từ ImageNet
LR = 1e-4
VN_REVIEWED_GLOB = None
VN_TRAIN_GROUPS = []
BASE_POOL_VERSION = None
EXTRA_POOL_SOURCES = []
VN_NEGATIVES_GLOB = None
HARD_NEGATIVE_WEIGHT = 1.0""",
    "hand_02": """RUN_TAG = 'hand_02'
HAND_01 = Path('/content/drive/MyDrive/SENTINEL/artifacts/sharp_object/hand_01')
INIT_FROM = HAND_01 / 'train' / 'best.pt'   # train tiếp từ hand_01
PREVIOUS_ZIP = HAND_01 / 'sharp_object_hand_01.zip'  # để so sánh ở bước 6
LR = 3e-5                     # lr nhỏ vì chỉ tinh chỉnh
VN_REVIEWED_GLOB = 'vn_weapons_reviewed_dataset_*.zip'  # zip duyệt mới nhất (011: 22 khung tự quay)
VN_TRAIN_GROUPS = ['user_filming']  # video tự quay chỉ dùng để train
BASE_POOL_VERSION = None
EXTRA_POOL_SOURCES = []
VN_NEGATIVES_GLOB = None
HARD_NEGATIVE_WEIGHT = 1.0""",
    "hand_03": """RUN_TAG = 'hand_03'
HAND_02 = Path('/content/drive/MyDrive/SENTINEL/artifacts/sharp_object/hand_02')
INIT_FROM = HAND_02 / 'train' / 'best.pt'   # train tiếp từ hand_02
PREVIOUS_ZIP = HAND_02 / 'sharp_object_hand_02.zip'  # để so sánh ở bước 6
LR = 3e-5
VN_REVIEWED_GLOB = 'vn_weapons_reviewed_dataset_*.zip'
VN_TRAIN_GROUPS = ['user_filming']
BASE_POOL_VERSION = 'v1'      # pool v2 = pool v1 (cache trên Drive) + bản âm COCO
EXTRA_POOL_SOURCES = ['coco_neg']  # người cầm gậy bóng chày / vợt / ô / ván trượt, không có vật sắc
VN_NEGATIVES_GLOB = 'vn_negatives_*.zip'  # nếu sau này có video bản âm tự quay
HARD_NEGATIVE_WEIGHT = 3.0    # lấy mẫu bản âm khó gấp 3""",
}

HAND_02_INTRO = """
# Người cầm vật sắc nhọn · hand_02: tinh chỉnh hand_01 với video tự quay

Giống hand_01 (pose → crop quanh tay → ResNet50), khác ở:

- **Train tiếp từ `best.pt` của hand_01**, lr 3e-5, tối đa 10 epoch.
- **Thêm zip VN duyệt mới nhất** (`vn_weapons_reviewed_dataset_*.zip`, hiện là 011): 22 khung tự quay dao phóng lợn
  ở các tư thế khó (dựng đứng, chĩa thẳng, sát người). Nhóm `user_filming` luôn vào **train**.
- **Bước 6:** chấm 16 clip tự quay theo thời gian, so với hand_01.

**Cách chạy:** GPU **L4**, *Run all*. Phiên tự ngắt khi xong hoặc lỗi.
Kết quả ở `MyDrive/SENTINEL/artifacts/sharp_object/hand_02/`. Ước tính 40–80 phút, phần lớn là bước 3.
"""

HAND_03_INTRO = """
# Người cầm vật sắc nhọn · hand_03: bớt báo nhầm khi cầm gậy

hand_02 chấm ô quanh tay **cầm gậy** tới 0.96–0.996 (video phá hoại), nên báo nhầm "cầm vật sắc nhọn".
hand_03 tinh chỉnh tiếp từ hand_02 với **bản âm khó** lấy từ dữ liệu có sẵn, không cần quay thêm:

- **COCO `coco_neg`** (tối đa 3000 ảnh): ảnh có người và gậy bóng chày / vợt tennis / ô / ván trượt tuyết,
  không có dao hay kéo. Mọi ô quanh tay ở các ảnh này là bản âm, lấy mẫu **×3**.
- Pool công khai **v2** = pool v1 đã cache + `coco_neg`; lần đầu tạo và cache lên Drive (thêm ~10–20 phút), lần sau dùng lại.
- Giữ toàn bộ dữ liệu hand_02 (zip VN 011, 22 khung tự quay ở train), lr 3e-5, tối đa 10 epoch.
- Ngưỡng vẫn chọn theo báo nhầm ≤ 3% trên ảnh không vũ khí của val, nay gồm cả ảnh cầm gậy/vợt.

**Bước 5:** `alert_hard_negative_test` = tỉ lệ ảnh *người cầm gậy/vợt/ô* bị báo nhầm (càng thấp càng tốt).
**Bước 6:** 16 clip dao phóng lợn tự quay, so với hand_02: `alert_share` không được tụt nhiều.

**Cách chạy:** GPU **L4**, *Run all*. Kết quả ở `MyDrive/SENTINEL/artifacts/sharp_object/hand_03/`. Ước tính 50–90 phút.
"""


def build(revision, variant="hand_01"):
    setup = SETUP.replace("{revision}", revision).replace("{variant}", VARIANTS[variant])
    setup = setup.replace("{{", "{").replace("}}", "}")
    if variant in ("hand_02", "hand_03"):
        setup = setup.replace("EPOCHS = 25", "EPOCHS = 10")
    if variant == "hand_03":
        setup = setup.replace("PUBLIC_POOL_VERSION = 'v1'    # dùng lại pool công khai sharp_02 đã cache trên Drive",
                              "PUBLIC_POOL_VERSION = 'v2'    # v1 + bản âm COCO (tạo lần đầu, lần sau dùng cache)")
    intro = {"hand_02": HAND_02_INTRO, "hand_03": HAND_03_INTRO}.get(variant, INTRO)
    parts = [("markdown", intro), ("code", setup),
             ("code", SOURCE), ("code", INSTALL), ("markdown", POOL_MD), ("code", POOL_CODE), ("markdown", VN_MD),
             ("code", VN_CODE), ("markdown", CROPS_MD), ("code", CROPS_CODE), ("markdown", TRAIN_MD),
             ("code", TRAIN_CODE), ("markdown", EVAL_MD), ("code", EVAL_CODE),
             *([("markdown", CLIPS_MD), ("code", CLIPS_CODE)] if variant != "hand_01" else []),
             ("markdown", "## Tự ngắt phiên\n\nKết quả đã nằm trên Drive nên ngắt phiên để không tốn compute unit.\n"),
             ("code", "disconnect('XONG. Kết quả ở ' + str(OUT))\n")]
    cells = [cell(kind, value, index) for index, (kind, value) in enumerate(parts)]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    output = OUTPUTS[variant]
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", output)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or len(sys.argv[1]) != 40 or (len(sys.argv) == 3 and sys.argv[2] not in VARIANTS):
        raise SystemExit("usage: build_sharp_hand_notebook.py <40-char source commit sha> [hand_01|hand_02]")
    build(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "hand_01")
