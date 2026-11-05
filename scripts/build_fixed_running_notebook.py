"""Build the pinned Colab notebook for the fixed-camera running pilot."""

import json
from pathlib import Path
from textwrap import dedent


SOURCE_REVISION = "a375d8018acc86d2941f5ba4120f82318d4e5994"


def cell(kind, value, index):
    return {
        "cell_type": kind,
        "metadata": {},
        "source": dedent(value).lstrip("\n").splitlines(keepends=True),
        "id": f"fixed-running-{index:02d}",
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


cells = [
    cell("markdown", """
    # MMAction2 · camera cố định · chạy nhanh

    Notebook tạo **checkpoint mới** từ `interaction_fall_wrestle.pt` hiện có.
    Nhánh đánh/vật/ngã được giữ nguyên trọng số; nhánh chạy nhanh khởi tạo từ
    checkpoint cha và được train bằng clip camera mặt đất cố định. Chạy tuần tự
    các cell trên **Colab Pro GPU**. Không cần tải checkpoint từ máy lên.

    Bộ train chỉ lấy **UCF-ARG ground**: 48 `running` là dương, 47 `jogging` và
    48 `walking` là âm. Một clip jogging lỗi giải mã đã được loại. Chia theo
    người: 01–08 train, 09–10 validation,
    11–12 test. Loại aerial, rooftop, phim, clip camera di động, CASIA cần xin
    quyền và các clip VIRAT quá nhỏ/không có nhãn chạy chậm đối chiếu. Video
    UCF-ARG quay ngoài trời; cần thử thêm camera trong nhà thực tế trước khi dùng demo.

    Nguồn dữ liệu: [UCF-ARG](https://vision.eecs.ucf.edu/data/UCF-ARG.html),
    bản clip [SPHAR](https://github.com/AlexanderMelde/SPHAR-Dataset). Video
    chỉ dùng cho mục đích nghiên cứu theo điều kiện của nguồn gốc; notebook
    chỉ tải 144 clip được chọn, không đưa video vào GitHub hay Drive output.
    """, 0),
    cell("code", """
    from pathlib import Path
    import hashlib, io, json, os, shutil, subprocess, sys, time, urllib.request, zipfile
    from google.colab import drive

    RUN_TAG = 'fixed_running_ground_01'  # Đổi tên nếu muốn chạy một thí nghiệm mới.
    EPOCHS = 40
    TRAIN_MINUTES = 90
    assert RUN_TAG and all(c.isalnum() or c in '_-' for c in RUN_TAG)
    subprocess.run(['nvidia-smi'], check=True)
    drive.mount('/content/drive', timeout_ms=180000)
    roots = [p for p in (Path('/content/drive/MyDrive'), Path('/content/drive/My Drive')) if p.is_dir()]
    if not roots:
        raise RuntimeError('Không thấy MyDrive sau khi mount Google Drive')
    DRIVE = next((p/'SENTINEL' for p in roots if (p/'SENTINEL').is_dir()), roots[0]/'SENTINEL')
    RUN = DRIVE/'artifacts/mmaction_fixed_running'/RUN_TAG
    RUN.mkdir(parents=True, exist_ok=True)
    probe = RUN/'drive_write_probe.txt'
    probe.write_text('write OK', encoding='utf-8')
    assert probe.read_text(encoding='utf-8') == 'write OK'
    WORK = Path('/content/fixed_running'); WORK.mkdir(exist_ok=True)
    CODE = WORK/'code'; CODE.mkdir(exist_ok=True)
    DEMO = CODE/'mmaction2_demo'
    WEIGHTS = WORK/'weights'; WEIGHTS.mkdir(exist_ok=True)
    DATASET = WORK/'dataset'
    CACHE = RUN/'feature_cache'
    print('Kết quả lưu trên Drive:', RUN)
    """, 1),
    cell("code", """
    # Pin đúng phiên bản trainer đã kiểm tra; không lấy code thay đổi giữa lúc train.
    SOURCE_REVISION = 'a375d8018acc86d2941f5ba4120f82318d4e5994'
    marker = WORK/'source_revision.txt'
    if (DEMO/'train_fixed_running.py').is_file() and (
            not marker.is_file() or marker.read_text().strip() != SOURCE_REVISION):
        raise RuntimeError('Runtime Colab chứa code khác phiên bản; hãy restart runtime')
    if not (DEMO/'train_fixed_running.py').is_file():
        source_zip = WORK/'source.zip'
        url = 'https://codeload.github.com/TranDuon/anomaly-behavior-detector-system/zip/' + SOURCE_REVISION
        with urllib.request.urlopen(url, timeout=180) as response, source_zip.open('wb') as output:
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(source_zip) as package:
            for item in package.infolist():
                target = (WORK/item.filename).resolve()
                assert target.is_relative_to(WORK.resolve()), item.filename
            package.extractall(WORK)
        source = next(p for p in WORK.iterdir() if (p/'mmaction2_demo/train_fixed_running.py').is_file())
        shutil.copytree(source, CODE, dirs_exist_ok=True)
        marker.write_text(SOURCE_REVISION)
    print('Trainer pinned:', SOURCE_REVISION)
    """, 2),
    cell("code", """
    # Mọi checkpoint đầu vào đều lấy từ Google Drive; không dùng upload thủ công.
    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import google.auth

    checks = {
        'interaction_fall_wrestle.pt': ('1K_KOzEyxpbrnQExEcS4Hy3UkeMCn144_',
          'ddb09180284c4093189c53b7decbfcc5b51fd2d9b7700536d104524bceb5c344'),
        'interaction_fall_wrestle.json': ('1Via7ET9g6iQMjR-pX7NRpj9gKQktFQWe', None),
        'action_slowonly_r101.pth': ('1aU7C2tCp6NQc1NDm5GRwdkWtgqkr3lkt',
          '163785949c5178484012a6593d12d6fcf4d36be404fa42b5899b314e91f4703b'),
        'person_faster_rcnn_r50.pth': ('1gbBBgvKhp1p3FNDRoTsV92_t2vqMwUyP',
          'a5d8aa15c75b961b9c73f28a6e92af370835639a2939ff890852b052a5256a5b'),
    }
    def sha256(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    service = None
    def drive_download(file_id, target):
        global service
        if service is None:
            auth.authenticate_user()
            credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
            service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        request = service.files().get_media(fileId=file_id)
        temporary = target.with_suffix(target.suffix+'.partial')
        with temporary.open('wb') as output:
            downloader = MediaIoBaseDownload(output, request, chunksize=8*1024*1024)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=3)
        temporary.replace(target)
    parent_run = DRIVE/'artifacts/mmaction_interaction_fall_wrestle/fall_wrestle_pilot_01'
    parent_public = DRIVE/'artifacts/mmaction_interaction_fall_wrestle/public_weights'
    alternatives = {
        'interaction_fall_wrestle.pt': [parent_run/'interaction.pt'],
        'interaction_fall_wrestle.json': [parent_run/'interaction.json'],
        'action_slowonly_r101.pth': [parent_public/'action_slowonly_r101.pth'],
        'person_faster_rcnn_r50.pth': [parent_public/'person_faster_rcnn_r50.pth'],
    }
    for name, (file_id, expected) in checks.items():
        target = WEIGHTS/name
        if target.is_file() and (expected is None or sha256(target) == expected):
            continue
        source = next((p for p in alternatives[name]
                       if p.is_file() and (expected is None or sha256(p) == expected)), None)
        if source is not None:
            shutil.copyfile(source, target)
        else:
            drive_download(file_id, target)
        if expected is not None and sha256(target) != expected:
            raise ValueError('Sai SHA-256 checkpoint Drive: '+name)
    meta = json.loads((WEIGHTS/'interaction_fall_wrestle.json').read_text(encoding='utf-8'))
    assert meta['checkpoint_sha256'] == checks['interaction_fall_wrestle.pt'][1]
    print('Đã xác minh checkpoint cha và 2 trọng số gốc từ Drive')
    """, 3),
    cell("code", """
    # Chỉ tải đúng camera ground, walking/jogging/running; kiểm tra Git blob từng clip.
    result = subprocess.run([sys.executable, str(DEMO/'prepare_fixed_running.py'),
                             '--destination', str(DATASET)], check=True)
    MANIFEST = DATASET/'fixed_running_manifest.json'
    rows = json.loads(MANIFEST.read_text(encoding='utf-8'))['rows']
    assert len(rows) == 143
    from collections import Counter
    print('Dataset:', Counter((r['action'], r['split']) for r in rows))
    """, 4),
    cell("code", """
    # Môi trường giống pipeline MMAction2 đã dùng để train checkpoint cha.
    os.environ.update(MPLBACKEND='Agg', CUDA_MODULE_LOADING='LAZY', OMP_NUM_THREADS='4')
    ENV = WORK/'env'; PY = ENV/'bin/python'
    def run(command, label, timeout=1800):
        log = RUN/(label+'.log')
        with log.open('a', encoding='utf-8') as output:
            result = subprocess.run([str(x) for x in command], stdout=output,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'{label} failed; see {log}\\n{log.read_text(encoding="utf-8")[-4000:]}')
        print(label, 'OK')
    if not PY.is_file():
        run([sys.executable,'-m','pip','install','-q','uv==0.8.22'], 'install_uv')
        run(['uv','python','install','3.10'], 'install_python')
        run(['uv','venv','--python','3.10','--seed',ENV], 'create_env')
    run([PY,'-m','pip','install','-q','numpy==1.26.4','setuptools==69.5.1','wheel'], 'install_base')
    run([PY,'-m','pip','install','-q','torch==2.0.1+cu118','torchvision==0.15.2+cu118',
         '--index-url','https://download.pytorch.org/whl/cu118'], 'install_torch')
    run([PY,'-m','pip','install','-q','mmcv==2.1.0','--only-binary=mmcv',
         '-f','https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html',
         'mmengine==0.10.7','mmdet==3.2.0','decord==0.6.0','einops==0.8.1',
         'importlib-metadata','scikit-learn==1.3.2','opencv-python==4.8.1.78'], 'install_openmmlab')
    if not (DEMO/'repo/.git').is_dir():
        run(['git','clone','--depth','1','--branch','v1.2.0',
             'https://github.com/open-mmlab/mmaction2.git',DEMO/'repo'], 'clone_mmaction')
    run([PY,'-m','pip','install','--no-deps','--no-build-isolation','-e',DEMO/'repo'], 'install_mmaction')
    os.environ['PYTHONPATH'] = os.pathsep.join(map(str,(CODE,DEMO,DEMO/'repo')))
    run([PY,'-c','import torch, mmcv; from mmcv.ops import roi_align; assert torch.cuda.is_available()'], 'check_cuda')
    """, 5),
    cell("code", """
    args = [PY, DEMO/'train_fixed_running.py', '--manifest', MANIFEST,
            '--weights', WEIGHTS, '--cache', CACHE, '--output', RUN,
            '--device', 'cuda:0', '--epochs', str(EPOCHS), '--minutes', str(TRAIN_MINUTES)]
    run(args+['--stage','features'], 'extract_fixed_running_features', timeout=4*3600)
    print('Đã lưu feature cache lên Drive:', CACHE)
    """, 6),
    cell("code", """
    run(args+['--stage','train'], 'train_fixed_running', timeout=TRAIN_MINUTES*60+300)
    report = json.loads((RUN/'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'completed'
    print('Validation:', report['development'])
    print('Test:', report['test'])
    """, 7),
    cell("code", """
    package = RUN/'mmaction_fixed_running_pilot.zip'
    metadata = json.loads((RUN/'interaction_fixed_running.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            'interaction_fixed_running.pt','interaction_fixed_running.json','report.json','protocol.json'}
    assert sha256(RUN/'interaction_fixed_running.pt') == metadata['checkpoint_sha256']
    assert metadata['parent_checkpoint_sha256'] == checks['interaction_fall_wrestle.pt'][1]
    os.sync()
    time.sleep(5)
    assert package.is_file() and package.stat().st_size > 1_000_000
    drive.flush_and_unmount()
    print('Checkpoint mới:', package)
    print('Báo cáo:', RUN/'report.json')
    print('Checkpoint cũ trên Drive vẫn giữ nguyên.')
    """, 8),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
target = Path(__file__).resolve().parents[1] / "colab/mmaction_fixed_running_ground_20261105.ipynb"
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(target)
