"""Build a self-contained public-data Colab experiment without pushing Git."""

import base64
import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]


def build():
    destination = ROOT / 'mmaction2_demo/results/pose_specialists_bundle'
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / 'source.zip'
    with zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as package:
        for name in ('pose_specialists.py', 'train_pose_specialists.py', 'train_three_actions.py'):
            package.write(ROOT / 'mmaction2_demo' / name, name)
        package.write(ROOT / 'models/yolov8n-pose.pt', 'yolov8n-pose.pt')
        package.write(ROOT / 'mmaction2_demo/results/three_actions_review/reviewed_manifest.json', 'review.json')
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    cells = []

    def add(kind, source):
        cells.append({'cell_type': kind, 'metadata': {}, 'id': 'pose-specialists-' + str(len(cells)),
                      'source': dedent(source).lstrip('\n').splitlines(keepends=True),
                      **({'execution_count': None, 'outputs': []} if kind == 'code' else {})})

    add('markdown', '''
    # Đối chứng NGÃ / VẬT bằng chuỗi pose — không thay ĐÁNH / CHẠY
    ST-GCN chính thức NTU60 → hai nhánh độc lập: ngã và tương tác cặp người.
    Hai cấu hình: đóng băng backbone, rồi fine-tune hai tầng cuối với learning rate nhỏ.
    Không đổi ôm thành vật; không đổi nằm thành ngã. Không upload video iPhone.
    Chỉ dùng 37 video public đã duyệt, nhãn AI hỗ trợ, chưa phải tập nghiệm thu độc lập.
    Đây là kiểm tra kiến trúc mới, không phải mở rộng dataset hay bảo đảm cải thiện.
    Ngưỡng/checkpoint chọn bằng development; test chỉ báo cáo. Không tự cài lên web.
    Cần runtime GPU. Trích pose + hai cấu hình tối đa 30 phút; cài môi trường chạy trước.
    ''')
    add('code', f'''
    from pathlib import Path
    import base64, hashlib, io, json, os, shutil, subprocess, sys, time, urllib.request, zipfile
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = Path('/content/drive/MyDrive/SENTINEL')
    DATA = DRIVE/'datasets'
    WORK = Path('/content/three_actions_v1')
    CODE = WORK/'code'; DEMO = CODE/'mmaction2_demo'
    DEMO.mkdir(parents=True,exist_ok=True)
    ARTIFACTS = DRIVE/'artifacts/pose_specialists_v1'
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    RUN = ARTIFACTS/('run_'+time.strftime('%Y%m%d_%H%M%S'))
    BUNDLE = DRIVE/'artifacts'/('pose_specialists_source_{checksum[:12]}.zip')
    assert BUNDLE.is_file(), 'Thiếu source bundle đã chuẩn bị trên Drive'
    assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == {checksum!r}, 'Sai source bundle'
    with zipfile.ZipFile(BUNDLE) as package:
        assert set(package.namelist()) == {{'pose_specialists.py','train_pose_specialists.py','train_three_actions.py','yolov8n-pose.pt','review.json'}}
        package.extractall(DEMO)
    MANIFEST=DEMO/'review.json'
    reviewed=json.loads(MANIFEST.read_text(encoding='utf-8'))
    print('37 public clips, giữ nguyên split:',len(reviewed['sources']))
    print('Output mới:',RUN)
    ''')
    add('code', '''
    from concurrent.futures import ThreadPoolExecutor
    def public_source(item):
        target=(DATA/item['video']).resolve()
        assert target.is_relative_to(DATA.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:
            return
        url=item['download_url']
        assert url.startswith('https://raw.githubusercontent.com/')
        with urllib.request.urlopen(url,timeout=120) as response:
            content=response.read(item['size']+1)
        assert len(content)==item['size'] and hashlib.sha256(content).hexdigest()==item['sha256']
        temporary=target.with_suffix('.partial'); temporary.write_bytes(content); temporary.replace(target)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(public_source,reviewed['sources']))
    print('Public sources verified')
    ''')
    old = json.loads((ROOT/'colab/mmaction_fixed_running_ground_20261105.ipynb').read_text(encoding='utf-8'))
    setup = ''.join(old['cells'][5]['source']).replace("log = RUN/(label+'.log')", "log = ARTIFACTS/('setup_'+label+'.log')")
    add('code', setup + "\nrun([PY,'-m','pip','install','--no-deps','ultralytics==8.2.103','ultralytics-thop==2.0.18','psutil==6.1.1','py-cpuinfo==9.0.0'], 'install_pose')\n")
    add('code', '''
    os.environ['MPLBACKEND']='Agg'
    print('Bắt đầu trích pose thật 15 FPS + fine-tune, không dùng nội suy từ 9 frame AVA.')
    command=[PY,DEMO/'train_pose_specialists.py','--manifest',MANIFEST,'--root',DATA,
        '--pose-weights',DEMO/'yolov8n-pose.pt','--output',RUN,'--minutes','30','--epochs','80']
    log=ARTIFACTS/('train_'+RUN.name+'.log')
    with log.open('w',encoding='utf-8') as output:
        process=subprocess.Popen([str(value) for value in command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1,env=os.environ.copy())
        import threading
        timer=threading.Timer(1830,process.kill); timer.start()
        try:
            for line in process.stdout:
                print(line,end=''); output.write(line); output.flush()
            returncode=process.wait()
        finally:
            timer.cancel()
    assert returncode==0, f'Dừng, không tự retry. Xem log: {log}'
    report=json.loads((RUN/'comparison.json').read_text())
    assert report['status']=='completed'
    print(json.dumps(report,ensure_ascii=False,indent=2))
    package=shutil.make_archive(str(RUN)+'_pilot','zip',RUN)
    print('Gói đối chứng, chưa thay model web:',package)
    ''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                             'colab': {'name': 'mmaction_pose_specialists_drive.ipynb'}}, 'cells': cells}
    target = ROOT / 'colab/mmaction_pose_specialists_drive.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (destination/'bundle.json').write_text(json.dumps({'sha256': checksum, 'source': str(source),
        'drive_name': 'pose_specialists_source_' + checksum[:12] + '.zip'}), encoding='utf-8')
    return target


if __name__ == '__main__':
    print(build())
