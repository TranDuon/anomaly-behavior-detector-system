import base64
import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zlib


ROOT = Path(__file__).resolve().parents[1]


def build():
    previous = json.loads((ROOT / 'colab/mmaction_fixed_running_ground_20261105.ipynb').read_text(encoding='utf-8'))
    overlays = {name: (ROOT / 'mmaction2_demo' / name).read_text(encoding='utf-8')
                for name in ['train_three_actions.py', 'action_engine.py', 'person_recovery.py']}
    payload = base64.b64encode(zlib.compress(json.dumps(overlays).encode())).decode()
    checksum = hashlib.sha256(payload.encode()).hexdigest()
    cells = []

    def add(kind, source):
        cells.append({'cell_type': kind, 'metadata': {}, 'id': f'three-actions-{len(cells)}',
                      'source': dedent(source).lstrip('\n').splitlines(keepends=True),
                      **({'execution_count': None, 'outputs': []} if kind == 'code' else {})})

    add('markdown', '''
    # Fine-tune riêng ĐÁNH – VẬT/KẸP CỔ – NGÃ

    Ba sigmoid độc lập: `strike`, `grapple`, `fall`. Có thể đồng thời vật và ngã.
    Bình thường là cả ba nhãn 0; không có một nhãn cảnh báo chung thay thế ba nhãn.
    Giữ nguyên checkpoint cũ. Fine-tune ba đầu từ checkpoint cha; backbone AVA
    và detector đóng băng. Đây chưa phải fine-tune toàn bộ backbone.

    **Chạy kiểm tra nhãn trên CPU trước. Chỉ đổi sang GPU khi audit báo ready.**
    Dữ liệu nhị phân cũ không tự đổi thành nhãn ba hành vi. `null` là chưa biết,
    không phải âm. Clip đã dùng để test, gồm AIRTLab 37/87, phải giữ ở test.
    Cần nhãn từng cửa sổ và video public trên Drive; không upload video iPhone.
    GPU tối đa 30 phút cho trích đặc trưng và train, có dừng sớm validation.
    ''')
    add('code', '''
    from pathlib import Path
    import base64, hashlib, io, json, os, shutil, subprocess, sys, time, urllib.request, zipfile, zlib
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = Path('/content/drive/MyDrive/SENTINEL')
    DATA = DRIVE/'datasets'
    MANIFEST = DATA/'three_actions_v1/review.json'
    RUN_TAG = 'three_actions_'+time.strftime('%Y%m%d_%H%M%S')
    RUN = DRIVE/'artifacts/three_actions_v1'/RUN_TAG
    RUN.mkdir(parents=True, exist_ok=True)
    WORK = Path('/content/three_actions_v1'); WORK.mkdir(exist_ok=True)
    CODE = WORK/'code'; DEMO = CODE/'mmaction2_demo'
    DEMO.mkdir(parents=True, exist_ok=True)
    WEIGHTS = WORK/'weights'; WEIGHTS.mkdir(exist_ok=True)
    print('Manifest:', MANIFEST)
    print('Output:', RUN)
    ''')
    add('code', f'''
    PAYLOAD = {payload!r}
    assert hashlib.sha256(PAYLOAD.encode()).hexdigest() == {checksum!r}
    overlays = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
    for name, source in overlays.items():
        assert Path(name).name == name
        (DEMO/name).write_text(source, encoding='utf-8')
    if not MANIFEST.exists():
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        template = {{'schema_version': 1, 'labels': ['strike','grapple','fall'],
            'policy': '0=no action, 1=visible action, null=unknown; reviewed windows only; subjects and recordings stay in one split',
            'source_example_not_training_data': {{'video':'relative/path.mp4','sha256':'REPLACE_WITH_REAL_SHA256',
                'group':'one-recording-or-subject','split':'train','reviewer':'name',
                'public_source_url':'https://publisher.example/dataset',
                'windows':[{{'anchor_seconds':2.0,'targets':{{'strike':0,'grapple':1,'fall':0}}}}]}},
            'sources': []}}
        MANIFEST.write_text(json.dumps(template,ensure_ascii=False,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(DEMO/'train_three_actions.py'),'--manifest',str(MANIFEST),
        '--root',str(DATA),'--output',str(RUN),'--stage','audit'],check=True)
    audit = json.loads((RUN/'label_audit.json').read_text())
    print('AUDIT:',audit['status'])
    if audit['status'] != 'ready':
        raise RuntimeError('CHƯA TRAIN: thiếu nhãn đã duyệt cho ba hành vi. Xem label_audit.json; chưa tải model GPU.')
    ''')
    add('markdown', '''
    ## Chỉ chạy phần sau khi audit ready và đã chọn GPU
    Không bỏ qua kiểm tra nhãn. Đường dẫn video trong review.json tính từ SENTINEL/datasets.
    Dùng group theo người/phiên quay, không chia ngẫu nhiên các cửa sổ cùng video.
    ''')
    add('code', '''
    audit = json.loads((RUN/'label_audit.json').read_text())
    assert audit['status'] == 'ready', 'Không train trên nhãn gộp hoặc chưa duyệt'
    subprocess.run(['nvidia-smi'],check=True)
    SOURCE_REVISION = 'fda444194032f69845176887eee9b978d3373876'
    archive = WORK/'source.zip'
    with urllib.request.urlopen('https://codeload.github.com/TranDuon/anomaly-behavior-detector-system/zip/'+SOURCE_REVISION,timeout=180) as response, archive.open('wb') as output:
        shutil.copyfileobj(response,output)
    with zipfile.ZipFile(archive) as package:
        assert all((WORK/item.filename).resolve().is_relative_to(WORK.resolve()) for item in package.infolist())
        package.extractall(WORK)
    source = next(path for path in WORK.iterdir() if (path/'mmaction2_demo/train_fixed_running.py').is_file())
    shutil.copytree(source,CODE,dirs_exist_ok=True)
    for name, source in overlays.items():
        (DEMO/name).write_text(source,encoding='utf-8')
    ''')
    add('code', ''.join(previous['cells'][3]['source']))
    add('code', ''.join(previous['cells'][5]['source']))
    add('code', '''
    os.environ['MPLBACKEND']='Agg'
    command=[PY,DEMO/'train_three_actions.py','--manifest',MANIFEST,'--root',DATA,
        '--weights',WEIGHTS,'--output',RUN,'--stage','train','--minutes','30','--epochs','120']
    run(command,'train_three_actions',timeout=2100)
    report=json.loads((RUN/'report.json').read_text())
    assert report['status']=='completed'
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Checkpoint riêng, chưa cài lên web:',RUN/'three_actions_pilot.zip')
    ''')
    cells[2]['source'] = (ROOT/'scripts/colab_three_actions_data.py').read_text(encoding='utf-8').splitlines(keepends=True)
    target = ROOT / 'colab/mmaction_three_actions_drive.ipynb'
    target.write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},
        'colab':{'name':target.name}},'cells':cells},ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return target


if __name__ == '__main__':
    print(build())
