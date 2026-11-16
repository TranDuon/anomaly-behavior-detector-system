"""Build the pinned Colab notebook for the vandalism extension (V2 data + V3 training).

    python scripts/build_vandalism_notebook.py <source_commit_sha>
"""

import json
import sys
from pathlib import Path
from textwrap import dedent

REPO = "TranDuon/anomaly-behavior-detector-system"
OUTPUT = Path(__file__).resolve().parents[1] / "colab" / "vandalism_vjepa2_v1.ipynb"


def cell(kind, value, index):
    return {
        "cell_type": kind,
        "metadata": {},
        "source": dedent(value).lstrip("\n").splitlines(keepends=True),
        "id": f"vandalism-{index:02d}",
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


def build(revision):
    cells = [
        cell("markdown", """
        # Phá hoại tài sản · V-JEPA 2 + attentive probe

        Notebook train **extension phá hoại tài sản** (đá/đập/xô/ném đồ). Chạy lần lượt từng cell trên
        **Colab Pro** (Runtime → Change runtime type → **L4** hoặc **A100**).

        - **Dữ liệu tải trong phiên**, không cần tài khoản, không lưu dataset lên Drive:
          - Kinetics-700 `smashing` + `breaking glass` (dương tính), 17 lớp âm khó.
          - ASHEN Vandalism (CCTV, nhãn theo video, trọng số 0.5) + ASHEN normal.
          - Charades: ném đồ bình thường và sinh hoạt trong nhà (âm khó).
        - **Model:** V-JEPA 2 ViT-L đóng băng → 64 token / cửa sổ 8 khung (3.75 fps, ~2.1 s,
          giống đầu vào runtime) → attentive probe, học multiple-instance (top-3 cửa sổ).
        - **Kết quả lưu ra Drive:** `MyDrive/SENTINEL/artifacts/vandalism/<RUN_TAG>/`, gồm
          `vandalism.pt`, `vandalism.json`, `report.json`, `protocol.json`, `vandalism_<RUN_TAG>.zip`.

        Thời gian ước tính trên L4:
        - Tải dữ liệu: 20–40 phút (khoảng 20–25 GB).
        - Trích feature: 45–90 phút.
        - Train probe: dưới 10 phút.

        Nếu phiên bị ngắt, chạy lại từ đầu: dữ liệu và feature đã có trong phiên sẽ được bỏ qua.
        Bật `CACHE_ON_DRIVE = True` nếu muốn giữ feature qua các phiên (tốn khoảng 5–10 GB Drive).

        Giấy phép nguồn: Kinetics (nhãn CC BY 4.0, video thuộc chủ YouTube), ASHEN (CC BY-NC 4.0),
        Charades (phi thương mại). Chỉ dùng cho đồ án / nghiên cứu.
        """, 0),
        cell("code", f"""
        from pathlib import Path
        import json, os, shutil, subprocess, sys, time, urllib.request, zipfile
        from google.colab import drive

        RUN_TAG = 'vjepa2_probe_01'   # đổi tên khi chạy thí nghiệm mới
        CACHE_ON_DRIVE = False        # True: giữ feature trên Drive để chạy lại không phải trích lại
        SOURCE_REVISION = '{revision}'

        assert RUN_TAG and all(c.isalnum() or c in '_-' for c in RUN_TAG)
        subprocess.run(['nvidia-smi'], check=True)
        drive.mount('/content/drive', timeout_ms=180000)
        roots = [p for p in (Path('/content/drive/MyDrive'), Path('/content/drive/My Drive')) if p.is_dir()]
        assert roots, 'Không thấy MyDrive sau khi mount Google Drive'
        OUT = roots[0] / 'SENTINEL' / 'artifacts' / 'vandalism' / RUN_TAG
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'drive_write_probe.txt').write_text('write OK', encoding='utf-8')
        WORK = Path('/content/vandalism_work')
        SRC, DATA = WORK / 'src', WORK / 'data'
        CACHE = (OUT / 'feature_cache') if CACHE_ON_DRIVE else (WORK / 'cache')
        for path in (WORK, DATA, CACHE):
            path.mkdir(parents=True, exist_ok=True)
        print('Kết quả sẽ lưu tại:', OUT)
        """, 1),
        cell("code", f"""
        # Mã nguồn được ghim theo commit để kết quả tái lập được.
        marker = SRC / '.revision'
        if not marker.is_file() or marker.read_text().strip() != SOURCE_REVISION:
            shutil.rmtree(SRC, ignore_errors=True)
            archive = WORK / 'source.zip'
            url = 'https://codeload.github.com/{REPO}/zip/' + SOURCE_REVISION
            with urllib.request.urlopen(url, timeout=300) as response, archive.open('wb') as output:
                shutil.copyfileobj(response, output)
            with zipfile.ZipFile(archive) as package:
                top = package.namelist()[0].split('/')[0]
                package.extractall(WORK)
            (WORK / top).rename(SRC)
            marker.write_text(SOURCE_REVISION)
        print('Mã nguồn:', SOURCE_REVISION)

        def run(args, log_name, cwd=SRC):
            \"\"\"Chạy lệnh, hiện log trực tiếp và lưu log ra Drive.\"\"\"
            started = time.time()
            with (OUT / log_name).open('a', encoding='utf-8') as log:
                log.write(f'\\n$ {{" ".join(map(str, args))}}\\n')
                process = subprocess.Popen(list(map(str, args)), cwd=cwd, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                code = process.wait()
            print(f'--- {{log_name}}: exit {{code}}, {{(time.time() - started) / 60:.1f}} phút')
            if code:
                raise RuntimeError(f'{{log_name}} lỗi (exit {{code}}), xem {{OUT / log_name}}')
        """, 2),
        cell("code", """
        # Thư viện: torch có sẵn trên Colab; ghim transformers cho V-JEPA 2.
        run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.1'], 'install.log')
        import torch, transformers
        assert torch.cuda.is_available(), 'Chọn GPU (L4/A100) trong Runtime → Change runtime type'
        print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| transformers', transformers.__version__)
        """, 3),
        cell("markdown", """
        ## V2 · Tải dữ liệu và tạo manifest

        Tải từng lớp Kinetics rồi chỉ giữ số clip cần (mặc định toàn bộ dương tính, 150 clip mỗi lớp âm),
        tải ASHEN theo split chính thức, lấy đúng các video Charades cần bằng HTTP Range.
        Chia train/dev/test không trùng video gốc (Kinetics theo `youtube_id`, Charades theo người quay).
        """, 4),
        cell("code", """
        run([sys.executable, '-u', '-m', 'extensions.vandalism.prepare_vandalism', '--root', DATA,
             '--neg-per-class', 150, '--neg-test-per-class', 30,
             '--ashen-normal-train', 60, '--ashen-normal-test', 60,
             '--charades-other-train', 1500, '--charades-other-test', 200], 'prepare.log')
        shutil.copy(DATA / 'prepare_summary.json', OUT / 'prepare_summary.json')
        print(json.dumps(json.loads((DATA / 'prepare_summary.json').read_text())['counts'], indent=1))
        print(shutil.disk_usage('/content'))
        """, 5),
        cell("markdown", """
        ## V3 · Trích feature V-JEPA 2 (đóng băng)

        Mỗi cửa sổ 8 khung → 64 token × 1024 (fp16). Clip ngắn tối đa 12 cửa sổ, video ASHEN dài tối đa 120.
        Feature đã có sẽ được bỏ qua khi chạy lại.
        """, 6),
        cell("code", """
        run([sys.executable, '-u', '-m', 'extensions.vandalism.train_vandalism', '--stage', 'features',
             '--data', DATA, '--cache', CACHE, '--batch', 32], 'features.log')
        """, 7),
        cell("markdown", """
        ## V3 · Train attentive probe

        Học multiple-instance (top-3 cửa sổ mỗi clip/video), dừng sớm theo AP trên dev.
        Ngưỡng được chọn trên dev: recall cao nhất mà precision ≥ 0.9 (không đạt thì lấy F1 tốt nhất).
        Checkpoint, báo cáo và file zip được ghi thẳng ra Drive.
        """, 8),
        cell("code", """
        run([sys.executable, '-u', '-m', 'extensions.vandalism.train_vandalism', '--stage', 'train',
             '--data', DATA, '--cache', CACHE, '--output', OUT, '--run-tag', RUN_TAG], 'train.log')
        """, 9),
        cell("code", """
        report = json.loads((OUT / 'report.json').read_text())
        print('Ngưỡng:', report['spec']['threshold'], '(', report['spec']['threshold_rule'], ')')
        print('Dev :', report['dev']['all_bags'])
        for source in ('kinetics700_2020', 'charades', 'ashen'):
            if source in report['test']:
                print(f'Test {source}:', {k: v for k, v in report['test'][source].items()})
        print('Báo nhầm theo lớp âm (test):', report['test'].get('false_alarm_by_negative_class'))
        print('\\nFile trên Drive:')
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                print(f'  {path.name:40s} {path.stat().st_size / 1e6:8.2f} MB')
        print('\\nBước tiếp theo: tải', f'vandalism_{RUN_TAG}.zip', 'về máy để cài (V4).')
        """, 10),
    ]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    if len(sys.argv) != 2 or len(sys.argv[1]) != 40:
        raise SystemExit("usage: build_vandalism_notebook.py <40-char source commit sha>")
    build(sys.argv[1])
