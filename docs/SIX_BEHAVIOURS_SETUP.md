# Cài và chạy chế độ "Sáu hành vi" (thêm Phá hoại tài sản và Cầm vật sắc nhọn)

Chế độ **"Sáu hành vi: Đánh / Vật / Ngã / Chạy / Phá hoại / Cầm vật sắc"** (backend `mmaction_separate_actions_ext`)
chạy bốn nhánh đánh / vật / ngã / chạy như chế độ "Tách Đánh / Vật / Ngã / Chạy", cộng hai nhánh mở rộng chạy trong một
tiến trình Python riêng (`extensions/env`).

## Yêu cầu

- Windows 10/11, GPU NVIDIA có driver hỗ trợ **CUDA 12.4** (đã chạy trên RTX 4050 Laptop 6 GB).
- **Git LFS** (`git lfs install`), vì checkpoint nằm trong LFS.
- **Python 3.12** cho môi trường mở rộng (qua `py -3.12` hoặc `uv`), ngoài các môi trường mà hệ thống gốc đã cần.
- Mạng cho lần chạy đầu: nhánh phá hoại tải backbone V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`, khoảng 1.3 GB) từ Hugging Face.

## Các bước

1. **Lấy mã và checkpoint**

   ```powershell
   git clone https://github.com/TranDuon/anomaly-behavior-detector-system.git
   cd anomaly-behavior-detector-system
   git lfs pull
   ```

   Checkpoint hai nhánh mới đã nằm sẵn trong repo, không cần cài thêm:

   | Thư mục | Nội dung |
   |---|---|
   | `weights/extensions/vandalism/` | `vandalism.pt` (attentive probe), `vandalism_tail.pt` (4 block cuối V-JEPA 2 đã tinh chỉnh), `vandalism.json` (spec, ngưỡng 0.667), `report.json` |
   | `weights/extensions/sharp_object/` | `hand_classifier.pt` (ResNet50, bản hand_04), `yolo11s-pose.pt`, `object_veto.pt` (YOLO11s COCO lọc đồ vật thường), `weapon_world.pt` (YOLO-World đã gắn danh sách vũ khí), `sharp_object_spec.json` (ngưỡng 0.854), `evaluation.json` |

2. **Cài hệ thống gốc** như [README](../README.md) và [hướng dẫn MMAction2](../mmaction2_demo/README_VI.md).

3. **Tạo môi trường mở rộng** (không đụng tới `.venv` và `mmaction2_demo\env`):

   ```powershell
   powershell -ExecutionPolicy Bypass -File extensions\setup.ps1
   ```

   Script tạo `extensions\env` (Python 3.12), cài `requirements-extensions.txt` (torch 2.6 cu124, transformers 4.57,
   Ultralytics 8.4.183 cài `--no-deps`) rồi chạy thử worker.

4. **Kiểm tra hai nhánh nạp được**

   ```powershell
   '{"op":"close"}' | extensions\env\Scripts\python.exe extensions\worker.py --extensions vandalism,sharp_object
   ```

   Dòng đầu tiên in ra phải có `"ready": true`, cả `vandalism` lẫn `sharp_object` trong `"extensions"`, `"load_errors": {}`
   và `"device": "cuda:0"`.

5. **Cấu hình**: chép `.env.example` thành `.env` (giữ `STREAM_FPS_CAP=20`, `JPEG_QUALITY=75` để hình lên web đều nhịp).

6. **Chạy**

   ```powershell
   .\run_local.cmd
   ```

   Mở http://localhost:8000 → bật **Cho phép checkpoint pilot** → chọn **"Sáu hành vi: Đánh / Vật / Ngã / Chạy / Phá hoại /
   Cầm vật sắc"** → chọn camera hoặc video. Lần đầu có thể mất vài phút để tải V-JEPA 2.

## Kiểm tra nhanh

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_extension_isolation.py tests\test_fight_runtime.py
```

Clip thử: chạy một video có người cầm dao, hàng **Cầm vật sắc nhọn** phải lên đỏ; video đập phá đồ, hàng **Phá hoại tài sản**
phải lên đỏ.

## Cập nhật model

| Nhánh | Train | Cài bản mới |
|---|---|---|
| Cầm vật sắc nhọn | `colab/sharp_object_hand_v4.ipynb` (GPU L4, Run all) | `extensions\env\Scripts\python.exe -m extensions.sharp_object.install_sharp_object sharp_object_hand_XX.zip` |
| Phá hoại | notebook vandalism trong `colab/` | `extensions\env\Scripts\python.exe -m extensions.vandalism.install_vandalism <zip>` |

Installer kiểm tra file, ghi SHA-256 vào `install.json` và giữ bản trước ở `weights\extensions\<tên>.previous`.
Khởi động lại web sau khi cài.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Không có hàng Phá hoại / Cầm vật sắc | Đang chọn nhầm chế độ, hoặc chưa bật checkpoint pilot |
| `Extension worker unreachable` / nhánh mở rộng tắt | Xem `logs\extension_worker_*.log`; chạy lại bước 3–4 |
| Không thấy `weights\extensions\...\*.pt` hoặc file chỉ vài trăm byte | Chưa `git lfs pull` |
| Hình giật trên laptop yếu | Giữ `STREAM_FPS_CAP=20`; nhánh phá hoại (V-JEPA 2 ViT-L) là phần nặng nhất |
