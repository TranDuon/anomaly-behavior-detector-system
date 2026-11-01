# Model đang được hỗ trợ

Cập nhật ngày 01/11/2026. Giao diện, API và CLI hiện hỗ trợ đúng ba backend:

| Backend | Model và dữ liệu đầu vào | File cần có |
|---|---|---|
| `legacy` | FightWatch X3D-S + pose, chạy CPU; tên backend giữ để tương thích đường chạy mặc định | `models/fight_model.ts`, `models/fight_runtime.json`, `models/yolov8n-pose.pt`, `yolov8n.pt` |
| `mmaction_original` | SlowOnly ResNet101 + Faster R-CNN ResNet50, RGB, môi trường GPU riêng | `mmaction2_demo/weights/action_slowonly_r101.pth`, `person_faster_rcnn_r50.pth` |
| `mmaction_interaction_v1` | Backbone và AVA gốc đóng băng, head tương tác v1 riêng | Hai trọng số MMAction2 gốc cùng `mmaction2_demo/weights/interaction.pt` và `interaction.json` |

`yolov8n.pt` và `yolov8n-pose.pt` là thành phần cần thiết của FightWatch, không phải hai lựa chọn model độc lập. Ba lựa chọn đều giữ cổng bật pilot trước khi nhận diện. Không có fallback tự động sang LSTM hoặc ST-GCN.

## Checkpoint interaction v1 từ Drive

- [Run nguồn](https://drive.google.com/drive/folders/1hk5eNnqToJezDlI2otsPOEFIWIBPzH0R): `mmaction_interaction_v1/runs/6a422ed9bd26264e33c509c1d5bfde4e590b8802`.
- [Gói ZIP đã tải và cài](https://drive.google.com/file/d/1WmmPIclF6cdwKSzhLR8F0AWqDFXPqhfi/view): SHA-256 `1f96266f805fd6d6c940d7d95f794e1ad459a356a54b1c5506e4c5ffbd64a258`.
- [Checkpoint](https://drive.google.com/file/d/1YzG1oRIgx3Iwr2sDExtdgEIPtu4UTy87/view): 2.133.047 byte, SHA-256 `8c5808db106aacd0f43b72a0e76178b2dec039e0a1766d996bb8374b7e4dddb7`.
- Thời gian cập nhật checkpoint trên Drive: **29/10/2026 02:25:27, UTC+7**. Thời gian copy/cài trên máy không được dùng làm thời điểm training.
- Đã cài lại từ ZIP bằng `mmaction2_demo/install_branch.py`; kiểm tra checksum, fingerprint backbone/preprocessing, state dict và báo cáo hoàn tất. Báo cáo được lưu tại `mmaction2_demo/results/interaction_v1_report.json`.
- Bản này hoàn tất 16 epoch; `approved_for_demo=false`, `real_phone_tested=false`. Đây là mốc phát triển tiếp, chưa phải xác nhận chất lượng thực tế.

## Các nhánh đã ngừng dùng trong ứng dụng

LSTM Autoencoder, ST-GCN, interaction v2, refinement v3 và warm-start v4 không còn được chọn qua giao diện/API/CLI. Hai backend cũ `mmaction_tuned` và `mmaction_tuned_refined` bị từ chối; gọi v1 bằng `mmaction_interaction_v1`.

Trọng số ST-GCN, LSTM, metadata ST-GCN và YOLOv8s-pose đã chuyển khỏi thư mục dự án sang `D:\digital-surveillance-model-backups\2026-11-01-retired-models` trên máy này. Bộ nguồn nghiên cứu, notebook và báo cáo cũ vẫn lưu để tham khảo; chúng không phải đường runtime được hỗ trợ. Không xóa dữ liệu hay sửa file trên Drive.

## Kiểm tra lại trên máy đã cài đủ trọng số

```powershell
.venv\Scripts\python.exe -m pytest tests/test_model_selection.py tests/test_browser_sync.py tests/test_fight_runtime.py tests/test_pose_prefetch.py -q -p no:cacheprovider
$env:PYTHONPATH = "$PWD\mmaction2_demo"
.venv\Scripts\python.exe -m pytest mmaction2_demo/test_interaction.py mmaction2_demo/test_warmstart_v4.py mmaction2_demo/test_launcher.py -q -p no:cacheprovider
$env:PATH = "$PWD\mmaction2_demo\env;$PWD\mmaction2_demo\env\Library\bin;$PWD\mmaction2_demo\env\Scripts;$env:PATH"
.venv\Scripts\python.exe scripts/verify_supported_models.py --source samples/scf_fight/fi061.mp4
```

Smoke check dùng một đoạn video local cho mỗi backend, kiểm tra có nạp model và trả dự đoán hữu hạn; xuất `logs/supported_models_smoke.json`. Đây không phải đo accuracy, so sánh chất lượng giữa các model hoặc kiểm thử camera.
