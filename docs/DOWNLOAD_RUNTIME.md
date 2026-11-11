# Chạy cùng bản local đã công bố

Code và các trọng số runtime hiện có được công bố cùng nhau. File trọng số được lưu bằng Git LFS; không cần lấy riêng từ Drive và không cần training lại.

## Lấy code và model

Máy cần Git và Git LFS. Chạy trong repo:

```powershell
git lfs install
git pull --ff-only origin main
git lfs pull
git lfs fsck
```

Nếu clone mới, chạy `git clone https://github.com/TranDuon/anomaly-behavior-detector-system.git`, vào thư mục repo rồi chạy `git lfs pull`. Không chỉ tải ZIP source từ GitHub: tùy thiết lập GitHub, ZIP có thể chứa pointer LFS thay vì trọng số thật.

## Cài môi trường và mở web

1. Nhánh MMAction2 hiện tại cần GPU NVIDIA tương thích CUDA. Chạy `mmaction2_demo/setup.cmd` để tạo môi trường GPU riêng và tải mã nguồn MMAction2 chính thức.
2. Chạy `run_local.cmd` để cài môi trường web và mở ứng dụng. Không sao chép `.venv` hoặc `mmaction2_demo/env` từ máy khác.
3. Chọn `Tách Đánh / Vật / Ngã / Chạy`, cho phép pilot để thử. Video upload mặc định phân tích đầy đủ rồi phát kết quả H.264; realtime là tùy chọn riêng.

`mmaction2_demo/weights/` có checkpoint AVA, detector người, các head pilot và JSON metadata/ngưỡng hiện có. `models/` có model tư thế và các artifact runtime FightWatch/LSTM cũ đã có trên máy; chúng không tự động thay thế backend được chọn. Nhánh vật hiện chỉ chẩn đoán, không phát cảnh báo. Đây vẫn là bản thử nghiệm, không cam kết cùng tốc độ hoặc chính xác trên mọi camera/máy.

Nguồn và checksum hai trọng số MMAction2 gốc nằm trong `mmaction2_demo/weights/provenance.json`. Việc công bố lại không thay đổi giấy phép hoặc điều kiện sử dụng của các model/thư viện nguồn.

Không công bố video riêng, dataset, log, `.env`, mật khẩu, cache huấn luyện hoặc môi trường Python. Các file đó không cần để chạy inference giống cấu hình này.
