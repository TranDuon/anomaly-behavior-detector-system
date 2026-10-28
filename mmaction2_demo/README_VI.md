# MMAction2 thử nghiệm độc lập

Tất cả nằm trong thư mục này: `repo/` (mã chính thức), `env/` (Python/GPU riêng), `weights/`, `results/` và hai file chạy. Không thay `.venv`, checkpoint hay server FightWatch cũ. Không training và không tải video riêng lên mạng.

- Repo: https://github.com/open-mmlab/mmaction2, bản phát hành ổn định v1.2.0.
- Model: SlowOnly ResNet101 OmniSource + Faster R-CNN ResNet50, đúng hai checkpoint mặc định của demo webcam chính thức. Đây là lựa chọn demo chính thức, KHÔNG khẳng định là model chính xác nhất toàn kho hoặc phù hợp nhất lớp học.
- Demo: `repo/demo/webcam_demo_spatiotemporal_det.py`, giữ nguyên mã upstream. `run_official.py` giới hạn hàng đợi đọc/hiển thị, giải mã file bằng một luồng và dùng bốn luồng CPU để tránh hết RAM trên máy 16 GB; không đổi trọng số, nhãn hoặc ngưỡng.
- Khi suy luận, bỏ đường dẫn proposals của tập validation bằng `--cfg-options test_dataloader.dataset.proposal_file=None`. Demo dùng vùng người từ detector trực tiếp, không cần tải dataset AVA để chạy.
- Nhãn mặc định có `fight/hit (a person)`, ôm, bắt tay; không hiện nhãn đá người 71. Không thêm luật cảnh báo hoặc tự hạ ngưỡng (giữ 0.4 mặc định demo).

## Chạy

Chỉ chạy khi cài đặt hoàn tất và có `results/install_verified.json`. Nếu chưa có, mở `setup.cmd`; lỗi tải mạng có thể chạy lại để tiếp tục các file `.partial`, không cần clone lại. Cài thư viện CUDA khoảng vài GB, thời gian phụ thuộc mạng. Việc clone xong không có nghĩa đã chạy model thành công.

1. Video: kéo file MP4 thả lên `run_video.cmd`, hoặc chạy file đó rồi nhập đường dẫn video. Xuất cửa sổ preview và video có nhãn trong `results/`.
2. Camera: mở DroidCam Client và kết nối iPhone trước. Chạy `run_camera.cmd`, nhập số camera (0 thường là laptop; DroidCam thường 1 hoặc 2, phải kiểm tra thực tế). Chỉ lúc bạn chạy lệnh này camera mới mở.
3. Nhấn Ctrl+C tại cửa sổ dòng lệnh để dừng demo/camera. Nhãn là dự đoán, không phải xác nhận chắc chắn hoặc hệ thống cảnh báo bạo lực đã nghiệm thu.

Chạy lại cài đặt nếu cần: `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1`. Tải vài GB thư viện CUDA từ nguồn chính thức và tạo env riêng. Môi trường phụ thuộc được cố định để tương thích Windows, không lấy tất cả thư viện phiên bản mới nhất.

Camera chạy GPU RTX 4060 local, không cần Colab và không đi qua web server localhost. CPU có thể chọn bằng `run_camera.cmd --device cpu` nhưng có thể chậm. Chưa đảm bảo phân biệt chính xác ôm/đánh nhau; phải quan sát nhãn và kiểm tra bằng video thực tế.

## Đã kiểm tra tại máy này

- Đã nạp đúng hai checkpoint, CUDA/ROIAlign hoạt động trên RTX 4060; thông tin ở `results/install_verified.json`.
- Đã chạy hết hai video public AIRTLab 48 và 109, xuất MP4 có khung/nhãn và đọc lại được; trạng thái ở `results/full_run_verified.json`.
- Hai video này chưa xuất nhãn `fight/hit (a person)` tại ngưỡng demo mặc định 0.4. Cài đặt chạy thành công KHÔNG phải bằng chứng nhận đánh nhau chính xác. Không hạ ngưỡng hoặc tuyên bố model đã tốt hơn FightWatch.
- Chưa mở hoặc kiểm thử camera iPhone vật lý. Các file `run_camera.cmd` và `run_video.cmd` chạy demo độc lập, không dùng trang localhost của FightWatch.
