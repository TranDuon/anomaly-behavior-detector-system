# Fine-tune MMAction2: chạy nhanh từ camera cố định

Notebook: [`colab/mmaction_fixed_running_ground_20261105.ipynb`](../colab/mmaction_fixed_running_ground_20261105.ipynb). Mở bằng Colab Pro, chọn GPU rồi chạy các cell theo thứ tự. Notebook lấy cả ba checkpoint đầu vào từ Google Drive đã dùng cho nhánh đánh/vật/ngã: `interaction.pt` của `fall_wrestle_pilot_01`, SlowOnly AVA và Faster R-CNN. Nếu file chưa hiện qua Drive mount, notebook dùng Google Drive API cùng tài khoản được cấp quyền; không cần tải file từ máy lên.

## Dataset được chọn

- Nguồn: [UCF-ARG](https://vision.eecs.ucf.edu/data/UCF-ARG.html), lấy các clip đã cắt trong [SPHAR](https://github.com/AlexanderMelde/SPHAR-Dataset) tại revision `6f2be47ef810865c1a15595d976e15169a0df4a1`.
- Chỉ dùng camera **ground** cố định: 48 `running` (dương), 47 `jogging` và 48 `walking` (âm). Một clip `jogging` bị hỏng đã loại; file tải được kiểm tra Git blob SHA-1. 12 diễn viên được tách theo người: 01–08 train, 09–10 development, 11–12 test.
- Không dùng các góc aerial, rooftop, clip phim/UCF101/HMDB51, CASIA cần xin quyền, hay VIRAT Ground có người quá nhỏ và thiếu nhãn chạy chậm đối chứng. Đây là chọn lọc theo camera demo, không phải trộn mọi video có chữ “running”.
- Video UCF-ARG quay ngoài trời. Với camera trong nhà như `vat.mp4`, tập này là mốc khởi đầu, chưa bảo đảm chất lượng tại cảnh demo. Nên quay thêm clip chạy nhanh/chạy chậm/đi bộ từ chính camera demo để đánh giá độc lập và fine-tune tiếp nếu cần.

## Cách train và đầu ra

MMAction2 SlowOnly AVA và detector người giữ nguyên. Đầu ra cũ đánh/vật/ngã cũng giữ nguyên tensor. Head chạy nhanh khởi tạo từ tầng ẩn của checkpoint cha, nhận đặc trưng AVA **từng người** cùng mức chuyển động trong vùng người, rồi học `running` so với `jogging/walking`. Chọn epoch bằng AP trên development; chọn ngưỡng trên development; test chỉ dùng sau khi đã chốt. Bộ test có 8 clip chạy nhanh, 8 chạy chậm và 8 đi bộ, nên các chỉ số vẫn là pilot.

Notebook lưu feature cache, log, `report.json`, `interaction_fixed_running.pt`, `interaction_fixed_running.json` và `mmaction_fixed_running_pilot.zip` vào `MyDrive/SENTINEL/artifacts/mmaction_fixed_running/<RUN_TAG>/`. Checkpoint cha không bị ghi đè. Điểm chạy nhanh là phân loại dáng chạy và chuyển động tương đối, không phải tốc độ km/h.

## Test local

Cài gói cha bằng `mmaction2_demo/env/python.exe mmaction2_demo/install_fall_wrestle.py <parent.zip>`, sau đó cài gói mới bằng `mmaction2_demo/env/python.exe mmaction2_demo/install_fixed_running.py <running.zip>`. Installer kiểm tra SHA-256, preprocessing, báo cáo hoàn tất, `weights_only=True` và tensor cha giống hệt bản đã xác minh. Các checkpoint được lưu riêng trong `mmaction2_demo/weights/`, không thay interaction v1 hay AVA gốc.

Trên web, chọn **MMAction2 đánh / vật / ngã chung** để test bản cha hoặc **MMAction2 đánh / vật / ngã + chạy · 09/11** để test bản mới, bật cho phép pilot rồi chọn video/camera. Bản mới giữ hai điểm và hai bộ xác nhận thời gian riêng; cảnh chạy không bị gắn nhãn đánh nhau. Nhánh chạy chỉ chấm từng người với detector threshold 0.8 như lúc train; bản cha dùng detector threshold 0.9. Vì tập người được phát hiện có thể khác, điểm đánh/ngã của hai chế độ không nhất thiết giống nhau mặc dù tensor cha không đổi. Chưa nghiệm thu camera lớp học; cần giữ camera cố định, đủ người trong khung hình.

Nhánh chạy có chốt điều kiện runtime v1: ghép được cùng người giữa frame hành vi và frame mới; bỏ cảnh cận bị cắt thân người, box không phù hợp và không có dịch chuyển thân/đổi tỷ lệ đủ rõ. Cần duy trì bằng chứng 0.8 giây để phát cảnh báo chạy; không đổi ngưỡng hay thời gian xác nhận đánh/ngã. `running_raw_score`, `running_score` sau chốt và `running_evidence` được ghi trong trace để phân biệt model báo nhầm và runtime từ chối xác nhận. Đây là heuristic bảo thủ, không phải pose xác minh đủ chân hay bộ hiệu chuẩn mới; có thể bỏ sót người chạy tại chỗ/cận cảnh. Không dùng các clip kiểm tra lỗi này để tuyên bố accuracy test độc lập.
