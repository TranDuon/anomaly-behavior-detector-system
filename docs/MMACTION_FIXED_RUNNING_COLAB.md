# Fine-tune MMAction2: chạy nhanh từ camera cố định

Notebook: [`colab/mmaction_fixed_running_ground_20261105.ipynb`](../colab/mmaction_fixed_running_ground_20261105.ipynb). Mở bằng Colab Pro, chọn GPU rồi chạy các cell theo thứ tự. Notebook lấy cả ba checkpoint đầu vào từ Google Drive đã dùng cho nhánh đánh/vật/ngã: `interaction.pt` của `fall_wrestle_pilot_01`, SlowOnly AVA và Faster R-CNN. Nếu file chưa hiện qua Drive mount, notebook dùng Google Drive API cùng tài khoản được cấp quyền; không cần tải file từ máy lên.

## Dataset được chọn

- Nguồn: [UCF-ARG](https://vision.eecs.ucf.edu/data/UCF-ARG.html), lấy các clip đã cắt trong [SPHAR](https://github.com/AlexanderMelde/SPHAR-Dataset) tại revision `6f2be47ef810865c1a15595d976e15169a0df4a1`.
- Chỉ dùng camera **ground** cố định: 48 `running` (dương), 47 `jogging` và 48 `walking` (âm). Một clip `jogging` bị hỏng đã loại; file tải được kiểm tra Git blob SHA-1. 12 diễn viên được tách theo người: 01–08 train, 09–10 development, 11–12 test.
- Không dùng các góc aerial, rooftop, clip phim/UCF101/HMDB51, CASIA cần xin quyền, hay VIRAT Ground có người quá nhỏ và thiếu nhãn chạy chậm đối chứng. Đây là chọn lọc theo camera demo, không phải trộn mọi video có chữ “running”.
- Video UCF-ARG quay ngoài trời. Với camera trong nhà như `vat.mp4`, tập này là mốc khởi đầu, chưa bảo đảm chất lượng tại cảnh demo. Nên quay thêm clip chạy nhanh/chạy chậm/đi bộ từ chính camera demo để đánh giá độc lập và fine-tune tiếp nếu cần.

## Cách train và đầu ra

MMAction2 SlowOnly AVA và detector người giữ nguyên. Đầu ra cũ đánh/vật/ngã cũng giữ nguyên tensor. Head chạy nhanh khởi tạo từ tầng ẩn của checkpoint cha, nhận đặc trưng AVA **từng người** cùng mức chuyển động trong vùng người, rồi học `running` so với `jogging/walking`. Chọn epoch bằng AP trên development; chọn ngưỡng trên development; test chỉ dùng sau khi đã chốt. Bộ test có 8 clip chạy nhanh, 8 chạy chậm và 8 đi bộ, nên các chỉ số vẫn là pilot.

Notebook lưu feature cache, log, `report.json`, `interaction_fixed_running.pt`, `interaction_fixed_running.json` và `mmaction_fixed_running_pilot.zip` vào `MyDrive/SENTINEL/artifacts/mmaction_fixed_running/<RUN_TAG>/`. Checkpoint cha không bị ghi đè. **ZIP này chưa tự thêm lựa chọn vào web**; sau khi xem kết quả Colab và thử video camera, cần cài và nối nhánh chạy nhanh vào worker/UI để demo. Điểm chạy nhanh là phân loại dáng chạy và chuyển động tương đối, không phải tốc độ km/h.
