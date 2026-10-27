# Lượt train lớn: RLVS + AIRTLab

## Nguồn và phạm vi

- [RLVS](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset): dùng nhãn clip do nhà phát hành cung cấp. Archive hiện tải có hai bản thư mục lặp; adapter chỉ dùng một bộ 2.000 video, không coi 4.000 file là 4.000 mẫu độc lập.
- [AIRTLab](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos): Bianculli et al., Data in Brief 33 (2020), DOI 10.1016/j.dib.2020.106587; nghiên cứu/giáo dục. Giữ tất cả clip âm khó (ôm, vỗ tay, vung tay…), chỉ chọn clip dương có fight/punch/kick/slap và không có club/stab/gunshot. Đứng sát, tranh cãi, can ngăn không mặc nhiên là đánh nhau.
- Không dùng NTU. Không thêm nhãn frame UBI đang có cảnh báo. Các tập từng xem nhiều lần không được gọi là final test mới.
- AIRTLab có hai góc quay cùng màn diễn, cùng phòng và diễn viên; hai camera chung nhóm. Đây không phải kiểm thử độc lập phòng/diễn viên, càng không chứng minh chất lượng với lớp học đông hoặc camera cầm tay.

## Thực nghiệm

1. Archive tải vào `MyDrive/SENTINEL/datasets/fight/`, kiểm CRC trước khi đổi tên `.partial`; chặn traversal/symlink và archive giải nén quá 40 GiB. Giải nén/cache RGB trên `/content`.
2. SHA256 và heuristic dHash ba mốc của RLVS gom clip trùng/gần trùng; AIRTLab gom hai camera cùng màn diễn. Hash không chứng minh loại hết source/actor overlap. Nhóm SHA256 trùng nhưng nhãn xung đột được cách ly, không đảo nhãn toàn bộ dữ liệu.
3. Chia nhóm theo seed 73: xấp xỉ 70/10/10/10% train/development/calibration/test, có cả hai lớp mỗi phần. Một cửa sổ trung tâm khoảng 1,875 giây/video; số chính xác, video loại vì ngắn và nhóm lưu manifest. Nhãn clip không đảm bảo từng cửa sổ có hành động dương, cần audit.
4. Audit mẫu 40 cửa sổ train/development ẩn nhãn/score, không xem test để chọn phương pháp. Audit mẫu không có nghĩa toàn bộ tập được người kiểm duyệt. Không tự ý đổi nhãn test sau khi thấy dự đoán.
5. X3D-S Kinetics400, tube ROI và toàn cảnh, 8 epoch probe + tối đa 8 fine-tune, early stopping, class weights và augmentation nhẹ. Checkpoint chọn bằng AP development; ngưỡng chọn trên calibration với mục tiêu precision và recall cùng ≥0,8. Nếu không có ngưỡng thỏa mãn, báo rõ không đạt.
6. Đóng băng ngưỡng trước test; report accuracy, precision, recall, F1, FPR, AP/AUC, confusion matrix và khoảng Wilson 95% (mô tả theo clip, chưa hiệu chỉnh phụ thuộc nhóm). Không thay hyperparameter sau khi xem test rồi vẫn gọi đó là test độc lập.
7. Budget cache/training 45 phút, giữ checkpoint epoch đã hoàn tất nếu hết giờ. Watchdog dừng tiến trình, không tự chạy lại vào output cũ. Lưu model, lịch sử, manifest, calibration, test predictions, log, bundle CPU và frozen features trên Drive. Video/cache RGB tạm có thể tạo lại từ archive.

## Local và camera

### Audit trước lượt chạy 27/10/2026

Manifest: train 1.577, development 216, calibration 228, test 202 cửa sổ từ các video khác nhóm; 1 clip quá ngắn và 2 clip trùng SHA256 nhưng nhãn xung đột được cách ly trước training. Train thực tế còn được giới hạn âm không thấy người ở khoảng 2%, không lọc development/calibration/test để làm đẹp điểm. Audit 40 mẫu train/development, ẩn nhãn/score: 31 có nhận xét, 9 chưa rõ; 6 nhận xét khác nhãn clip nhà phát hành. Nhận xét được chốt trước mapping; SHA256 `9963dbb505efd24fe5b37b85bf0a7a548b8572029f72e03d505edda5759793a2`, lưu `FIGHT_LARGE_AUDIT.json`.

Đây là nhận xét AI qua sáu frame, không đủ xác nhận nhãn sai: AIRTLab cố tình diễn động tác âm giống đánh nhau, còn cửa sổ trung tâm có thể bỏ lỡ hành động dương trong video. Không tự đổi nhãn/loại clip theo các nhận xét này; giữ nhãn publisher để chạy benchmark clip-level có kiểm soát, ghi rõ không phải tập active-action gold. Các điểm nghi vấn cần xem video đủ ngữ cảnh hoặc người kiểm duyệt trước nghiệm thu thực tế. Không suy 6/31 thành tỷ lệ nhiễu toàn tập.

- `run_local.cmd` cài bundle gồm TorchScript CPU + metadata/ngưỡng đúng cặp. Chưa nghiệm thu camera thật thì `approved_for_demo=false`, chỉ chạy pilot khi người dùng bật.
- Upload thường: xử lý hết file theo timestamp video, xuất video đánh dấu.
- Upload bật **Phát lại theo thời gian thực**: đọc file đúng nhịp và đưa qua reader/buffer camera. Khi CPU chậm có thể bỏ nhịp suy luận; giao diện hiển thị độ trễ. Đây là phép thử đường dữ liệu, không phải quay iPhone thật.
- Gate không ghép hai dự đoán điểm cao cách xa nhau thành bằng chứng liên tục; gián đoạn suy luận trên 1,5 giây phải xác minh lại. Clip bằng chứng ghi từng mẫu nguồn thay vì một frame mỗi lần suy luận.
- iPhone chưa kết nối nên chưa kiểm chứng ánh sáng, rung tay, che khuất, nhiều người và độ trễ mạng thực tế. Không thể bảo đảm trước 80% cho môi trường này từ accuracy của dữ liệu công khai.

## Kiểm thử

`tests/test_fight_large.py`: group split, camera-pair leakage, giữ cùng-scene âm/dương, lọc vũ khí, ZIP lỗi/traversal, ngưỡng không thể đạt mục tiêu, khoảng tin cậy, gate khi inference gián đoạn và processor upload so với phát lại camera có nhịp thực. Predictor giả kiểm đường dữ liệu, không đo độ chính xác AI; bundle thật cần kiểm CPU riêng sau training.
