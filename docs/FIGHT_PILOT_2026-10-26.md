# Pilot X3D-S — 26/10/2026

## Phạm vi và nguồn kết quả

- Run: `fight_pilot_x3d_v1`, GPU Colab NVIDIA L4.
- Code training: `9f089ca5ada44042c4693fd805b80a692f03fe34`.
- Code phục hồi export: `1aba47c30d79a39eea1bd84f4fa09122a23e008e`.
- 48 video nguồn UBI-Fights, 380 clip đã cache; 271 train, 109 validation.
- Chỉ dùng phần train chính thức; official test không tham gia chọn checkpoint/ngưỡng.
- Chia theo video/nhóm near-duplicate xấp xỉ. Chưa kiểm toán sạch mọi camera/cảnh.
- Số đo là trên clip được chọn, không phải toàn video, frame-level hay sự kiện realtime.
- Artifact trên Drive: `SENTINEL/artifacts/fight_pilot_x3d_v1/`.

## Kết quả

Checkpoint tốt nhất là probe epoch 4; ba epoch fine-tune ngắn không vượt được checkpoint đó.

| Chỉ số validation | Kết quả |
| --- | ---: |
| ROC AUC gộp | 0,7100 |
| Average precision | 0,4487 |
| AUC trong video (7 video có cả hai nhãn) | 0,7500 |
| Precision tại ngưỡng validation | 33,93% |
| Recall | 76,00% |
| F1 | 0,4691 |
| Tỷ lệ dương tính giả | 44,05% |
| Ngưỡng được chọn | 0,1137288 |
| AUC trong video của baseline chuyển động | 0,2500 |

Với 25 clip dương và 84 clip âm, tương ứng TP=19, FN=6, FP=37, TN=47. Trong 56 clip bị báo đánh nhau, có 37 clip báo nhầm. Sanity overfit 8 clip đạt nhưng chỉ xác minh đường học, không chứng minh khả năng tổng quát.

**Quyết định: `stop_and_review`, `approved_for_demo=false`.** Model có tín hiệu tốt hơn baseline trên mẫu này nhưng chưa đạt precision >=75%; không được gọi là model nhận diện đánh nhau chính xác. Không chạy train nhiều giờ chỉ dựa trên kết quả này.

## Thời gian và gói runtime

Training/log ban đầu chạy khoảng 16:11:50–16:14:36 UTC (bao gồm cache, đánh giá và lỗi export), không phải một phiên train toàn dataset. Export lần đầu lỗi custom `SwishFunction` của PyTorchVideo. `best.pt` đã được lưu trước lỗi nên không mất training.

Đã thay Swish bằng phép toán tương đương có thể lưu TorchScript, thêm test ghi–nạp X3D thật và chạy `--evaluate-only` trên checkpoint cũ. Phục hồi đánh giá/export mất khoảng 12,4 giây, không train lại.

- ZIP runtime: 11.576.842 byte, chỉ có `models/fight_model.ts`, `models/fight_runtime.json`.
- CPU local đã tải, nạp và chạy forward thành công; chế độ mặc định chặn checkpoint chưa duyệt, chế độ pilot cho phép thử chủ động.
- Đo một forward sau warm-up, 4 luồng CPU local: khoảng 0,563 giây/vùng. Đây chỉ là classifier, không bao gồm YOLO, đọc/ghi video và các vùng khác; chưa đủ kết luận realtime.
- 60 test local qua với thư mục tạm mới; thư mục temp mặc định của máy có lỗi quyền.
- `execution_failure.json` trong artifact là dấu vết lỗi export ban đầu; kết quả phục hồi nằm trong `export_recovery_log.txt` và `report.json`.

## Việc cần làm trước lần training tiếp theo

1. Xem lại clip theo timestamp trong `predictions.json`, ưu tiên 37 false positive và 6 false negative; ảnh tĩnh audit chỉ kiểm tra crop sơ bộ.
2. Kiểm tra vùng tương tác có giữ đủ người/ngữ cảnh trong suốt clip, đặc biệt cảnh đông người, video chèn khung và cảnh gần.
3. Kiểm tra nhãn/tốc độ lấy mẫu, giữ nguyên validation và official test; không chỉnh nhãn theo mong muốn để nâng điểm.
4. Chỉ chạy pilot mới sau khi có giả thuyết sửa cụ thể, so sánh trên cùng split và baseline. Sau khi đạt mới thiết kế train lớn, test độc lập và đo sự kiện/độ trễ video–camera iPhone.

Camera iPhone chưa được kiểm thử trên thiết bị thật; không coi hỗ trợ nguồn camera trong giao diện là bằng chứng chất lượng mô hình.
