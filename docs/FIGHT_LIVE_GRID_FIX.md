# Đồng bộ upload và camera trên CPU — 28/10/2026

## Sửa thực tế

- Worker camera không còn gọi predict chỉ ở frame cuối mỗi batch. Mọi mẫu được đưa qua cùng bộ lập lịch stride của FightRuntime như upload; không đổi FPS, số frame, vùng người, threshold hay gate.
- Khi hàng đợi hữu hạn bị tràn, đóng sự kiện cũ, reset clip/gate và thu lại đủ dữ liệu; không nối chuyển động qua đoạn đã mất mẫu. API có `analysis_incomplete` và bộ đếm mất mẫu. Không tăng queue vô hạn hoặc che trạng thái stale.
- Dùng `torch.jit.freeze` trên bản model trong RAM để gộp các phép tính suy luận; không ghi đè trọng số trên đĩa. API báo `cpu_optimization=frozen`.
- Prefetch ba frame detector bằng một lần YOLO predict khi chúng cùng kích thước, tái sử dụng cache và giữ nguyên letterbox. API bật tối ưu này; các luồng tạo cache train mặc định vẫn tuần tự. API báo `pose_prefetch=true`.
- Validator kiểm toàn bộ timestamp/điểm và khoảng bắt đầu–kết thúc cảnh báo, không coi vài timestamp khớp hoặc số cảnh báo bằng nhau là đã đạt. Sai số thời gian kết thúc cho phép một bước lấy mẫu (1/12 giây trong gói này).

Tham khảo triển khai chính thức: [PyTorch freeze](https://github.com/pytorch/pytorch/blob/main/torch/jit/_freeze.py), [Ultralytics predict/batching](https://docs.ultralytics.com/modes/predict). Không thay bằng một model nhận dạng khác.

## Đối chứng local

Gói parent RGB control được chạy cách ly, không thay model của server, không mở camera thật. Cùng năm video public, cùng trọng số và ngưỡng trước/sau sửa:

| Chỉ số | Trước sửa | Sau sửa + frozen + prefetch |
|---|---:|---:|
| Số nguồn có số cảnh báo upload/replay giống nhau | 4/5 | 5/5 |
| Toàn bộ timeline điểm và khoảng cảnh báo khớp | Chưa đạt | 5/5 |
| Sai số điểm giữa upload/replay trên timestamp tương ứng | 0, nhưng thiếu nhiều timestamp | 0, đủ toàn bộ timestamp |
| TP/TN/FP/FN upload, nhãn cấp clip | 2/2/1/0 | 2/2/1/0 |
| TP/TN/FP/FN replay, nhãn cấp clip | 2/1/2/0 | 2/2/1/0 |
| P95 suy luận của clip fight AIRTLab, giây | 0.970 | 0.618 |
| Tuổi prediction lớn nhất quan sát, giây | 3.577 | 2.093 |

Đây là số đo của các lượt chạy trên máy này, không phải cam kết throughput. Batching có lợi ở một số nguồn, không nhanh hơn mọi nguồn. Vẫn có các lượt poll stale; **freshness/integration gate chưa đạt**. Không nâng ngưỡng stale, hạ threshold hoặc bỏ bớt cửa sổ để ghi kết quả đạt giả. Một báo nhầm của chính classifier vẫn tồn tại trong cả hai chế độ; việc loại cảnh báo thừa do scheduling không có nghĩa model đã phân biệt hoàn toàn ôm/bắt tay.

Log public cách ly: `logs/live-grid-batch-cpu-v1/cpu_acceptance.json`. Nhánh frozen không batch cũng đạt timeline parity nhưng vẫn trễ; nhánh frozen strict không gộp số học chậm hơn. ONNX Runtime được thử trên ba tensor public, sai số xác suất tối đa khoảng 7.75e-6 nhưng chậm hơn frozen PyTorch ở cả ba mẫu nên **không tích hợp**, không thêm dependency vào requirements. Xem [hướng dẫn tối ưu ONNX chính thức](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).

## Hai recording iPhone

Đã replay lại với parent, timestamps gốc, frozen và prefetch: kết quả nhận/bỏ sót không đổi. Tất cả recording, frame và log riêng vẫn ở local; không upload, không train, không chọn threshold theo chúng. Chưa có lần kiểm thử iPhone vật lý mới.

Do đó lượt này xử lý sự khác biệt do scheduling và giảm một phần chi phí CPU, **chưa khắc phục việc model bỏ sót recording thứ hai**, chưa chứng minh 80% camera, chưa tự cài parent hoặc model mới. Cần giải quyết dữ liệu dương/khả năng biểu diễn tương tác cận cảnh trước lượt fine-tune tiếp theo; không chạy lại cấu hình chỉ tăng cửa sổ âm đã không tiến triển.

Regression cuối: **226 test CPU qua**, gồm suy luận cố ý chậm hơn stride, reset sau mất mẫu, kiểm timeline đầy đủ, prefetch/caching và replay recording. SHA-256 của checkpoint đang cài vẫn là `7cd20e23e1590689a1d08bf63a9aa967c831a71dd985fed1a09fae5bd32b7d09`. Không dùng GPU, không training lại trong lượt sửa scheduling này.
