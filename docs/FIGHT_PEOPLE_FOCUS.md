# Ưu tiên bối cảnh có người — people_focus_v1

Đã chạy trên Colab; xem [kết quả và giới hạn](FIGHT_PEOPLE_RESULTS.md). Không kết luận mô hình tốt hơn chỉ vì thay đổi cohort.

Không gán nhãn giao thông theo tên video hay giả định có xe là nhiễu. Bộ lọc tự động chỉ xác định có người đủ rõ để phù hợp demo; không phân loại xe. Người đánh nhau ở ngoài đường vẫn thuộc phạm vi, đám đông đi bộ bình thường vẫn là mẫu âm quan trọng.

## Quy tắc khóa trước khi xem kết quả

- Dùng đúng 380 clip/split cũ, không đưa official test vào lựa chọn phương pháp.
- YOLO person chạy trên đầu/giữa/cuối clip, cùng cấu hình runtime: confidence 0,4, NMS IoU 0,45, size 640.
- Người đủ rõ: bbox cao ít nhất 18% chiều cao ảnh và diện tích ít nhất 0,8% ảnh. Cần ít nhất hai người đủ rõ trong ít nhất hai trong ba frame được lấy mẫu. Không cộng box của ba thời điểm thành số người.
- Quyết định lọc không đọc nhãn fight/normal hoặc điểm phân loại. Mỗi clip giữ hoặc loại đều có lý do và ảnh ba frame với bbox để kiểm tra. Detector có thể bỏ sót người bị che/gần quá; không gọi các clip bị loại là không có đánh nhau.
- Train chủ yếu là tương tác nhiều người, giữ cả dương/âm. Thêm tối đa 15% clip train âm có một người đủ rõ để model không nhầm một người vận động. Không thêm control từ validation.
- Validation mục tiêu chỉ gồm cảnh nhiều người; giữ nguyên ranh giới video/scene-group từ split cũ. Báo tỷ lệ giữ/loại theo cả nhãn và split, không giấu positive khó.

## So sánh công bằng

Đây là thay đổi phạm vi đánh giá theo mục tiêu demo, không phải số đo cũ tự nhiên tăng lên. Tính lại baseline v1, interaction v2 và hard-negative v2 trên cùng subset mới; đồng thời vẫn đánh giá model mới trên toàn bộ 109 clip validation gốc ở cùng ngưỡng của people cohort. Official test không dùng; kết quả vẫn là development validation đã dùng nhiều lần.

Cần ít nhất 12 clip mỗi lớp trong train và 8 mỗi lớp trong validation. Nếu không đủ, báo inconclusive và mở rộng nguồn train/audit; không âm thầm nới quy tắc lọc hoặc xóa nhãn khó để đạt điểm. Pipeline không đặt luật nhiều người = đánh nhau, không tự duyệt checkpoint demo.

## Artifact

`SENTINEL/artifacts/fight_people_focus_v1/` chứa `domain_audit.json`, `gallery.json`, ảnh `previews/`, `cohort_manifest.json`, `cohort_report.json`, baseline trên cùng cohort, history/checkpoint/predictions/errors và runtime nếu pilot train đủ dữ liệu. Dùng lại cache interaction của v2; không thay đổi preprocessing/model runtime để tạo lợi thế đánh giá. Budget script 12 phút, không train toàn dataset.
