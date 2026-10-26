# Kết quả đối chứng v2 — 27/10/2026

Training thực tế: commit `636f3b9fa83cbebd36bd1bf45e4efc5cc845df82`, NVIDIA L4, 673,06 giây (~11 phút 13 giây), trong budget 25 phút. Artifact: `SENTINEL/artifacts/fight_ablation_v2/`. Số liệu máy đọc được: [experiments/fight_ablation_v2.json](experiments/fight_ablation_v2.json).

## Kết quả trên cùng 109 clip validation

| Nhánh | Train clips | AP | ROC AUC | AUC trong video | Precision (*) | Recall (*) | Recall tối đa khi precision >=75% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline v1 | 271 | 0,4487 | 0,7100 | 0,7500 | 33,9% | 76% | 4% |
| Interaction v2 | 271 | 0,4666 | 0,6638 | 0,5000 | 25,8% | 92% | 24% |
| Full-frame v2 | 271 | 0,4195 | 0,6419 | 0,5000 | 28,4% | 84% | 0% |
| Full-frame + hard negatives | 303 | 0,4697 | 0,6567 | 0,4375 | 29,0% | 72% | 36% |

(*) Chọn F1 cao nhất trong các ngưỡng có recall >=70%. V1 dùng lưới ngưỡng, v2 dùng mọi ranh giới score; vì vậy phải xem cả AP/đường precision–recall, không chỉ ngưỡng được chọn. Baseline recall@precision75 được tính lại từ đúng `predictions.json` cũ trên mọi ranh giới score.

**Chưa có nhánh nào đạt precision >=75% và recall >=60%. Không thay model demo; toàn bộ bundle vẫn `approved_for_demo=false`.** AP tốt nhất tăng 0,0210, nhưng ROC AUC và AUC trong video giảm. Có cải thiện ở vùng vận hành ít báo nhầm của nhánh hard-negative (recall 36% so với 4%), vẫn bỏ sót 16/25 clip dương. Không coi một chỉ số tăng nhỏ là chứng minh phương pháp đã tốt hơn toàn diện.

Ở ngưỡng 0,5, nhánh hard-negative có precision 60%, recall 36%, FPR 7,14% (TP=9, FP=6, FN=16, TN=78). Nâng ngưỡng chỉ đổi trade-off, không giải quyết được thiếu khả năng phân biệt.

## Những gì đã kiểm tra/sửa

- Split/video/timestamp/nhãn validation giữ nguyên, official test không tham gia. Đây là development validation đã dùng nhiều lần; không phải kết quả test độc lập.
- Detector/preprocessing được dùng chung với local; weights, confidence, NMS IoU=0,45 và image size được lưu trong metadata bundle.
- Batch hiệu dụng 8 bằng tích lũy gradient; learning rate backbone/head tách riêng, gradient clipping, cosine schedule, giữ BatchNorm eval; chọn checkpoint theo AP.
- Chọn 32 mẫu âm khó trong 96 cửa sổ ứng viên không chồng lấn từ video train. Không thêm video/clip validation vào train.
- Xuất clip gốc và crop của cả lỗi dương giả/âm giả, kèm timestamp/score, H.264 để xem trong trình duyệt. Tối đa 6 clip cho mỗi loại lỗi mỗi nhánh; toàn bộ lỗi có trong JSON.
- Kiểm toán 48/48 video: số frame nhãn khớp hoàn toàn số frame video; không thấy lỗi lệch độ dài annotation. Điều này không chứng minh nội dung nhãn luôn đúng.
- Giữ nguyên 16 frame/8 FPS để đối chứng không đổi cửa sổ/nhãn; chưa thử lại cấu hình thời gian pretrained.
- 67 test local qua sau khi bổ sung kiểm toán nhãn. Bundle hard-negative thực tế đã được tải và chạy forward CPU, xác minh metadata full-frame/IoU=0,45; chế độ mặc định vẫn chặn checkpoint chưa duyệt. Chỉ lưu thử trong `logs/`, không cài vào `models/` làm model demo.

## Quan sát từ clip lỗi

- `N_725`, khoảng giây 76: đám đông đi lại tại sự kiện; crop tách ra một số người đi bộ và làm mất bối cảnh rộng. Đây là ví dụ báo nhầm chuyển động bình thường.
- `F_157`, khoảng giây 52: khung camera nhà hàng nằm trong viền đen lớn; người nhỏ/mờ khi giảm về 182 px. Cần audit vùng nội dung và độ phân giải hiệu dụng, không chỉ tăng epoch.
- `F_192`, khoảng giây 12: tương tác nằm sát góc dưới-trái, phần lớn khung là cầu thang trống. Khó quan sát/che khuất là rủi ro thực, không thể hứa model luôn xử lý đúng.
- `F_21`, khoảng giây 38: có người nằm đất và người xung quanh nhưng annotation ở cửa sổ này là 0; vùng positive tiếp theo bắt đầu ở giây 48. Đây là trường hợp cần xem động tác trong video đầy đủ và quy ước nhãn, không tự sửa nhãn validation vì model báo khác.

Quan sát trên chỉ là audit một số lỗi, không phải kết luận mọi lỗi đều do crop hoặc dataset sai.

## Hướng tiếp theo

1. Giữ mọi số đo hiện tại, không tăng epoch/đổi ngưỡng để làm đẹp điểm và không thay checkpoint đang dùng bằng nhánh v2 chưa đạt.
2. Thử riêng xử lý viền đen/vùng nội dung và crop theo nhóm người qua thời gian; lưu clip audit để kiểm tra có giữ đủ người/ngữ cảnh. Không dùng luật 'có hai người gần nhau' làm nhãn đánh nhau.
3. Tăng độ đa dạng và số clip của phần train, thêm mẫu âm khó từ chính train. Ba nhánh hiện tại vẫn chỉ có 271–303 clip train; kết quả không chứng minh đã khai thác hết X3D/UBI-Fights.
4. Thử recipe thời gian pretrained trong một thí nghiệm mới, kiểm tra lại nhãn cho cửa sổ mới, thay vì âm thầm thay duration trong so sánh hiện tại.
5. Sau khi có tiến bộ ổn định mới đánh giá độc lập, đo cảnh báo theo sự kiện/toàn video, CPU local và camera iPhone. Không chuyển ngay sang training nhiều giờ trên cùng thiết lập chưa đạt.

Runtime Colab được giữ mở theo yêu cầu người dùng để xem báo cáo; GPU chưa được giải phóng tự động.
