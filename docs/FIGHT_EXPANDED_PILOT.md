# Pilot mở rộng: nhóm đông người, không dùng NTU

Phạm vi demo mong muốn: nhóm đông trong lớp học, hành lang và ngoài trời, quay iPhone cầm tay. Đây là mục tiêu kiểm thử, không phải chất lượng đã đạt. Không tiếp tục đăng ký/tải NTU. Dùng archive UBI-Fights đang có trên Drive; chưa xác minh public footage tương đương mọi cảnh demo.

## Thay đổi so với people-focus

- Không loại mẫu đánh nhau chỉ vì detector không thấy hai người: cách đó đã bỏ nhiều mẫu khó và làm validation quá nhỏ.
- Mở rộng có kiểm soát tới 200 video nguồn, tối đa 24 cửa sổ/video; một nửa quota mỗi video dành cho đánh nhau và một nửa cho bình thường. Video bình thường chỉ có quota âm. Mỗi cửa sổ có ít nhất 70% frame đánh nhau hoặc hoàn toàn bình thường, không chồng nhau trong cùng video.
- Giữ toàn bộ mẫu âm có người, kể cả đông người không đánh nhau; nền không phát hiện người tối đa 5% train sau chọn. Detector chưa phân loại được bối cảnh lớp học/hành lang, nên không tự gán nhãn môi trường.
- Bỏ viền đen ngoài ảnh bằng ba frame, có chặn crop quá nhỏ. Ghép bbox qua ba frame thành vùng bao chuyển động, tránh ba bbox của cùng người được coi như ba người. Nhóm gần nhau vẫn có thể gộp cả đám đông: chưa phải bộ tracking tương tác hoàn chỉnh, cần xem source/crop trong audit.
- Train và inference gọi cùng `build_clip_inputs`; giữ thời gian 16 frame/8fps để baseline và model mới nhìn cùng cửa sổ. Fine-tune mô phỏng dịch/rung/xoay nhẹ cho cả hai lớp. Không coi augmentation là chứng minh độ chính xác trên điện thoại thật.

## Chống rò rỉ và đối chứng

Chỉ lấy official train, giữ official test riêng. Validation nguồn cũ không vào train; train nguồn cũ không chuyển sang validation mới. Nhóm scene hash gần nhau được nối bắc cầu, đồng thời giữ nhóm cảnh cũ. Nhóm nối cả train và validation cũ bị cách ly. Scene hash là kiểm tra gần trùng xấp xỉ, không chứng minh mọi scene độc lập.

Baseline v2 interaction và model mới được chấm lại trên chính các clip validation mở rộng, cùng thời gian và nhãn. Không so AP mới trực tiếp với 109 clip cũ hoặc 24 clip people-focus. Đây là development validation dùng chọn model/threshold, chưa phải test nghiệm thu độc lập.

## Thao tác và lưu trữ

Mở `colab/fight_expand_drive.ipynb`, chọn L4 rồi chạy từng cell. Sync GitHub bỏ thay đổi tracked trong bản repo Drive như quy trình local-push đã thống nhất, không xóa dataset/artifact. Chỉ copy file tracked sang `/content`, không copy toàn dataset theo repo.

- Input: `MyDrive/SENTINEL/datasets/fight/UBI_FIGHTS.zip`.
- Artifact: `MyDrive/SENTINEL/artifacts/fight_expand_v1/`.
- Video subset: `/content/fight_expand_data/UBI_FIGHTS`.
- Cache RGB tạm: `/content/fight_expand_cache`.
- Drive giữ manifest, đặc trưng pretrained nén, checkpoint, history, baseline, predictions, report, clip lỗi và ZIP runtime CPU. Đặc trưng nén không thay thế RGB cache khi fine-tune lại. Xóa runtime mất RGB/subset nhưng không mất các artifact Drive.

Bộ đếm pilot 45 phút bắt đầu tại cell GPU/install. Chuẩn bị có timeout 15 phút, train nhận phần budget còn lại. Nếu hết thời gian giữa epoch thì khôi phục best checkpoint từ epoch hoàn tất; nếu chưa đủ dữ liệu/cache/epoch thì dừng inconclusive, không tự nối thêm nhiều giờ. Audit video lỗi là bước hậu xử lý riêng, timeout 3 phút. Runtime vẫn có thể giữ GPU sau khi tiến trình dừng; cần ngắt trong Colab khi không dùng nữa.

## Tiêu chí đọc kết quả

Xem precision/recall/F1 cùng threshold, AP, ROC-AUC, median in-video AUC và số video có cả hai nhãn. Recall@precision75 chỉ là chỉ số thăm dò trên validation hiện tại. `promising` yêu cầu recall@precision75 >=0.6, in-video AUC >=0.65 trên ít nhất ba video mixed và AP hơn baseline cùng cohort. `approved_for_demo` vẫn False; không tự đưa checkpoint vào sản phẩm.

Nếu không cải thiện, phải xem video nguồn/crop của báo nhầm và bỏ sót để phân biệt lỗi vùng nhìn, nhãn, thiếu hard negatives hoặc khả năng học thời gian; không chỉ tăng epochs. Người bị che khuất hoàn toàn hoặc hành động ngoài khung không thể được đảm bảo nhận biết chỉ bằng làm nét camera. Demo thực tế vẫn cần được kiểm thử trước khi khẳng định đạt.
