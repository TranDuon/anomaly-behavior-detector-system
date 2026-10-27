# FightWatch: quy trình thử trước khi train lớn

Ứng dụng chính chỉ phát hiện đánh nhau. LSTM/Ped2 và luật chạy nhanh, đứng yên, gần nhau không còn được gọi bởi CLI hoặc API chính. Các module/nghiên cứu cũ được giữ để đối chiếu kết quả, không dùng làm fallback.

## Chạy local

Chạy `run_local.cmd`, mở http://localhost:8000. Chưa có checkpoint mới vẫn mở được giao diện và xem thử camera. Không tạo cảnh báo từ model cũ.

- Video: chọn file, bật chế độ pilot nếu cần, bấm Phân tích. File kết thúc thì dừng, không tự lặp lại.
- iPhone: cài DroidCam trên iPhone và client chính thức trên Windows. Kết nối USB (cần Apple Mobile Device Support nếu Windows chưa nhận) hoặc cùng Wi-Fi. Trong app bấm Quét camera, chọn camera rồi Xem thử. Cần kiểm tra độ phân giải thực nhận; bắt đầu 720p nếu thiết bị/gói ứng dụng hỗ trợ.
- IP stream: dùng URL RTSP/HTTP trực tiếp từ ứng dụng camera, không phải URL trang web xem camera.
- Nút Cho phép checkpoint pilot phải bật chủ động. Điểm model là điểm phân loại, chưa được hiệu chỉnh thành xác suất thực tế.

Khi bấm **Bật nhận diện**, khung xanh `PERSON` là người detector tìm thấy; `PERSONS` đếm người ở frame cuối của lượt chấm. Khung vàng là vùng đang xác minh, đỏ là vùng cảnh báo đánh nhau; không có nghĩa mọi người trong vùng đều tham gia. Khung cập nhật theo lượt phân tích, không phải tracking từng frame. Tái dùng detection có sẵn, không thêm lượt chạy YOLO. **Xem thử** chỉ truyền hình, không chạy model/vẽ khung người. Trạng thái **Chưa xác nhận đánh nhau** không chứng minh hành vi an toàn: model vẫn có thể bỏ sót. Lỗi khung hiển thị và lỗi phân loại hành vi phải kiểm tra riêng.

Kết quả gồm mốc sự kiện, ảnh, clip bằng chứng (tối đa 30 giây mỗi phần) và video đã đánh dấu. File lưu trong `uploads/evidence/`. Nút xuất JSON lưu báo cáo phiên. Nhiều frame của một sự kiện không tạo nhiều cảnh báo bắt đầu.

## Notebook Colab

[Mở pilot trên Colab](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/fight_pilot_drive.ipynb)

Chọn GPU, chạy tất cả cell. Drive mount/OAuth phải được chủ tài khoản xác nhận. Notebook:

1. Mount `MyDrive/SENTINEL`.
2. Đồng bộ đúng repo `TranDuon/anomaly-behavior-detector-system` trên Drive từ `origin/main`. Bản sao code chưa commit trên Drive bị thay thế; dữ liệu/artifact ở ngoài repo được giữ.
3. Chép code sang `/content/fightwatch-code`, lưu commit vào artifact.
4. Xác nhận GPU, cài dependency và test logic.
5. Chỉ giải nén 48 video cân bằng từ phần train của UBI-Fights đã có. Không tải lại archive 8,12 GB và không trích toàn bộ ~80 giờ.
6. Tạo manifest tối đa 8 clip/video, có nhãn đánh nhau và bình thường trong cùng video khi có. Các frame ranh giới không rõ bị bỏ.
7. Nhóm near-duplicate theo dHash trước khi chia train/validation. Đây là xấp xỉ, không chứng minh đã tách sạch mọi camera; phải xem ảnh audit. Official test không tham gia chọn model.
8. Trích vùng tương tác theo cùng code runtime, lưu cache có fingerprint của config/backbone.
9. Thử overfit 8 clip để kiểm tra đường học; sau đó khôi phục head, train probe ngắn và fine-tune hai block cuối trên train.
10. Chọn checkpoint/ngưỡng bằng validation. Xuất AUC trong video, AUC gộp, precision, recall, FPR, baseline chuyển động, ảnh crop và dự đoán có timestamp.

Ngân sách mặc định 30 phút từ lúc chạy script (chuẩn bị/cài đặt tách riêng). Script kiểm tra deadline giữa các bước; watchdog notebook kết thúc tiến trình khi vượt ngân sách + 120 giây cho đánh giá/xuất. Nếu không đủ kết quả thì báo `inconclusive`, không tự coi là thành công. Nếu đã có `best.pt`, cell cấu hình tự đặt `EVALUATE_ONLY=True`: chỉ đánh giá/xuất checkpoint, không train lại. Đổi số video/clip hoặc seed thì đổi RUN_NAME; đổi backbone thì dùng cache tương ứng.

## Cổng quyết định pilot

`promising` đòi hỏi: ít nhất 3 video validation có cả hai nhãn; AUC trong video >= 0,65 và hơn baseline >= 0,05; precision >= 0,75; recall >= 0,60. Đây là tiêu chí sàng lọc phương pháp, không phải nghiệm thu sản phẩm. `stop_and_review`: xem crop/nhãn/lỗi trước khi mở rộng. `inconclusive`: chưa đủ dữ liệu hay hết budget.

Một probe đóng băng kém chưa chứng minh backbone không dùng được; pilot có thêm fine-tune ngắn. Thất bại sanity chỉ ra pipeline/đặc trưng nhỏ chưa học được và dừng trước khi tốn nhiều GPU.

Kết quả nằm trong `SENTINEL/artifacts/fight_pilot_x3d_v1/`:

- `manifest.json`, `cached_manifest.json`, `selected_videos.json`, `code_revision.txt`
- `sanity.json`, `history.json`, `best.pt`, `report.json`, `run_log.txt`
- `audit_previews/`, `audit_previews.json`, `predictions.json`, `validation.png`: audit có cả đánh nhau, bình thường cùng cảnh và video bình thường; ảnh giữa clip không đủ để xác minh hành động theo thời gian
- `training_code_revision.txt`, `export_code_revision.txt`: tách nguồn code huấn luyện và code xuất lại
- `sentinel_fight_only_runtime.zip`: `models/fight_model.ts` và `models/fight_runtime.json`

TorchScript chứa trọng số toàn mô hình; local không cần tải X3D/PyTorchVideo riêng. Notebook chỉ xuất checkpoint pilot, luôn `approved_for_demo=false`. Chép ZIP vào Downloads rồi chạy launcher. Tác vụ train toàn bộ và đánh giá test độc lập chỉ nên được thiết kế tiếp sau khi có pilot đạt và kiểm tra trực quan; notebook hiện tại chủ động không chạy phiên train nhiều giờ.

## Điều kiện nghiệm thu còn phải làm

- Kiểm tra trên test UBI-Fights và SCF độc lập; tránh tiếp tục chỉnh theo test.
- Đo báo nhầm trên đoạn bình thường dài, precision/recall theo sự kiện và độ trễ thực tế.
- Đo CPU trên cùng checkpoint đã export và cùng số vùng.
- Kiểm tra gần/xa, một người chuyển động, tiếp xúc bình thường, đông người, che khuất, camera khựng hình.
- Replay cùng frame và timestamp phải cho cùng kết quả; nguồn live mất frame có thể khác và cần đo riêng.

Training mới có thể cải thiện chất lượng, nhưng pipeline/UI không tự tạo ra độ chính xác. Kết quả pilot thực nghiệm mới quyết định có mở rộng hay đổi phương pháp.

## Kết quả pilot ngày 26/10/2026

Xem [báo cáo thực nghiệm](FIGHT_PILOT_2026-10-26.md). Lần X3D-S đầu tiên xuất checkpoint thành công nhưng **không đạt cổng pilot**; không dùng làm model demo mặc định và chưa chạy train toàn bộ.
