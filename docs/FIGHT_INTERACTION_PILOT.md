# Pilot hành động theo thời gian

## Mục tiêu

Khắc phục nhầm cử chỉ/ôm/high-five và bỏ sót đấm/đá mô phỏng. Giữ checkpoint `fight_large_v1` làm đối chứng. Không suy accuracy clip gộp thành độ chính xác camera. iPhone chưa kết nối.

## Dữ liệu và giới hạn

Đọc đúng `SENTINEL/datasets/fight (1)/pku_mmd_sample`: 3 RGB nguồn, nhãn frame và danh mục hành động. Positive IDs 18/26 được kiểm tra theo tên trong danh mục, không suy từ tên file. ID19 đá đồ vật là âm, ID27 push chưa chốt và confidence=0 là không chấm. Giả định nhãn publisher 1-based inclusive, chuyển 0-based và kiểm bounds/frame count. Cửa sổ dương gắn nhãn đoạn hành động; cửa sổ âm không chồng vùng dương/không rõ cùng biên bảo vệ 0,25 giây.

Chỉ `0005-M` train; `0176-L` và `0360-R` cùng phía chẩn đoán, tránh giả định hai nguồn đó độc lập diễn viên. Chưa xác minh actor/session chính thức nên không tuyên bố cross-subject. Đây là pilot rất nhỏ: chỉ hai occurrence dương PKU trong train, các cửa sổ/augmentation không tạo thêm mẫu độc lập. Các nguồn chẩn đoán từng được xem; không phải test cuối mới. Không có calibration PKU độc lập.

AIRTLab/RLVS giữ train/development/calibration từ manifest v1, chọn subset theo SHA256 trước khi chấm. Không lấy test v1 làm train/selection/calibration, không nhân đôi sample AIRTLab upload lại. Mục tiêu nhãn người dùng vẫn cần rà đủ ngữ cảnh trước nghiệm thu.

Nguồn PKU Phase I chính thức hiện đòi quyền truy cập tài khoản đang dùng. Phase II công khai có 41 class nhưng nhãn đã kiểm không chứa đấm/đá vào người; không tải nó rồi giả định đã mở rộng dương. Có thể dùng về sau cho âm khó. Không né quyền truy cập hoặc xin quyền chỉnh sửa.

## Đối chứng cố định trước khi chạy

1. Baseline v1 không đổi weights/ngưỡng, chấm cửa sổ và video liên tục trên hai nguồn PKU chẩn đoán.
2. `aligned_8fps`: 16 frame/8fps, lấy nhiều vị trí thời gian trong đoạn có nhãn thay vì chỉ giữa file.
3. `short_12fps`: 16 frame/12fps, khoảng nhìn ngắn hơn cho động tác ngắn. Giữ stride 0,5 giây và policy xác nhận 0,8 giây để so sánh cùng gate.

Khởi tạo cả hai từ cùng SHA256 weights v1. Cache gắn cả SHA256 video, config và backbone thực tế. Mỗi epoch lấy cân bằng dataset × class (có lặp); ghi riêng lượt lấy mẫu và occurrence độc lập. 2 probe + tối đa 6 fine-tune, checkpoint chọn bằng macro AP development RLVS/AIRTLab. Augmentation nhẹ dùng cả hai lớp. Ngưỡng global tối đa hóa precision/recall thấp nhất của các dataset calibration, không tùy biến theo tên dataset khi chạy app.

Không đưa PKU chẩn đoán vào chọn checkpoint/ngưỡng. Không hứa macro calibration đảm bảo PKU chưa có calibration. Kết quả báo riêng từng cohort; nếu không đạt, không tự tăng giờ train hoặc gắn cờ demo.

## Đo video liên tục

Cùng `FightRuntime`, preprocess, timestamp và `EventGate`; GPU chỉ dùng để chấm thí nghiệm, không gọi đó là kiểm thử tốc độ local CPU. Match event một-một với interval publisher, cho trễ cuối 1 giây và IoU tối thiểu 0,1 trên interval mở rộng. Alarm luôn bật không được tự coi là bắt đúng mọi event. Báo precision/recall sự kiện, false alerts/phút bình thường, độ trễ bắt đầu và vùng bỏ qua. Không so trực tiếp phép đo này với mAP paper khác protocol.

## Vận hành

[Notebook](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/fight_interaction_drive.ipynb). Cache/train/chấm/xuất tối đa 20 phút, watchdog 21 phút; chuẩn bị dữ liệu tách riêng. Checkpoint, manifest, điểm calibration/diagnostic, timeline và báo cáo từng variant lưu `MyDrive/SENTINEL/artifacts/fight_interaction_pilot_v1`. Đã có comparison thì chỉ đọc lại, không tự retry.

Không tự thay model local hoặc chọn variant bằng kết quả trên các nguồn đã xem. Muốn quyết định triển khai cần thêm nguồn/người diễn mới, kiểm thử CPU upload/replay, rồi camera iPhone thật. `approved_for_demo=false` luôn được giữ trong pilot này.
