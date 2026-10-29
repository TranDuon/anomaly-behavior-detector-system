# MMAction2 Interaction Fusion v2

## Mục tiêu

Bản v2 sửa hai lỗi đối nghịch đã thấy khi thử camera:

- nhánh fine-tuned v1 có thể học đường tắt “có hai người” và báo đỏ trước khi có đánh nhau;
- điểm AVA gốc thường quá thấp nên bỏ sót hành vi đánh nhau thật.

V2 không chỉ thay threshold. Mỗi ứng viên một người/cặp người được phân loại từ đặc trưng AVA đã khóa và các tín hiệu phụ: điểm AVA gốc, số người, cờ cặp người, chuyển động toàn ảnh, chuyển động trong vùng tương tác, khoảng cách tâm và IoU của cặp người. Vì điểm sai của v1 từng cao hơn điểm đúng, chỉ tăng threshold không thể giải quyết lỗi này.

## Dữ liệu và cách train

Script [train_interaction_v2.py](../mmaction2_demo/train_interaction_v2.py) chỉ chấp nhận manifest indoor đã qua gate và tăng trọng số cho các hard negative: hai người tiếp cận/đứng gần/rời đi, ôm, bắt tay, high-five, vỗ vai, đưa đồ, cử chỉ, chuyển động một người và cảnh bình thường. Từ epoch 4, các negative mà model vẫn cho điểm cao tiếp tục được lấy mẫu nhiều hơn.

Trước lần train Colab tiếp theo, mở rộng manifest theo [đề xuất dữ liệu v2](MMACTION_DATASET_V2.md). Không dùng run từ manifest v1 làm checkpoint nghiệm thu mới.

Video camera riêng không được tải lên Drive hoặc đưa vào train. Chúng chỉ dùng làm bộ kiểm thử chấp nhận cục bộ sau khi chọn checkpoint, tránh làm rò rỉ dữ liệu test vào train/calibration.

## Chạy trên Colab

1. Upload `artifacts/mmaction_indoor_review_v2/indoor_review_pack.zip` vào `My Drive/SENTINEL/artifacts/mmaction_indoor_review_v2/` trên Google Drive. Tên file và thư mục phải giữ nguyên.
2. Mở `colab/mmaction_interaction_drive.ipynb` trên GitHub hoặc Colab, chọn GPU rồi chạy lần lượt bốn cell. Notebook kiểm tra manifest trước khi cài môi trường; nếu báo thiếu gói, kiểm tra lại bước 1.
3. Notebook khởi tạo từ checkpoint AVA SlowOnly R101 gốc; detector và backbone AVA được khóa. Chỉ fusion head mới được train. Cache, checkpoint và báo cáo được lưu trên Drive.
4. Notebook tái sử dụng cache v1 nếu có, bổ sung tín hiệu v2 rồi lưu cache tại `SENTINEL/artifacts/mmaction_interaction_v2/feature_cache.zip`. Mỗi revision có thư mục riêng tại `SENTINEL/artifacts/mmaction_interaction_v2/runs/<git-sha>/`.
5. Tải `mmaction_interaction_v2_pilot.zip` từ thư mục run đã hoàn tất. Chỉ cài nếu có `report.json` và calibration đạt gate.

Gói dữ liệu hiện đã tạo: 512 video indoor (AIRTLab 230, SBU 282), chia theo nhóm nguồn thành train 401, development 36, calibration 37 và test 38; 209 nguồn fight, 52 nguồn có pre-fight context và 0 mục quarantine. Các mốc pre-fight của AIRTLab có AI hỗ trợ, chưa phải nhãn human-gold; xem `DATASET_SOURCES.md` trong ZIP và rà các đoạn này trước khi kết luận chất lượng model. ZIP không nằm trong Git; tải lên Drive thủ công từ đường dẫn ở bước 1.

Không cài checkpoint nếu run thiếu `report.json` hoặc calibration chưa đạt mục tiêu. Báo cáo khóa các điều kiện ban đầu: precision tối thiểu 0,75, recall tối thiểu 0,60 và false-positive rate tối đa 0,10 trên calibration. Đây là gate kỹ thuật, không phải bằng chứng model đã đạt chất lượng triển khai thực tế.

## Cài và kiểm thử

Trong PowerShell tại thư mục repo:

```powershell
.\mmaction2_demo\install_branch.cmd "C:\duong-dan\mmaction_interaction_v2_pilot.zip"
.\run_local.cmd
```

Trên web, chọn `MMAction2 + nhánh fine-tuned`, bật checkpoint pilot rồi chạy ít nhất:

- clip hai người đi vào nhưng chưa tương tác: không được báo đỏ;
- ôm, bắt tay, high-five và cử chỉ mạnh: không được báo đỏ;
- clip đánh nhau thật: phải bắt được và thời điểm bắt đầu không trễ quá mức chấp nhận;
- một người chuyển động mạnh và cảnh không người: không được báo đỏ;
- video camera 30 giây: kiểm tra log `fusion_context`, điểm AVA gốc, điểm v2 và timestamp nguồn.

Nếu false positive vẫn còn, bổ sung đúng loại hard negative public tương ứng rồi train lại một revision mới; không sửa threshold trên chính bộ camera dùng để nghiệm thu.

## Phát video và clip bằng chứng

Web xử lý file upload không còn coi ảnh tiến độ là video phát lại. Khi phân tích xong, trình duyệt phát file kết quả ở FPS nguồn. Clip sự kiện của file upload cũng được dựng lại từ video nguồn ở FPS nguồn thay vì FPS lấy mẫu của model. Camera/replay tiếp tục tách luồng đọc khung hình khỏi luồng suy luận, nên model chậm không khóa việc thu và hiển thị khung hình mới nhất.
