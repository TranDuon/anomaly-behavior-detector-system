# Kết quả RGB / pose / nhãn hành động phụ — 27/10/2026

## Trạng thái

Đã training thật trên NVIDIA L4 và xuất hai checkpoint. **Chưa duyệt demo, chưa tự thay checkpoint local và chưa thử iPhone thật.** Không được dùng accuracy của một cửa sổ/clip để gọi là accuracy của camera liên tục.

Cập nhật cài thử local sau báo cáo: theo yêu cầu rõ ràng của người dùng, đã cài `rgb_aux` được chọn bằng development vào `models/` và mở server CPU ở `127.0.0.1:8001`. Đây là kích hoạt pilot để thử iPhone, không phải nâng trạng thái nghiệm thu. Giao diện/API hiển thị tên, ngày training và training revision; checksum gói `4666646a40895008f3fca78355e3f53a8d783c85430fccc3be90c8bc875f6702`. Warmup, forward hai region và 138 kiểm thử code qua; chưa tự bật camera hoặc xác minh điện thoại thật.

- Code training: `37c06874e03a707fef51ceb7bf6ef61612ae2e5d`; code đóng gói/báo cáo: `15bcc58e39affac1d957ff98eab3fd201acfb873`.
- Tổng cache/train/evaluate/export: 2.374,40 giây, khoảng 39 phút 34 giây; cài đặt và stage dữ liệu tính riêng.
- Hai cấu hình đều hoàn thành 3 epoch probe + 6 epoch fine-tune. Checkpoint tốt nhất ở fine-tune epoch 2; dừng sau 4 epoch không cải thiện tiêu chí development ưu tiên. Không tiếp tục nhiều giờ chỉ vì training loss giảm.
- Khởi tạo từ large-v1 X3D, không khởi tạo từ checkpoint PKU pilot. Không dùng PKU để train hoặc chọn ngưỡng/model/gate.
- Dữ liệu và giới hạn phương pháp: [protocol](FIGHT_FUSION.md).

## Dữ liệu thực sự sử dụng

1.526 clip train: 760 dương, 766 âm; development 216; calibration 228; public diagnostic 202. Hai góc quay AIRTLab cùng một lần diễn được nhóm chung khi chia tập. Loại hai clip friendly-punch mơ hồ, không tự đổi nhãn. Chỉ giới hạn âm nền không thấy người ở train; không bỏ mẫu khó trong development/calibration/test.

Sampler lấy 1.024 lượt/epoch có lặp để cân bằng nguồn và nhãn, không phải 1.024 video mới mỗi epoch. Archive RLVS có các thư mục lặp: 4.000 entry video không phải 4.000 video độc lập. Manifest public sử dụng 2.000 RLVS duy nhất trước chia tập.

## Kết quả và đối chứng

Baseline được chạy lại với cùng crop từ detector pose, 12fps và calibration riêng; không trộn con số này với baseline production có preprocessing cũ.

| Model | Nguồn | Clip | TP / TN / FP / FN | Accuracy | Precision | Recall |
|---|---|---:|---|---:|---:|---:|
| Baseline cùng preprocessing | AIRTLab | 20 | 5 / 10 / 4 / 1 | 75,00% | 55,56% | 83,33% |
| RGB + nhãn phụ | AIRTLab | 20 | 6 / 14 / 0 / 0 | 100,00% | 100,00% | 100,00% |
| RGB + pose + nhãn phụ | AIRTLab | 20 | 6 / 14 / 0 / 0 | 100,00% | 100,00% | 100,00% |
| Baseline cùng preprocessing | RLVS | 182 | 89 / 84 / 5 / 4 | 95,05% | 94,68% | 95,70% |
| RGB + nhãn phụ | RLVS | 182 | 83 / 87 / 2 / 10 | 93,41% | 97,65% | 89,25% |
| RGB + pose + nhãn phụ | RLVS | 182 | 84 / 87 / 2 / 9 | 93,96% | 97,67% | 90,32% |

**Có đánh đổi:** báo nhầm RLVS giảm 5 xuống 2 nhưng bỏ sót tăng 4 lên 10 ở RGB-aux. Không kết luận mọi mặt đều tốt hơn.

Chọn `rgb_aux` bằng development, không bằng test: worst-source min(precision, recall) tăng 0,7500 lên 0,9231; macro AP tăng 0,9355 lên 0,9899. Bản pose có cùng worst P/R nhưng macro AP 0,9897 thấp hơn một chút. Pose chưa cho thấy lợi ích thêm rõ ràng trong chẩn đoán liên tục.

Ngưỡng calibration độc lập: RGB-aux `0.958552896976471`; RGB-pose-aux `0.9529496133327484`. Gate xác nhận `0.4s`, chọn từ development whole-clip any-alarm. Đây không phải calibration sự kiện có nhãn thời gian chính xác. Không hạ ngưỡng theo kết quả PKU.

## Các hành động dễ nhầm

Ở cửa sổ trung tâm public AIRTLab, cả hai bản đều không báo nhầm 14/14 clip âm. Theo tag publisher: greet 10 clip/5 lần diễn; handgestures 4/2; high-five 4/2; hug 2/1; jump 4/2. Tag có thể chồng nhau, không cộng các nhóm thành tổng độc lập.

Đây chỉ là 10 lần diễn AIRTLab qua 20 góc quay cho cả hai nhãn. Với chỉ 6 positive clip, cận dưới Wilson 95% của precision/recall khoảng 60,97%, không chứng minh hệ thống đạt 100% hoặc ít nhất 80% ngoài đời. Handshake cụ thể chỉ có trong các đoạn PKU nhỏ; can ngăn chưa có tập nghiệm thu riêng. Nhãn phụ là weak clip tags, chưa phải nhãn thời gian đã rà soát.

## Video liên tục: vẫn chưa đạt

Hai nguồn PKU đã được xem ở các lần trước chỉ là **chẩn đoán lặp lại**, không phải test độc lập mới. Mỗi nguồn có 2 sự kiện đánh nhau; cả hai model bắt 1/2 ở `0176-L` và 1/2 ở `0360-R`, tổng **2/4 sự kiện, bỏ sót 2**, không báo giả theo quy tắc event matching đã đóng băng, trên khoảng 1,85 phút âm. Độ trễ hai sự kiện bắt được khoảng 1,23 và 1,47 giây theo source timestamp, không phải độ trễ iPhone đầu-cuối.

Chẩn đoán raw score cho thấy ôm vẫn có peak cao, còn một cú đấm thấp hoặc peak quá ngắn. Gate chặn được peak ôm nhưng cũng bỏ lỡ cú đấm ngắn. Vì vậy hạ threshold hoặc bỏ gate đơn thuần không giải quyết triệt để. Hai người gần nhau không tự bị loại; khó nằm ở phân biệt tiếp xúc lành tính/bạo lực và ngữ cảnh thời gian, không chỉ độ nét ảnh.

## Export và CPU

- Hai gói gồm TorchScript hai input, JSON cấu hình/ngưỡng/gate và trọng số pose chính thức. Dữ liệu pose vắng mặt được mask, không ngầm đổi sang model khác.
- Cả 6 mẫu tensor chọn cố định theo tag/SHA đã qua đối chiếu GPU/CPU với tolerance `1e-4`; không chọn mẫu theo score. Đây không bao gồm detector, hàng đợi camera hoặc điện thoại.
- Runtime RGB-aux bỏ tính pose theo thời gian vì nhánh đó bị vô hiệu hóa trong model; vẫn dùng cùng detector pose và crop. Kiểm thử xác nhận output không phụ thuộc pose ở cấu hình này.
- Giới hạn luồng giải mã FFmpeg, preview 960px và output upload 1280px; **không giảm độ phân giải frame đưa vào preprocessing model**. Có fallback cho backend không hỗ trợ tùy chọn giải mã; camera native không ép FFmpeg.
- Kiểm tra CPU sơ bộ với video PKU 1080p đã phát hiện upload/replay khác nhịp dự đoán, có stale và khác sự kiện. Các tối ưu trên **chưa đủ để coi realtime đạt**; không tự cài candidate chỉ vì export đúng.
- 137 kiểm thử code qua, không có nghĩa 137 clip thực tế đều đúng.

### Kiểm tra API thực tế trên 6 video CPU

Đã chạy đủ 12 lượt, mỗi video qua upload và paced camera-replay với đúng gói RGB-aux. Năm clip public được chọn cố định theo action tag/SHA, không chọn theo score; nguồn thứ sáu là PKU `0360-R`, không có một nhãn whole-clip để tính accuracy.

| Video | Upload: số alarm | Replay: số alarm | Nhận xét |
|---|---:|---:|---|
| AIRTLab cam1/58, greet/hug/jump | 1 | 1 | **Báo nhầm**, dù cửa sổ trung tâm của clip này chấm đúng |
| AIRTLab cam2/40, greet/high-five | 0 | 0 | Không alarm ở phép thử này |
| AIRTLab cam2/59, greet/handgestures | 0 | 0 | Không alarm ở phép thử này |
| AIRTLab violent cam2/8 | 1 | 1 | Có alarm, không có temporal gold để kết luận chính xác thời điểm |
| RLVS V_453 | 1 | 1 | Có alarm, không có temporal gold để kết luận chính xác thời điểm |
| PKU 0360-R, nhiều hành động | 1 | 2 | **Không khớp upload/replay**; không coi hai alarm replay là đã bắt đúng hai sự kiện |

Whole-clip any-alarm trên 5 clip public: mỗi mode TP=2, TN=2, FP=1, FN=0 (4/5 đúng). **Không dùng 80% trên 5 clip này để cam kết 80% demo**. Chúng đã được xem trước, số lượng rất nhỏ và không phải bài kiểm tra iPhone độc lập.

Ở PKU, upload có 87 dự đoán, replay chỉ 62; median inference replay khoảng 0,642s so với stride model 0,5s. Lag lớn nhất 2,997s; prediction age lớn nhất 3,769s; 3 sample rơi, 44 lần polling gặp kết quả stale. Score ở 6 timestamp trùng nhau bằng nhau, nhưng như vậy không bù được các cửa sổ không được chấm. Những clip âm có nhiều region cũng có median inference tới khoảng 0,88s ở replay. **Cổng freshness/parity chưa đạt.**

Vì vậy tồn tại hai lỗi riêng: mô hình vẫn nhầm ở một số đoạn ngoài cửa sổ trung tâm; CPU chưa giữ được lịch chấm đều khi có decoder và lưu bằng chứng. Không được chữa bằng cách đổi định nghĩa metric, xóa clip lỗi, tăng timeout stale hoặc hạ threshold theo test. Raw report được giữ nguyên tại `logs/fusion-v1/cpu-selected/cpu_acceptance.json`; summary riêng lưu kết luận không duyệt demo.

## Lưu trữ và tái lập

[Artifact trên Drive](https://drive.google.com/drive/folders/1XAq1HZu3cpxCMyP6O8egO5mPSODeP1On): `run.json`, `source_manifest.json`, `provenance.json`, training log, mỗi model có `best.pt`, history, protocol, calibration, test predictions, runtime ZIP, learning curves và PKU timelines. `window_cache.zip` khoảng 3.465 MiB; `cpu_validation.zip` khoảng 88 MiB. Không cần giữ GPU để xem các báo cáo này.

Kiểm tra local không ghi đè `models/`:

```powershell
.venv\Scripts\python.exe -m scripts.fight_fusion_qa unpack logs/fusion-v1/cpu_validation.zip logs/fusion-v1/qa
.venv\Scripts\python.exe -m scripts.fight_interaction_cpu_smoke logs/fusion-v1/cpu_validation.zip logs/fusion-v1/cpu_smoke_report.json
.venv\Scripts\python.exe -m scripts.fight_cpu_acceptance --bundle logs/fusion-v1/qa/rgb_aux/runtime.zip --sources logs/fusion-v1/qa/sources.json --output logs/fusion-v1/cpu-selected --budget-minutes 12
```

Unpack chỉ dùng thư mục đích mới/trống. CPU acceptance lưu raw timestamp, score, event, prediction age/lag và dropped samples; `completed` chỉ nghĩa phép đo kết thúc, không tự có nghĩa nghiệm thu đạt.

## Hướng tiếp theo có căn cứ

1. Tách mã hóa/lưu bằng chứng ra khỏi đường suy luận, đo từng công đoạn CPU và so sánh lịch cửa sổ upload/replay; không che độ trễ bằng cách tăng stale timeout. Cần throughput ổn định ở nhịp đã hiệu chỉnh trước khi hứa realtime.
2. Bổ sung nhiều cửa sổ có mốc bắt đầu/kết thúc từ dữ liệu train; đối chứng nhánh ngắn cho cú đánh và nhánh dài cho tương tác, thay vì chỉ một cửa sổ trung tâm 1,25 giây với nhãn cả clip. Chọn cấu hình bằng development riêng rồi calibration lại.
3. Cần nguồn độc lập cho ôm/bắt tay/high-five/vung tay/can ngăn và đánh nhau, có nhóm/địa điểm mới. Độ phân giải 1080p không thay thế được đa dạng hành động và nhãn đúng. PKU hiện có quá ít RGB, không bắt buộc dùng nó để train.
4. Chốt một tập nghiệm thu mới chưa dùng chọn model, chấm recall sự kiện, false alarms/phút, latency và từng hard-negative riêng. Kết nối iPhone để kiểm tra rung, mất FPS, nhiều người/che khuất trong lớp/hành lang/ngoài trời. Nếu chưa có nguồn này, phải ghi rõ chưa xác minh, không hứa 80% cho demo.
