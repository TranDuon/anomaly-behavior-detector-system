# Fine-tune từ checkpoint RGB-aux hiện tại

## Đã sửa gì

- `fight/refinement.py` giữ **toàn bộ** classifier RGB-aux (backbone, head và residual fusion), không chỉ chép head RGB cũ. SHA nguồn được khóa tại `7cd20e23e1590689a1d08bf63a9aa967c831a71dd985fed1a09fae5bd32b7d09`. Đầu ra trước training được đối chiếu với TorchScript đang cài.
- Hai cấu hình `rgb_control` và `rgb_interaction` khởi tạo cùng nguồn, seed 941 và tập cửa sổ. Probe tối đa 3 epoch; sau đó mở gradient hai block cuối với LR 1e-5, đầu mới LR 1e-4. BatchNorm/dropout của classifier nguồn giữ eval để không làm trôi thống kê ngầm.
- Cache được tạo lại từ video gốc đã kiểm SHA, theo cửa sổ đã duyệt; fingerprint gồm SHA classifier đầy đủ, SHA video/pose, mốc frame, FPS và cấu hình RGB/pose/motion. Không đọc đặc trưng legacy baseline. Khi fine-tune, tính lại RGB từ clip, không dùng đặc trưng đã đóng băng.
- Nhãn cả clip không tự chuyển thành nhãn từng đoạn. File review mặc định `pending`; thời gian chưa gán nhãn không được coi là bình thường. Chặn split leakage theo SHA/scene/performance và đếm hai góc quay cùng biểu diễn một lần.
- Chọn model trên development, chọn ngưỡng trên calibration. Thời gian xác minh được đối chứng 0/0.4/0.8 giây **chỉ trên sự kiện development đã duyệt**, không hạ theo hai video iPhone. Test chỉ chấm sau khi đóng băng protocol.

## Quy trình thuận tiện

1. Mở `colab/fight_temporal_review_drive.ipynb` bằng **CPU**. Notebook pull main, dùng archive AIRTLab/RLVS trên Drive, tạo preview và giao diện review. Không cần NTU/PKU và không mở camera.
2. Gói nằm tại `SENTINEL/artifacts/fight_temporal_review_v1/temporal_review_pack.zip`. Giải nén trên máy; chạy `python -m http.server 8002 --bind 127.0.0.1 --directory <thư mục đã giải nén>` rồi mở `http://127.0.0.1:8002/via_review.html`. Bấm **Mở project chờ duyệt** để dùng Oxford VIA3 chính thức. Preview H.264 giữ FPS/số frame của nguồn; training vẫn đọc video gốc. Chuẩn bị dang dở có thể tiếp tục theo đúng danh sách và checksum, không chọn lại mẫu.
3. Xem clip, gán đoạn đầu/cuối và action trong VIA, lưu project bằng nút Save. Dùng `python -m scripts.fight_via import --pack <review_pack> --manifest <source_manifest.json> --project <via_saved.json> --reviewer <ma_nguoi_duyet> --method human --approve-reviewed --output <annotations.reviewed.json>`, rồi đặt file kết quả vào `SENTINEL/artifacts/fight_temporal_review_v1/`. Nếu AI hỗ trợ, dùng `--method ai_assisted`, không ghi thành human gold. Không có `--approve-reviewed` thì vẫn pending; video trống không tự thành normal. Đoạn không chắc dùng `uncertain`; không gây chấn thương thật. Giao diện tối giản tại `/` chỉ là phương án phụ; xem [đối chiếu mã nguồn mở](OPEN_SOURCE_COMPARISON.md).
4. Cell preflight kiểm tra tối thiểu 3 biểu diễn độc lập cho mỗi lớp/nguồn/split. Điều kiện này chỉ đủ cho pilot, **không phải chứng minh 80% camera**. Mẫu public đã xem trước vẫn là diagnostic lặp lại. Cần bộ quay độc lập chưa dùng để chọn model trước nghiệm thu.
5. Sau khi preflight qua và được duyệt ngân sách mới, dùng `colab/fight_refinement_drive.ipynb`. Theo yêu cầu cho phép fine-tune dài hơn ngày 28/10, notebook có watchdog toàn phiên 45 phút, gồm cả setup/staging, không reset khi chạy lại cell; runner tối đa 40 phút, mỗi cấu hình tối đa 15 phút. Vẫn dừng sớm nếu development không tiến triển; không tự retry hay cài model local. Cache public được đóng gói lưu Drive trước khi giải phóng runtime nếu còn thời gian.
6. Artifact mới ở `SENTINEL/artifacts/fight_refinement_v1`: checkpoint, lịch sử, protocol đóng băng, baseline hiện tại, metrics cửa sổ và sự kiện, runtime ZIP và gói QA GPU/CPU. Thiếu thời gian/lỗi được ghi thất bại; không biến run dở thành kết quả tốt.

Hai video iPhone vẫn ở máy và không được notebook đọc/upload/train/chọn ngưỡng. Công cụ local `scripts/fight_saved_recording_diagnostic.py` dùng để replay sau khi đã đóng băng model. Không vừa train vừa chứng minh kết quả trên chính hai video đã xem.

## Tiêu chí kiểm thử

Chấm riêng recall bỏ sót, precision, báo nhầm/phút **thời gian bình thường đã duyệt**, độ trễ bắt đầu và freshness CPU. Khoảng trống chưa gán nhãn bị loại khỏi mẫu số; báo động lan qua khoảng trống nhưng chạm thời gian bình thường đã duyệt không tự được bỏ qua. Báo cáo baseline gồm ngưỡng/gate đang cài và baseline hiệu chỉnh công bằng trên cùng tập calibration/development.

Một sự kiện thật chỉ ghép tối đa một cảnh báo; cảnh báo lặp được tính false positive riêng. Ngoài số báo nhầm, ghi cả số giây và tỷ lệ thời gian bình thường bị phủ báo động, tránh việc một cảnh báo kéo dài trông như đạt recall tốt. Notebook chuẩn bị tự giải phóng CPU khi xong/lỗi, watchdog tối đa 20 phút; không giữ GPU trong lúc chờ duyệt.

Cửa sổ video và sự kiện không phải cùng chỉ số. Precision/recall 80% là mục tiêu kiểm thử, không phải cam kết; số mẫu nhỏ, cùng người/phòng hoặc nhãn AI hỗ trợ không chứng minh tổng quát hóa. Không tự đặt `approved_for_demo` hoặc `real_phone_tested` thành true.

## CPU: đo rồi mới tối ưu

`FIGHT_CPU_THREADS` cấu hình 1–8 threads, mặc định vẫn 4 và được áp dụng lại sau detector warmup. API trả số threads đang dùng. Không tự giảm vùng người, FPS lấy mẫu hay ngưỡng để che độ trễ.

`python -m scripts.fight_cpu_profile --models models --examples <public-cache1.npz> <public-cache2.npz> --output <new-profile.json>` so sánh 1/2/4 threads, kiểm sai số điểm ≤1e-4, rồi đề xuất cấu hình. Đây là microbenchmark classifier, chưa gồm decode/detect/pose/render; phải qua `scripts/fight_cpu_acceptance.py` với cả upload và paced replay trước khi áp dụng. Gói model vẫn không tự được thay.

## Trạng thái lần sửa này

Đã tạo danh sách **91 nguồn public chờ duyệt** từ manifest đang có, không chọn bằng điểm model. Chưa có nhãn thời gian được duyệt: preflight hiện chặn training thay vì tự bịa nhãn. Chưa chạy lượt GPU mới và chưa có checkpoint cải thiện mới.

Ngày 28/10/2026 đã chạy notebook CPU với revision `6f5f6a33ca57554e1e8dd9be6613ba14fc4f6c4d`, tạo đủ 91 preview H.264 và [gói review trên Drive](https://drive.google.com/file/d/1vCBijxxj76Cy7mHB7lAfAebbGRuhQTL-/view?usp=drivesdk), 43.632.169 bytes. [Thư mục log/artifact](https://drive.google.com/drive/folders/17LAvACt2MTOdsjieP4xCAPpQB5T8yPwX). Runtime đã tự unassign; không dùng GPU. Gói đã tải về `logs/refinement-review-drive-v1`, đối chiếu checksum đủ 91 preview và kiểm source identity; manifest public gốc cũng đã chép vào thư mục local để import nhãn thuận tiện. Không có source nào tự được duyệt và checkpoint ứng dụng không đổi.

Sau đó đã duyệt bằng hình ảnh 64 nguồn public ở 4 fps, ghi rõ `ai_assisted`, reviewer, frame bằng chứng và lý do từng đoạn. Có xem thêm frame chi tiết khi chưa rõ; những đoạn vẫn mơ hồ được giữ `uncertain`, không sao chép nhãn cả clip. [File nhãn riêng trên Drive](https://drive.google.com/file/d/1uKNSgOPlXfJ0mAIcC-mTkAPZPmNc--z_/view?usp=drivesdk) có SHA-256 `1dd6f227b438e9b265985d5836a100fd0d32697a86979d56b604a7b1740b9fee`, đã tải ngược và đối chiếu byte. Preflight: 120 cửa sổ (train 74, development 15, calibration 16, test 15), mỗi cohort có ít nhất 3 nhóm độc lập theo manifest. Đây là pilot nhỏ, không phải tập train lớn hay nhãn human-gold; các nguồn chưa xem vẫn pending. Bộ nhãn gốc và hai video iPhone không đổi.

Đã sửa lỗi làm tròn frame cuối vượt khỏi khoảng duyệt bằng lấy endpoint floor; tránh loại nhầm cửa sổ âm hoặc lấn sang đoạn uncertain. Báo cáo training nay tách riêng số báo nhầm, thời gian báo động trên hug/greeting/high-five/gesture/solo motion, không chỉ accuracy gộp. Nhóm hành động không có mẫu được ghi thiếu dữ liệu, không ghi là đã đạt.

Đối chiếu đầu ra khởi tạo trên mẫu public với đúng checkpoint local đã qua. Đo CPU tại máy này: 4 threads nhanh hơn 1/2 trong microbenchmark; vì vậy giữ mặc định 4, không tuyên bố đã khắc phục độ trễ realtime. Cần tiếp tục đo cả pipeline, hoặc đánh giá model nhẹ hơn nếu CPU vẫn không đáp ứng.

Tham khảo: [fine-tuning MMAction2](https://mmaction2.readthedocs.io/en/latest/user_guides/finetune.html), [định dạng nhãn người/thời điểm của AVA](https://sites.research.google/gr/ava/download/). Dự án hiện dùng pipeline riêng, không tuyên bố đã chuyển sang MMAction2 hoặc đạt kết quả AVA.
