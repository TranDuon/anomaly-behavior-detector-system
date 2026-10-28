# Classroom repair — 28/10/2026

## Đã xác minh trước lượt train mới

Checkpoint `rgb_control` phục hồi từ refinement v2 nhận được một trong hai đoạn iPhone khi replay local; đoạn còn lại vẫn bỏ sót. Chưa nghiệm thu live iPhone. Hai đoạn đã được xem nên chỉ là regression diagnostic, không phải test độc lập; không dùng để train, chọn ngưỡng hoặc tải lên Drive.

Model đang cài không bị thay tự động. Cảnh cận chỉ có một người chiếm phần lớn khung hình và sát mép được ghi hạn chế quan sát: chưa xác minh được tương tác, không kết luận bình thường. Thông báo này không chặn cảnh báo đang xác minh/đã phát hiện và không thay ngưỡng.

Đã thử adapter AVA chính thức của PyTorchVideo: có nhãn fight/hit, hug và handshake riêng, nhưng điểm trên các nguồn public vẫn chồng lấn và classifier mất khoảng 1–2 giây/cửa sổ trên CPU máy này. Không bật vào runtime, không chọn ngưỡng theo điện thoại, không dùng hug làm veto tuyệt đối. Tham khảo [inference chính thức](https://pytorchvideo.org/docs/tutorial_torchhub_detection_inference), [triển khai model](https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/resnet.py) và [webcam spatiotemporal detection của MMAction2](https://github.com/open-mmlab/mmaction2/blob/main/demo/webcam_demo_spatiotemporal_det.py). Adapter không có nghĩa là đã chuyển dự án sang MMAction2.

## Cấu hình sửa tập trung

- Bắt đầu từ đúng `rgb_control/best.pt` của run v2, xác minh trọng số, manifest, nhãn và pose SHA. Không quay về feature legacy, không train lại cả hai kiến trúc.
- Preflight thực tế: 74 cửa sổ train đã duyệt + **251 cửa sổ âm**, thuộc **42 nhóm biểu diễn AIRTLab**, tổng 325. Các cửa sổ thêm chỉ từ video train có nhãn non-violent và tag an toàn; không đổi development/calibration/test. Đây là nhãn publisher cấp clip, được ghi riêng, không giả thành temporal human-gold. Greeting chung không tự thành nhãn handshake.
- Lấy mẫu cân bằng theo nguồn/lớp và nhóm biểu diễn để hai camera/các cửa sổ lặp không được đếm như biểu diễn độc lập. Cận cảnh dùng cùng crop cho mọi frame, không tạo jitter giả; không cắt ngẫu nhiên chân khỏi mẫu đá hoặc bịa nhãn dương khi tương tác đã bị cắt mất. Chỉ cấu hình RGB được dùng augmentation này vì pose cũ không còn đúng tọa độ sau crop.
- Fine-tune RGB thực tế mỗi epoch, không dùng feature cache của backbone cũ làm đầu vào fine-tune. Giữ backbone BN/dropout eval như pipeline trước.
- Chọn checkpoint bằng precision/recall sự kiện development, số báo nhầm/phút, phần thời gian bình thường bị báo động và AP bảo vệ từng nguồn; loss giảm đơn thuần không đủ. Dừng sau ba epoch không tiến triển hoặc tối đa 12 epoch.

## Không mất lượt chạy vì đánh giá gián đoạn

`python -m scripts.fight_classroom_refine --phase check --manifest <source_manifest.json> --annotations <annotations.reviewed.json>` kiểm tra không cần GPU.

Các pha `prepare`, `train`, `evaluate` có ngân sách riêng. Pha train chỉ chạm train/development và dành tối đa 60% thời gian còn lại sau cache cho optimizer. Calibration/test chỉ chạy sau khi đóng băng checkpoint; calibration chọn ngưỡng, development chọn gate. Không train tiếp trong cùng output sau khi selection đã frozen.

`resume.pt` lưu model hiện tại/best, optimizer, RNG, lịch sử và trạng thái epoch đã train nhưng chưa chấm xong. Nếu chết giữa epoch, quay về epoch hoàn chỉnh gần nhất; nếu chết lúc chấm, không train lại epoch đó. Mỗi video chấm xong lưu trace ngay, kèm fingerprint trọng số/nguồn/FPS/clip/pose/code và checksum. `--resume` từ chối khác identity; không lấy trace cũ cho trọng số mới. Timeout ghi `budget_stopped`, không ghi completed giả.

Notebook `colab/fight_classroom_drive.ipynb` giữ cấu trúc Drive cũ, pull main rồi đưa code sang `/content`, không quét file untracked. Artifact mới ở `SENTINEL/artifacts/fight_classroom_refinement_v1`; parent v2 giữ nguyên. Watchdog 45 phút gồm setup/staging/evaluation, không reset khi chạy lại cell; kết thúc tự giải phóng runtime. Cache public được đóng gói để tiếp tục, không đọc thư mục recordings local.

## Điều kiện trước khi dùng demo

Chưa có số accuracy mới chỉ vì code đã qua unit test. Khi train xong, đọc riêng report sự kiện/hành vi âm, so sánh parent trên cùng dữ liệu, kiểm tensor GPU/CPU, upload và paced replay CPU; sau đó mới replay hai recording local. Gói export vẫn `approved_for_demo=false`, `real_phone_tested=false`. Không hứa nhận được tương tác hoàn toàn khuất ngoài khung hình; cần camera thấy vùng tương tác và kiểm thử độc lập trước nghiệm thu lớp học.
