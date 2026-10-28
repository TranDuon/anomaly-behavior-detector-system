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

Notebook `colab/fight_classroom_drive.ipynb` giữ cấu trúc Drive cũ, fetch main rồi archive đúng `FETCH_HEAD` sang `/content`, không quét hoặc ghi đè working tree trên Drive. Artifact mới ở `SENTINEL/artifacts/fight_classroom_refinement_v1`; parent v2 giữ nguyên. Watchdog 45 phút gồm setup/staging/evaluation, không reset khi chạy lại cell; kết thúc tự giải phóng runtime. Cache public được đóng gói để tiếp tục, không đọc thư mục recordings local. Lỗi bootstrap được lưu cùng stage/traceback trước khi unassign, tránh mất bằng chứng khi Colab xóa output. Lượt phục hồi phải dùng thời gian còn lại của ngân sách đã duyệt, không tự nhận thêm 45 phút.

## Điều kiện trước khi dùng demo

Chưa có số accuracy mới chỉ vì code đã qua unit test. Khi train xong, đọc riêng report sự kiện/hành vi âm, so sánh parent trên cùng dữ liệu, kiểm tensor GPU/CPU, upload và paced replay CPU; sau đó mới replay hai recording local. Gói export vẫn `approved_for_demo=false`, `real_phone_tested=false`. Không hứa nhận được tương tác hoàn toàn khuất ngoài khung hình; cần camera thấy vùng tương tác và kiểm thử độc lập trước nghiệm thu lớp học.

## Kết quả chạy thật trong lượt này

Code revision `b967c61f1423744f2f7619ccf5355b40d97bf896`, Tesla T4, seed 941. [Artifact public trên Drive](https://drive.google.com/drive/folders/1U4toYxbrTj7pWK7zGfQeeQqGhQMrpK-N). Đồng bộ/prepare hoàn tất; cache gồm 325 cửa sổ train và 15 development. Không chấm test trong lượt này.

Training hoàn tất **3 epoch, 1.536 lượt lấy mẫu có lặp**, không phải 1.536 video độc lập. Loss lần lượt **0.108367, 0.039346, 0.032239**. Không epoch nào thắng tiêu chí development đã khai báo: AIRTLab vẫn nhận 3/3 sự kiện, không báo nhầm; RLVS vẫn 3/3 nhưng có một báo nhầm, precision 75%. Đây chỉ là sáu sự kiện public đã duyệt, không phải accuracy camera. Thời gian bình thường bị cảnh báo trên RLVS tăng từ khoảng 1.60 lên 2.16 giây ở các epoch mới. Vì thế không chọn checkpoint mới dù loss giảm; không kéo dài training chỉ để lấy loss đẹp.

`train_status.json`: `completed`, `training.improved=false`. `selection_frozen.json` giữ parent; `resume.pt` giữ cả optimizer/RNG và trọng số epoch cuối để kiểm tra, không phải gói deploy. Cache public 790.001.065 bytes đã lưu Drive. **Không sinh runtime ZIP/test report mới, không thay model local**. Notebook đã hiển thị Reconnect và tự ngắt GPU trước hạn cuối của tổng ngân sách được duyệt; không chạy lại để gia hạn.

Đã kiểm thử toàn bộ 217 test CPU; sáu test phục hồi/cache được chạy lại sau bổ sung mã đóng gói QA. Một optimizer step trên cache public thật có loss hữu hạn và thay trọng số trong RAM; không ghi đè model đang cài.

## Vấn đề realtime được đo riêng, không đổ hết cho dataset

Đã chạy parent RGB control trong API cách ly qua **upload và paced camera replay** của cùng năm video public, không mở camera thật. Upload có 2 TP/2 TN/1 FP; replay 2 TP/1 TN/2 FP theo nhãn cả clip. Một clip bình thường không báo ở upload nhưng báo ở replay. Trên các timestamp trùng, sai số điểm bằng 0; đây không phải bằng chứng mọi timestamp/cảnh báo giống nhau.

Kiểm tra integration **không đạt**: độ trễ/tuổi prediction lớn nhất quan sát được khoảng **3.577 giây**, có 51 lượt poll stale trong một clip; p95 inference khoảng **0.970 giây** ở clip đó, dài hơn stride 0.5 giây. Worker live hiện chỉ gọi predict trên mẫu cuối mỗi batch; khi CPU bận, các endpoint thực sự được chấm khác upload. Code và kết quả đo phù hợp với cơ chế này; không thể sửa bằng cách cài checkpoint hoặc hạ ngưỡng đơn thuần. Log cách ly local: `logs/classroom-parent-cpu-v1/cpu_acceptance.json`.

Hai việc còn phải đạt trước demo: (1) scheduling/cấu hình inference đủ nhẹ cho CPU, kiểm lại cả timestamp, khoảng cảnh báo và freshness của hai chế độ, không chỉ số sự kiện; (2) tăng khả năng phân biệt tương tác cận cảnh trên dữ liệu dương và hard-negative theo thời gian. Tăng epoch hoặc thêm nhiều cửa sổ âm không tự giải quyết hai lỗi này. Lượt này chưa chứng minh đã xử lý hết ôm/bắt tay hoặc đã đạt 80%; không đổi cadence/threshold sau khi xem kết quả để biến một lượt không tiến triển thành thành công.
