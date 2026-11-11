# Đồng bộ khung người khi xem video/camera

MMAction2 chấm hành vi ở frame giữa bộ đệm 9 mẫu (index 4). Worker chạy detector thêm trên frame cuối để có khung người mới nhất, không đổi đặc trưng/điểm phân loại ở frame giữa. `box_timestamp` là thời điểm frame cuối, `action_keyframe_timestamp` là thời điểm phân loại và `prediction_timestamp` là lúc bộ đệm đủ dữ liệu. Nhãn/vùng cảnh báo chỉ chuyển sang khung mới khi ghép hình dáng/màu sắc/vị trí đủ rõ; trường hợp mơ hồ giữ khung người xanh, không tự gán ai là người đánh/người bị đánh.

`fight/box_tracking.py` giữ tối đa 160 ảnh xám/4 giây ở chiều ngang tối đa 640 pixel. Khi kết quả AI đến, khung được khởi tạo tại timestamp gốc và bám tới hình hiện tại bằng Lucas–Kanade optical flow, kiểm tra ngược và loại điểm lệch. Các lượt AI tiếp theo tạo lại điểm bám. Khung mất dấu, nguồn thay đổi kích thước hoặc không có frame tham chiếu thì ẩn; không giữ khung cũ vô hạn.

Đường xem camera/replay chạy trong thread preview riêng, không chờ phân loại. File xuất kết quả cũng dùng cơ chế đồng bộ này. Ảnh bằng chứng MMAction2 lấy đúng frame tương ứng với khung; timestamp ảnh được lưu riêng. API `/api/status` có `overlay.tracking_ms` và số người đang bám. Worker warm-up detector/AVA bằng dữ liệu giả trước khi bắt đầu nguồn để giảm khựng lượt đầu; clip replay ngắn được cập nhật khung cuối sau khi suy luận hoàn tất. Trọng số, ngưỡng phát hiện người và ngưỡng phân loại không đổi.

Đây là bám khung hiển thị, không phải hệ thống định danh người hoặc bằng chứng xác định ai đánh ai. Optical flow có thể mất dấu khi che khuất, thiếu sáng, chuyển động mạnh hoặc ít texture; người mới chỉ xuất hiện khi detector chấm lượt tiếp theo. Phân loại AVA vẫn cần khoảng 2.13 giây dữ liệu và xử lý clip; FPS hiển thị cao không đồng nghĩa cảnh báo không có độ trễ. Không cam kết đúng mọi cảnh hoặc mọi hành vi.

## Độ ổn định IPC trên Windows

Worker GPU và web trao đổi 9 frame BGR uint8 qua một named memory mapping tái sử dụng, không tạo/xóa một NPZ lớn cho mỗi lượt chấm. Giới hạn mapping 256 MiB, chỉ một yêu cầu đang chờ; web không ghi đè buffer trước khi worker trả kết quả. Kích thước nguồn đổi thì mapping được tạo lại, đóng khi kết thúc phiên. Không đổi kích thước ảnh, checkpoint, nhịp lấy mẫu hay ngưỡng để đổi lấy tốc độ. Ngoài Windows vẫn dùng đường file cũ.

Trace có `timings.frame_copy_seconds`, `worker_read_seconds`, `action_seconds`, `current_detector_seconds` và loại transport để tách trễ truyền frame khỏi trễ model/detector. Không quy kết GPU hay antivirus khi chưa có số đo từng phần. Khi replay đã hoàn tất, giao diện không còn hiển thị độ trễ cuối phiên như một lỗi đang diễn ra.
# Sensitive person detection

The web worker enables `sensitive_overlap_v2`: upright person proposals use a
0.6 confidence threshold with duplicate suppression. Weak horizontal proposals
are not accepted directly. When such a proposal is missed, two rotated detector
passes may recover it only with confidence >= 0.8 and spatial agreement.
This adds inference cost on difficult frames. It is not a guarantee of person
recall under occlusion. Training scripts retain the original detector defaults;
checkpoint weights and alarm thresholds are unchanged. Changing proposals may
change action scores, so deployment calibration needs revalidation.

## Overlap and running verification v2

The runtime detector uses RCNN NMS IoU 0.7 instead of 0.5 to retain overlapping
person proposals. A single bounded crop pass may confirm one extra proposal
inside a disproportionately large person region (confidence >= 0.85). Near-center
nested duplicates are suppressed. This is proposal recovery, not proof of two
distinct identities: full occlusion and merged bodies remain unresolved cases.
Pair features already include both singleton and person-pair candidates; their
checkpoint architecture is unchanged. Reference for the detector test config:
https://github.com/open-mmlab/mmdetection/blob/main/docs/en/user_guides/config.md

Running requires AVA run/jog score >= 0.4 in the existing temporal clip, supported
body motion and an unambiguous current person match. Grab-dominant and strongly
overlapping people do not confirm running. The existing 0.8-second running gate
still requires repeated observations. Loss of corroborating evidence clears the
running gate immediately, without changing the separate fight/fall gate.
This conservative rule may miss running in overlapping crowds or when AVA itself
fails. It is not a new trained gait model or an independently calibrated threshold.
Public regression clips are functional checks, not an accuracy benchmark.

## Separate-backend pose/RGB synchronization

RGB action boxes and current pose observations must be aligned before fusion. Previously an old RGB runner box could be appended next to a fresh pose runner box, both then incorrectly stamped with the current time. The host now projects the RGB snapshot to the sampled frame with a separate bounded optical-flow tracker (maximum sample gap .4 seconds, geometry expiry .8 seconds). Per-action regions are remapped with the boxes. Only pose lying candidates add geometry to the RGB inventory; upright pose detections do not create a second person inventory. The normal high-FPS display tracker still uses its original .2-second gap limit. Missing flow support drops a box rather than drawing an expired location. This changes display geometry, not checkpoint weights, action thresholds or confirmation times.

A reliable upright pose can relocate an existing RGB box through a bounded one-to-one size/distance association; ambiguous matches are not forced. This helps small fast runners where sparse optical flow can lock onto the background. Such matches replace the existing geometry rather than appending a second box, and retain that person's action attribution. They are geometric associations, not identity recognition or a guarantee through crossings and occlusion.

## GPU pose and bounded live latency (2026-11-11)

The separate backend runs the unchanged YOLOv8n-pose model at 640 pixels inside the isolated CUDA worker, including non-AVA sample times. CPU pose is retained for isolated unit-test/fallback runtimes, not silently selected after a GPU error. Worker reset clears posture confirmation between sources. Shared-memory pixels are mutable, so pose's pointer-keyed cache is cleared for each request; otherwise subsequent frames would reuse old joints.

Person detections use a bounded 24-entry byte-content cache within a worker. The current frame becomes a later action keyframe, so an exact cached detector result avoids redundant inference without lowering resolution, changing thresholds or approximating a nearby frame. Cache outputs are copied and no video frames are retained. Action, detector, pose, full host pipeline duration and cache hits are logged separately. Detection and ROI action inference retain their required dependencies; capture and high-FPS display remain separate threads.

Live inference keeps at most `ceil(sample_fps * .8)` queued samples and discards samples older than .8 seconds at dequeue. Sample loss is counted, clears causal action state, and marks analysis incomplete; it must not be described as full-video coverage. This favors current camera results over seconds of backlog, but action models still require 2.13 seconds of observations and confirmation. It is not instantaneous action recognition.

Uploaded videos default to full offline analysis with no live queue drops, followed by H.264/yuv420p/faststart browser playback at the original FPS. The processing preview is not a realtime speed benchmark. Encoding is cancellable and preserves the generated original if it fails. The optional realtime checkbox remains a camera-like test with possible dropped model samples. Explicit Stop and shutdown release the worker; successful completion keeps the worker warm for the same backend's next source.

Local checks are recorded in `mmaction2_demo/results/realtime_gpu_pipeline_regression.json`: the previously slow neckhold replay ended with .394 seconds lag versus the earlier 5.779 seconds, with 3 dropped samples and a 1.765-second peak; the running replay had .213-second final lag, .248-second peak and zero drops. These are single-machine functional measurements, not guaranteed latency or recognition accuracy. Full upload checks retained strike and lying alerts with zero sample drops and produced H.264 files. Checkpoints, action thresholds and confirmation periods were not changed by this performance work.
