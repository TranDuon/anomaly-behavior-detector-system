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

## Separate running policy v3 (2026-11-12)

### Running-only depth association v4

The separate backend can recover a high-confidence running identity through detector matches at all four intermediate transitions from the action keyframe to the current frame. Missing or ambiguous intermediate matches abort recovery; an existing identity assignment cannot be stolen. Only running score/reference fields are restored. AVA strike labels, strike/grapple/fall scores and regions, the shared detector, and the historical fixed-running backend are unchanged.

A depth-specific support path requires five matched boxes with consistent growth/shrinkage (absolute log height change at least .3), bounded per-step scale changes and bounded center displacement. With that additional evidence only, pilot >= .75 and AVA run >= .3 can support confirmation. Existing general support (.85/.25), clipped-body AVA requirement (.6), interaction/overlap vetoes and confirmation times remain. Size change alone never confirms running. These are development heuristics, not newly calibrated probabilities or an independent accuracy result.

The frontal street regression now confirms running at 13.33 seconds where direct keyframe-to-current association previously lost the runner. The earlier low-confidence/receding interval is not claimed solved. Recovery conditionally adds at most three detector calls per action inference, using the existing frame cache; no new checkpoint or training is involved. Regression output is `logs/running_depth_final_regression.json`.

The seek-based regression above did not reproduce the web decoder's exact images. A subsequent actual web run still missed the event: an intermediate detector returned both a full body and a near-identical torso crop, making association ambiguous. The running-only fallback now uses upper-body appearance with bounded motion/scale and excludes near-identical truncated proposals from its association candidates. It does not remove boxes from the shared detector or change other action fields. Distinct similar-clothed people still cause abstention when association is ambiguous.

Actual HTTP `/api/upload` with realtime replay subsequently recorded `fast_running` at 13.358876 seconds, ending at 13.626054, score .967937, with a red person snapshot. The report is `logs/running_actual_web_verification.json`. This run dropped one sample, so it is not a complete-coverage or latency acceptance test. Further replay was not started because the user had already begun another source. Do not treat the earlier seek-only result as web acceptance.

This later behavior change applies only to `mmaction_separate_actions`. The historical fixed-running backend retains its original rules. No checkpoint, detector, strike threshold, lying policy or shared `EventGate` implementation changes.

Two strong, supported observations on the same running track can confirm after .4 seconds instead of always waiting .8 seconds (three AVA decisions). Ordinary candidates still wait .8 seconds. Strong support requires pilot >= .85, AVA run/jog >= .25, visible body motion, no grab dominance and no substantial person overlap. This replaces the hard AVA >= .4 veto only for that strong tier. A clipped moving person additionally needs AVA >= .6; clipping alone no longer rejects such evidence. These tier constants are experimental, not newly calibrated probabilities.

Confirmation is per person, using recent action-box correspondence, bounded geometric matching and a small upper-body HSV histogram. It is not face recognition or guaranteed identity through occlusion. The track history can bridge one missing action observation (up to 1.15 seconds), but does not draw or alarm on an absent/unlocalized person. Explicit stationary or contradictory evidence resets confirmation. Different people cannot pool their isolated positive observations into one alarm. Ambiguous overlaps and unreliable matches can still be missed; matching requires broader validation before camera acceptance.

Local functional results are stored in `mmaction2_demo/results/running_v3_regression.json`. The street video is a regression case used to diagnose this policy, not an independent accuracy benchmark. Existing strike false positives are not claimed fixed by this running-only change.

## Atomic preview publication and running notices

The separate backend previously exposed its intermediate RGB snapshot while combining the latest pose result. A concurrent preview reader could therefore see timestamps go backward and reseed its optical-flow tracks. Inference now works on a thread-local draft and publishes the completed snapshot in one replacement. Errors leave the last completed snapshot intact; age/error handling still applies. Historical backends retain their existing decision path.

A confirmed running result and its display geometry are distinct. If display tracking fails, a fresh confirmed running result keeps its red header with `BOX UNAVAILABLE`, but draws no fabricated rectangle. A timestamped `RECENT RUNNING CONFIRMED` notice may remain for up to 1.2 seconds after the source observation so a short event is visible to the viewer. It does not extend model state, event duration or person highlights; it expires, resets with the source, and does not cover another active action label. A rectangle can still be absent when its current location is uncertain.

Each completed snapshot also carries the most recent confirmed running timestamp. This lets preview recover an event even if a short positive snapshot was replaced between two rendered frames. The remembered source timestamp is not refreshed by drawing the notice, so it cannot persist indefinitely or manufacture another event.

The preview regression report is `mmaction2_demo/results/preview_atomic_regression.json`. It checks monotonic published timestamps and counts confirmed headers, actual red-box frames and recent notices separately. Do not equate a visible header with successful person localization. This change does not modify weights, action scores, confirmation gates or event recording decisions.

## Small-person display fallback (2026-11-12)

Only when pose has no reliable person boxes and all detected people are small (height <= 30% and width <= 15% of the frame), the separate backend supplies a render-only `display_geometry` snapshot from actual person-detector observations. It is disabled whenever strike, grapple or fall is verifying/active. Close-up fights, lying scenes, pose-supported scenes and historical backends retain the previous display path. This explicit scope prevents a running-display experiment from replacing close-up fight geometry. Every observation retains its original source timestamp; the sparse RGB optical-flow result is not restamped and reused as a new display detection. The high-FPS renderer chooses this one geometry path instead of applying the pose/RGB flow output followed by another tracker.

The fallback uses OpenCV's standard `KalmanFilter` on box center/size and velocity, associated by bounded geometry and upper-body color similarity. It never learns motion from stationary car/background corners. Render predictions do not modify the filter's observation state or clock. Predictions expire after .65 seconds. A first observation without velocity is hidden once older than .1 seconds until a second real observation arrives. Small detector corrections are blended for .12 seconds, bounded to a quarter of body height; actual direction changes are not globally forbidden. Ambiguous associations can still be lost.

For this display fallback only, a detector-only GPU request can refresh observations between AVA decisions, bounded to the 3.75 FPS sample cadence. Display loss can request another observation; no independent thread writes worker IPC. `display_detect` does not call pose observation, advance action samples, change confirmation gates, or replace the model result. Action labels are mapped conservatively from the reference frame; an uncertain association does not paint an arbitrary person red. No new model weights or OpenCV package changes are required.

OpenCV reference: https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html . A NanoTrack trial was rejected because its image-template tracking also locked onto the parked car in this regression clip; its downloaded trial weights are not part of the runtime.

The exact small-runner replay is recorded in `mmaction2_demo/results/small_runner_final_regression.json`: zero backward center steps exceeding 5 pixels, zero dropped samples, approximately .226-second final lag on this machine. Initial boxes may be withheld until motion is observable; this is not a promise of immediate detection or perfect tracking through occlusion. `detector_geometry_regression.json` also checks the earlier runner, strike and lying clips. Their sampled action scores and alert times remain unchanged. The short small-runner clip itself still has no running alarm under the existing action model; this change fixes display geometry, not its classification.
