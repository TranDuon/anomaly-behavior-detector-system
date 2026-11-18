# Kiến trúc mở rộng hành vi

Tài liệu cho Giai đoạn 0: cách thêm hành vi mới (phá hoại tài sản, cầm vật sắc nhọn…) mà **không làm thay đổi kết quả 4 nhánh đã đạt** (strike, grapple, fast_running, fall).

## Nguyên tắc

- Backend mặc định `mmaction_separate_actions` không đổi. Hành vi mới chỉ chạy trong backend `mmaction_separate_actions_ext`.
- Hành vi mới chạy trong **extension worker**, một tiến trình và môi trường Python riêng (`extensions/env`, torch mới). Core worker (`mmaction2_demo/env`, torch 2.0.1) giữ nguyên.
- Runtime **không chờ** extension. Extension chậm, crash hay trả dữ liệu sai chỉ làm tắt extension; core vẫn chạy như cũ.
- Extension chỉ **thêm** nhãn vào `active_behaviors`, `action_states`, `action_scores`, `action_regions` và khoá `extensions`. Không ghi đè nhãn core. `state` tổng vẫn chỉ tính từ core.

## Thành phần

| File | Vai trò |
|---|---|
| `fight/behaviors.py` | Registry nhãn hiển thị, thứ tự ghi sự kiện, thời gian gate. 4 entry core giữ đúng giá trị cũ (khoá bằng `tests/test_behavior_registry.py`). |
| `fight/extension_client.py` | Phía server: khởi động worker, gửi frame qua shared memory riêng, đọc kết quả mới nhất, bỏ kết quả cũ hơn 1.5 s. Không bao giờ ném lỗi vào core. |
| `extensions/worker.py` | Tiến trình extension: giao thức JSON `infer` / `reset` / `close`, mỗi extension lỗi chỉ báo lỗi cho riêng nó. |
| `extensions/base.py` | Lớp `Extension` (`load`, `observe`) + registry; có `selftest` không cần model để kiểm IPC. |
| `extensions/setup.ps1` | Tạo `extensions/env` (Python 3.12, torch 2.6 cu124) từ `requirements-extensions.txt`. |
| `scripts/compare_behavior_traces.py` | Golden trace: so 4 nhánh core với bản ghi trước khi sửa. |

## Luồng một bước

1. Core chạy như cũ và trả kết quả (`push_separate`).
2. Nếu đến lượt (mặc định cứ 2 lần suy luận core thì 1 lần; core chậm hơn 450 ms/bước thì 4 lần mới 1), runtime chép 9 frame sang shared memory **riêng** của extension và gửi yêu cầu. Worker còn bận thì bỏ lượt.
3. Runtime lấy kết quả extension mới nhất (nếu không cũ hơn 1.5 s), cập nhật `EventGate` riêng của từng extension, rồi thêm nhãn vào kết quả.

## Thêm một hành vi mới

1. Tạo `extensions/<ten>.py`: lớp kế thừa `Extension`, đặt `name`, `display_label`, `threshold`, `confirm_seconds`, `release_seconds`; viết `load()` và `observe()`; gắn `@register`.
2. Thêm tên module vào `KNOWN_MODULES` trong `extensions/worker.py`.
3. Thư viện mới thêm vào `requirements-extensions.txt` (ghim phiên bản), chạy lại `extensions/setup.ps1`.
4. Checkpoint đặt trong `weights/extensions/` qua script `install_<ten>.py` (kiểm zip, SHA, spec).
5. Bật trong `fight/mmaction_runtime.py` (`EXTENSION_NAMES`).
6. Chạy vòng kiểm tra bên dưới.

## Vòng kiểm tra sau mỗi thay đổi

```bash
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```
Chỉ được còn đúng các test trong `tests/KNOWN_FAILURES.txt` (baseline: 12 failed, 400 passed, 3 skipped).

```bash
.venv\Scripts\python.exe scripts\compare_behavior_traces.py compare
.venv\Scripts\python.exe scripts\compare_behavior_traces.py compare --backend mmaction_separate_actions_ext
```
Cả hai phải báo 13/13 video khớp golden.

## Khoá không được chạm

- `mmaction2_demo/action_engine.py`, `mmaction2_demo/worker.py`, `separate_actions_engine.LABELS`.
- `weights/separate_actions_calibration.json` và mọi checkpoint đang dùng. File mới luôn có tên mới.
- Không cài thêm gì vào `.venv` hay `mmaction2_demo/env`.

## Golden trace

- Danh sách video: `tests/golden/behavior_reference.json` (đường dẫn + SHA-256). Bản sao video ở `tests/golden/videos/` (không đưa vào git).
- Kết quả gốc: `tests/golden/traces/*.json`, ghi một lần ở commit `36d5c73` bằng `compare_behavior_traces.py record`.
- So sánh: trạng thái, sự kiện, số người, nhãn active và trạng thái từng nhãn phải giống hệt; điểm lệch ≤ 0.001; box lệch ≤ 2 px.

## Cầm súng ngắn (`handgun`) — khung, chưa có checkpoint

- `extensions/handgun/extension.py`: lớp con của `SharpObjectExtension`, cùng cách hand-crop (pose → crop quanh tay → ResNet nhị phân → veto → YOLO-World vẽ khung). Chỉ khác cấu hình: nhãn `HANDGUN`, từ vựng locator `handgun/pistol/revolver`, **không** veto `cell phone`, `remote`, `hair drier`, `mouse` (súng trong tay hay bị COCO gọi nhầm là các vật này), không dùng veto cây lau nhà (CLIP).
- Pose chạy một lần mỗi bước cho cả `sharp_object` và `handgun` (cache theo mảng `frames` của worker + file/cấu hình pose).
- `fight/behaviors.WEAPON_BEHAVIORS`: các extension "cầm vũ khí" — overlay chỉ tô người cầm và vẽ khung vũ khí; runtime giữ khung riêng cho từng extension.
- Cài: `extensions\env\Scripts\python.exe -m extensions.handgun.install_handgun handgun_<RUN_TAG>.zip` (zip từ `train_hand_classifier.py --stage export --name handgun`). Chưa cài thì runtime không bật `handgun`, mọi nhánh khác không đổi.
- Dữ liệu: `datasets/handgun/class_map.json` (Granada pistol, Open Images `Handgun`; hard negative: điện thoại, remote, máy sấy, chuột, máy khoan…).
