# Train hành vi đánh nhau — luồng A (đặc trưng hình ảnh)

Notebook: [`colab/train_visual_fight_drive.ipynb`](../colab/train_visual_fight_drive.ipynb). Mở trên Colab Pro, chọn `Runtime > Change runtime type > L4 GPU`, giữ `AUTO_DISCONNECT = True` rồi `Runtime > Run all`. Kết quả và log nằm ở `MyDrive/SENTINEL/artifacts/<RUN_NAME>/`.

## Vì sao có luồng A

Luồng B (khung xương + ST-GCN, [FIGHT_TRAINING.md](FIGHT_TRAINING.md)) sau 3 lần train đạt AUC theo frame 0,81 trên toàn bộ tập test UBI-Fights, nhưng AUC trung vị trong từng video đánh nhau chỉ 0,54. Chỉ đếm số người đã phân biệt video đánh nhau với video bình thường tốt hơn model (0,80 so với 0,76). Model nhận ra *cảnh* có đánh nhau (đông người, khung xương rõ) chứ không nhận ra *lúc* đánh nhau. Luồng A dùng đặc trưng hình ảnh pretrain trên Kinetics, nơi các lớp như đấm bốc, vật nhau, tát đã được học sẵn.

## Pipeline

1. **Trích đặc trưng** (`scripts/extract_visual.py`, chạy một lần trên Colab GPU): S3D (torchvision, Kinetics-400, đóng băng). Mỗi clip 16 frame lấy ở 15 FPS (~1 s), mỗi giây một clip, cả khung hình resize về 224×224 (không cắt mép). Đặc trưng 2048 chiều: bản đồ 1024 kênh lấy trung bình theo thời gian và không gian, nối với trung bình theo thời gian rồi max theo không gian (đám đánh nhau nhỏ trong khung hình rộng không bị trung bình hóa mất). Mỗi clip lưu kèm năng lượng chuyển động (trung bình |sai khác| giữa các frame xám 64×36). Mỗi video một file `<feature-root>/<dataset>/<key>.npz`; luôn trích toàn bộ video, video dài được xử lý dạng luồng; `--shard k/n` chạy song song.
2. **Train** (`scripts/train_visual_fight.py`): `FightTemporalHead` = LayerNorm → Linear 2048→256 → 3 khối tích chập nhân quả (kernel 3, dilation 1/2/4; 15 s ngữ cảnh) → logit mỗi clip. Nhãn clip: đánh nhau nếu ≥ 50% frame là đánh nhau; bình thường nếu không có frame nào và cách đoạn đánh nhau ≥ `--boundary-guard` giây; còn lại bỏ qua.
   - **Lấy mẫu:** chuỗi `--sequence-length` clip liên tiếp; `--fight-share` chuỗi lấy từ video có đánh nhau, 80% trong số đó chứa đoạn đánh nhau (nên có cả đoạn đánh và đoạn yên của cùng cảnh).
   - **Loss:** BCE cân bằng hai lớp + `--rank-weight` × loss xếp hạng trong từng chuỗi (mọi clip đánh nhau phải có điểm cao hơn mọi clip yên của cùng video).
   - **Validation:** 15% video train (chia theo video), chấm trên video đầy đủ, trải điểm ra frame như lúc test. Checkpoint chọn theo `--select-on` (mặc định trung bình của AUC gộp và AUC trung vị trong video), ngưỡng theo balanced accuracy trên frame. Đường cơ sở năng lượng chuyển động được tính trên cùng frame.
3. **Đánh giá** (`scripts/evaluate_visual_fight.py`): cùng giao thức với luồng B: UBI-Fights theo frame trên toàn bộ tập test và nhóm `_short`; nhóm `_crowd` (≥ 5 người/frame, đếm từ khung xương của luồng B khi có `--pose-root`); Surveillance Camera Fight theo clip; AUC trung vị trong từng video đánh nhau; mọi chỉ số kèm bản `motion_baseline_*`.

## Chạy thật (CPU)

`StreamingFightScorer` nhận từng frame của camera, lấy mẫu đúng lưới 15 FPS như lúc trích, chạy S3D mỗi giây một lần và chạy head trên 15 clip gần nhất. Test xác nhận điểm chạy trực tiếp bằng điểm offline của cùng clip. Đo trên Ryzen 7 7735HS (8 luồng): S3D ~390 ms/clip, cả pipeline (giải mã + S3D) ~4,7 lần thời gian thực, tức ~1/5 CPU cho một camera.

## Artifact

| File | Nội dung |
|---|---|
| `visual_fight_head.pt` | Đầu dự đoán; metadata ghi cấu hình trích đặc trưng và số clip ngữ cảnh |
| `visual_fight_threshold.json` | Ngưỡng từ validation, kèm đường cơ sở chuyển động |
| `config.json`, `training_history.json`, `split.json` | Cấu hình, lịch sử từng epoch, danh sách video train/validation |
| `evaluation/evaluation.json`, `per_video.csv`, `*.png` | Kết quả test, từng video, đường ROC/PR, timeline |
| `sentinel_visual_fight_runtime.zip` | Gói `models/` cho máy chạy thật (trọng số S3D do torchvision tự tải) |
