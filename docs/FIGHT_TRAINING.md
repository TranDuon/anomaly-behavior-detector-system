# Train hành vi "đánh nhau" (luồng B: khung xương + ST-GCN)

Notebook chính: `colab/train_fight_drive.ipynb` — mở thẳng trên Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/train_fight_drive.ipynb)

Sau khi mở: `Runtime > Change runtime type > L4 GPU`, rồi `Runtime > Run all`, hoặc chạy lần lượt từng cell.

**Treo máy:** giữ `AUTO_DISCONNECT = True` (mặc định). Chạy xong — hoặc một mục bị lỗi — notebook ghi lý do vào `MyDrive/SENTINEL/artifacts/<RUN_NAME>/run_log.txt`, đồng bộ Drive rồi ngắt runtime để không tốn compute unit. Model được đóng gói vào Drive thay vì mở hộp thoại tải về. Giữ tab trình duyệt mở (Colab Pro không chạy nền khi đóng tab). Chạy lại `train_fight.py --resume` sau khi đã train xong không train thêm; xóa `checkpoint_last.pt` nếu muốn train lại từ đầu.

## Khối lượng dữ liệu

Từ lần 3 (`fight_v3`), khung xương được trích trên **toàn bộ** video (video gốc không bị sửa). Tính từ file nhãn của UBI-Fights:

| Tập | Video | Toàn bộ (lần 3) | Lần 1–2 |
|---|---|---|---|
| Train, đánh nhau | 176 | 6,0 giờ (29% là đánh nhau) | 3,5 giờ: đoạn đánh nhau ±10 s |
| Train, bình thường | 757 | 60,1 giờ | tối đa 4,2 giờ: 20 s/video |
| Test, đánh nhau | 40 | 1,5 giờ | 1,5 giờ |
| Test, bình thường | 27 | 11,4 giờ | 0,45 giờ: 60 s/video |
| **Tổng** | | **79 giờ** (+ ~3 giờ RLVS) | **~9,7 giờ** |

Lần 1–2 cắt video để tiết kiệm thời gian Colab. Lần 2 cho thấy cách cắt đó làm model kém đi: mẫu âm chỉ là đầu video bình thường và vài giây quanh cuộc đánh, validation chạy trên dữ liệu đã cắt nên checkpoint và ngưỡng không khớp với tập test. Các tùy chọn cắt (`--fight-margin-seconds`, `--max-seconds-normal`, `--max-seconds-test-normal`; trong notebook là `FIGHT_MARGIN_SECONDS`, `MAX_SECONDS_NORMAL`, `TEST_NORMAL_SECONDS`, mặc định `None`) vẫn còn để chạy thử nhanh. Cache bị cắt được trích lại khi yêu cầu toàn bộ video; cache đã đầy đủ được dùng lại.

Tập test là toàn bộ tập test chuẩn (`evaluation.json` ghi `"official_protocol": true`). Để so với lần 1–2, `evaluate_fight.py --short-normal-seconds 60` báo cáo thêm nhóm `<nhóm>_short`, trong đó video bình thường chỉ tính 60 s đầu như tập test cũ.

Các bước giảm tài nguyên khác:

- **FP16 trên GPU** (`--half auto`) và **bỏ giải mã màu** (`grab()`) cho frame không được lấy mẫu.

- **Trích khung xương song song:** `--shard k/n`, notebook chạy `POSE_SHARDS` tiến trình (theo số CPU, tối đa 6) cùng dùng một GPU, vì tracking phải chạy tuần tự trong từng video nên một tiến trình để GPU rảnh phần lớn thời gian.
- **Dựng cửa sổ dạng ma trận:** khoảng cách và tầm với của mọi cặp người tính một lần thay vì lặp từng cặp. Trên UBI-Fights thật: cảnh quán bar 19 người/frame 79 → 18 ms/cửa sổ, cảnh ít người 8 → 4,5 ms (CPU máy local).
- **Đánh giá bước 0,3 giây** (`--stride 3`): chấm ít hơn 3 lần số cửa sổ; frame ở giữa được nội suy.
- **Train dùng mọi CPU** để dựng cửa sổ (`--num-workers`), GPU không phải chờ.

## Dataset

| Dataset | Vai trò | Nguồn |
|---|---|---|
| UBI-Fights | Bộ chính: train + test chuẩn (`test_videos.csv`), nhãn theo frame | https://socia-lab.di.ubi.pt/EventDetection (zip 8,1 GB, chứng chỉ SSL đã hết hạn nên tải bằng `--no-check-certificate`) |
| Surveillance Camera Fight | Test chéo theo clip | https://github.com/sayibet/fight-detection-surv-dataset (MIT) |
| Real Life Violence Situations | Train thêm (bật mặc định từ lần 3, cần `kaggle.json`) | Kaggle `mohamedmustafa/real-life-violence-situations-dataset` (chưa có giấy phép rõ, chỉ dùng nghiên cứu) |
| UCF-Crime (Fighting + Normal, tập test) | Tùy chọn: test chéo theo frame | https://www.crcv.ucf.edu/projects/real-world/ (tải thủ công, kèm `Temporal_Anomaly_Annotation*.txt`) |

RWF-2000 không dùng được: nhóm tác giả đã gỡ video và cấm phân phối lại khi chưa có sự đồng ý của SMIIP Lab.

## Các bước

```text
dedup_videos.py  →  extract_pose.py  →  train_fight.py  →  evaluate_fight.py
```

Mọi script nhận dataset theo dạng `--dataset tên:đường_dẫn[:split]`, với tên là `ubi`, `scf`, `rlvs` hoặc `ucf`.

1. **Loại video trùng** (`scripts/dedup_videos.py`): so dHash của 16 frame mỗi video. Chỉ cặp trùng giữa hai dataset khác nhau bị loại khỏi train (giữ bản ở tập test). Cặp trong cùng dataset chỉ được ghi vào báo cáo, vì camera cố định quay hai thời điểm khác nhau cũng có hash gần giống, và để giữ tập chia chuẩn của tác giả.
2. **Trích khung xương** (`scripts/extract_pose.py`): YOLOv8n-pose ở `imgsz=1280` + BoT-SORT có ReID (`configs/botsort_reid.yaml`), lấy mẫu 10 FPS; `--shard k/n` để chạy song song. Mỗi video một file `<pose-root>/<dataset>/<key>.npz`; chạy lại sẽ bỏ qua video đã có đủ phần được yêu cầu.
3. **Train** (`scripts/train_fight.py`): cửa sổ 24 mẫu (2,4 giây); nhãn cửa sổ = đánh nhau nếu ≥ 50% frame là đánh nhau, bình thường nếu 0%, bỏ qua nếu ở giữa. Cửa sổ bình thường cách đoạn đánh nhau dưới 1 s cũng bị bỏ (`--boundary-guard`), vì nhãn ranh giới không chính xác. Mỗi cửa sổ là một túi gồm tối đa 16 cặp người trong bán kính 1,5 lần chiều cao người, xếp theo mức tương tác (chuyển động tay/chân so với thân, chia cho độ gần hoặc khoảng cách từ tay/chân tới đầu/thân người kia). Khi train, 1/4 số suất chọn ngẫu nhiên ngoài top để cặp đánh nhau bị xếp hạng thấp vẫn vào túi. Điểm của cửa sổ gộp điểm các cặp bằng smooth max chuẩn hóa theo số cặp, `(1/r)·log(trung bình exp(r·l))` với `--pool-r 5` (MIL): khác với max, điểm không tăng khi cảnh có thêm cặp người bình thường. Model cũ (không có `pool_r`) vẫn dùng max.
   - **Lấy mẫu:** mỗi epoch `--samples-per-epoch` cửa sổ: 50% đánh nhau, 25% bình thường trong video có đánh nhau (cùng khung cảnh), 25% video bình thường. Sau `--hard-negative-warmup` epoch, `--hard-negative-fraction` của mỗi nhóm bình thường được lấy tỉ lệ với điểm model chấm gần nhất (hard-negative mining).
   - **Validation:** 15% video train, chia theo video. Chấm mọi cửa sổ của video đầy đủ (`--val-stride`), trải điểm ra frame như lúc đánh giá; checkpoint chọn theo AUC theo frame, ngưỡng theo balanced accuracy trên frame. Log có thêm AUC trung vị trong từng video đánh nhau. Checkpoint lưu mỗi epoch, `--resume` để train tiếp.
4. **Đánh giá** (`scripts/evaluate_fight.py`): UBI-Fights và UCF-Crime theo frame (AUC, AP, F1…), mọi dataset theo clip, và riêng nhóm video đông người (`<dataset>_crowd`, trung bình ≥ 5 người/frame, chỉnh bằng `--crowd-threshold`). Mỗi nhóm theo frame có thêm `fight_video_frame_auc_median`: AUC trung vị trong từng video đánh nhau, cho biết model có tìm được đoạn đánh nhau trong video hay chỉ nhận ra video đánh nhau. Ngưỡng lấy từ validation (`fight_threshold.json`), không chỉnh theo nhãn test. `predict_windows(..., return_pairs=True)` trả thêm ID cặp người có điểm cao nhất để dashboard khoanh đúng người.

## Artifact

```text
models/fight_stgcn.pt          model + thông số cửa sổ (metadata.window)
models/fight_threshold.json    ngưỡng từ validation
```

Khi chạy thật phải dùng cùng `pose_model`, `pose_fps` và tracker như lúc trích khung xương (đọc từ `core/config.py`).

## Vì sao YOLOv8n-pose ở imgsz 1280

Số người phát hiện trung bình mỗi frame (10 frame mỗi video; 4 video UBI-Fights lấy trong đoạn đánh nhau) và thời gian trên CPU máy local:

| Model | imgsz | UMN đám đông 320×240 | vtest 768×576 | F_41 (UBI) | F_88 (UBI) | ms/frame CPU |
|---|---|---|---|---|---|---|
| n | 640 | 2,6 | 0,9 | 5,1 | 12,4 | 37 |
| n | 1280 | 9,6 | 5,5 | 10,6 | 17,8 | 106 |
| s | 640 | 2,0 | 1,3 | 5,8 | 13,7 | 79 |
| s | 1280 | 10,4 | 5,6 | 10,5 | 19,0 | 412 |

Ở 640 cả hai model bỏ sót phần lớn người cao dưới ~100 px. Ở 1280, bản `n` tìm được gần bằng bản `s` mà nhanh gấp ~4 lần. Trích khung xương có tracking ở cấu hình này đạt ~7,5 mẫu/giây trên CPU.

Trong đoạn đánh nhau của 4 video UBI-Fights trên, trung bình 5–19 người/frame; 225/229 cửa sổ đánh nhau có ít nhất một cặp ứng viên. Số cặp ứng viên mỗi cửa sổ là 13–130 (bán kính 2,5) nên phải xếp hạng; ST-GCN chấm 16 cặp mất ~87 ms trên CPU, 32 cặp ~230 ms.

Hạn chế đã biết: trên cảnh quán bar rất đông (F_88), cặp xếp hạng cao nhất theo heuristic không trùng nhóm đang xô xát khi kiểm tra bằng mắt. Model ST-GCN chấm cả 16 cặp nên vẫn có thể bắt được, nhưng nhóm `_crowd` trong báo cáo đánh giá là nơi kiểm chứng điều đó.

Đổi model hoặc `imgsz` thì phải trích khung xương và train lại, vì lúc chạy thật phải dùng đúng cấu hình lúc train.
