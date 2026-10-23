# Train hành vi "đánh nhau" (luồng B: khung xương + ST-GCN)

Notebook chính: `colab/train_fight_drive.ipynb` — mở thẳng trên Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/train_fight_drive.ipynb)

Sau khi mở: `Runtime > Change runtime type > T4 GPU` (hoặc L4/A100 với Colab Pro), rồi `Runtime > Run all`, hoặc chạy lần lượt từng cell.

## Dataset

| Dataset | Vai trò | Nguồn |
|---|---|---|
| UBI-Fights | Bộ chính: train + test chuẩn (`test_videos.csv`), nhãn theo frame | https://socia-lab.di.ubi.pt/EventDetection (zip 8,1 GB, chứng chỉ SSL đã hết hạn nên tải bằng `--no-check-certificate`) |
| Surveillance Camera Fight | Test chéo theo clip | https://github.com/sayibet/fight-detection-surv-dataset (MIT) |
| Real Life Violence Situations | Tùy chọn: train thêm | Kaggle `mohamedmustafa/real-life-violence-situations-dataset` (chưa có giấy phép rõ, chỉ dùng nghiên cứu) |
| UCF-Crime (Fighting + Normal, tập test) | Tùy chọn: test chéo theo frame | https://www.crcv.ucf.edu/projects/real-world/ (tải thủ công, kèm `Temporal_Anomaly_Annotation*.txt`) |

RWF-2000 không dùng được: nhóm tác giả đã gỡ video và cấm phân phối lại khi chưa có sự đồng ý của SMIIP Lab.

## Các bước

```text
dedup_videos.py  →  extract_pose.py  →  train_fight.py  →  evaluate_fight.py
```

Mọi script nhận dataset theo dạng `--dataset tên:đường_dẫn[:split]`, với tên là `ubi`, `scf`, `rlvs` hoặc `ucf`.

1. **Loại video trùng** (`scripts/dedup_videos.py`): so dHash của 16 frame mỗi video. Chỉ cặp trùng giữa hai dataset khác nhau bị loại khỏi train (giữ bản ở tập test). Cặp trong cùng dataset chỉ được ghi vào báo cáo, vì camera cố định quay hai thời điểm khác nhau cũng có hash gần giống, và để giữ tập chia chuẩn của tác giả.
2. **Trích khung xương** (`scripts/extract_pose.py`): YOLOv8n-pose ở `imgsz=1280` + BoT-SORT có ReID (`configs/botsort_reid.yaml`), lấy mẫu 10 FPS. Mỗi video một file `<pose-root>/<dataset>/<key>.npz`; chạy lại sẽ bỏ qua video đã có. `--max-seconds-normal` cắt bớt video bình thường của tập train.
3. **Train** (`scripts/train_fight.py`): cửa sổ 24 mẫu (2,4 giây); nhãn cửa sổ = đánh nhau nếu ≥ 50% frame là đánh nhau, bình thường nếu 0%, bỏ qua nếu ở giữa. Mỗi cửa sổ là một túi gồm tối đa 16 cặp người trong bán kính 1,5 lần chiều cao người, xếp theo mức tương tác (chuyển động tay/chân so với thân, chia cho độ gần hoặc khoảng cách từ tay/chân tới đầu/thân người kia). Khi train, 1/4 số suất chọn ngẫu nhiên ngoài top để cặp đánh nhau bị xếp hạng thấp vẫn vào túi. Điểm của cửa sổ = điểm lớn nhất trong các cặp (MIL). Validation chia theo video từ tập train; checkpoint lưu mỗi epoch, `--resume` để train tiếp.
4. **Đánh giá** (`scripts/evaluate_fight.py`): UBI-Fights và UCF-Crime theo frame (AUC, AP, F1…), mọi dataset theo clip, và riêng nhóm video đông người (`<dataset>_crowd`, trung bình ≥ 5 người/frame, chỉnh bằng `--crowd-threshold`). Ngưỡng lấy từ validation (`fight_threshold.json`), không chỉnh theo nhãn test. `predict_windows(..., return_pairs=True)` trả thêm ID cặp người có điểm cao nhất để dashboard khoanh đúng người.

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
