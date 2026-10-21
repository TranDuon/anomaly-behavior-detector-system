# Training dataset trên Google Colab và Google Drive

Notebook chính: `colab/train_ucsd_ped2_drive.ipynb`.

## Cấu trúc Drive

Sau khi chạy các cell khởi tạo, Drive có cấu trúc:

```text
MyDrive/SENTINEL/
├── repo/
│   └── anomaly-behavior-detector-system/
├── datasets/
│   └── ucsd_ped2/
│       └── UCSD_Anomaly_Dataset.tar.gz
├── features/
│   └── ucsd_ped2/
│       ├── smoke/
│       ├── train/
│       └── test_stride1/
└── artifacts/
    └── ucsd_ped2/
        ├── lstm_autoencoder.pt
        ├── feature_scaler.npz
        ├── threshold.json
        ├── config.json
        ├── split.json
        ├── training_history.json
        └── evaluation/
            ├── evaluation.json
            ├── frame_scores.csv
            ├── per_clip_metrics.csv
            ├── roc_curve.png
            ├── precision_recall_curve.png
            ├── confusion_matrix.png
            └── score_timelines.png
```

Repository và artifacts nằm trên Drive để tồn tại qua nhiều phiên Colab. Dataset
được copy và giải nén vào `/content/datasets` trong từng phiên để tránh đọc hàng
nghìn ảnh TIFF trực tiếp từ Drive.

## Khởi tạo lần đầu

1. Fork hoặc push source code lên repository GitHub của bạn.
2. Upload notebook lên Colab hoặc mở notebook từ repository.
3. Chọn `Runtime > Change runtime type > T4 GPU`.
4. Sửa `REPO_URL` trong cell cấu hình nếu repository không phải repository mặc định.
5. Chạy cell mount Drive.
6. Chạy cell clone lần đầu.
7. Đặt `UCSD_Anomaly_Dataset.tar.gz` vào đường dẫn notebook in ra:

```text
MyDrive/SENTINEL/datasets/ucsd_ped2/UCSD_Anomaly_Dataset.tar.gz
```

8. Chạy smoke test một clip trước khi trích xuất toàn bộ dataset.
9. Kiểm tra `detections` và `sequences` trong smoke manifest lớn hơn 0.
10. Chạy full extraction, training, test extraction, evaluation và cell cài artifacts.

## Đồng bộ code sau mỗi lần push

Notebook có cell riêng chạy:

```bash
git fetch origin main
git checkout main
git reset --hard origin/main
```

GitHub/local là nguồn chuẩn, nên cell bỏ các thay đổi tracked do Colab tự lưu trước khi
đồng bộ. Các file `.env`, checkpoint, feature cache và dataset đều được đặt ngoài Git
hoặc bị ignore nên không bị xóa.

## Các giai đoạn độc lập

### Trích xuất feature

```bash
python scripts/extract_features.py \
  --input /content/datasets/.../UCSDped2/Train \
  --output-dir /content/drive/MyDrive/SENTINEL/features/ucsd_ped2/train \
  --stride 10
```

Mỗi clip tạo một file `.npz`. Chạy lại cùng lệnh sẽ bỏ qua clip đã cache, cho phép
resume sau khi Colab ngắt phiên.

### Train từ feature cache

```bash
python scripts/train_features.py \
  --features-dir /content/drive/MyDrive/SENTINEL/features/ucsd_ped2/train \
  --output-dir /content/drive/MyDrive/SENTINEL/artifacts/ucsd_ped2 \
  --dataset-name "UCSD Ped2" \
  --epochs 50 \
  --batch-size 64 \
  --val-ratio 0.2 \
  --seed 42 \
  --device cuda
```

Validation được chia theo clip thay vì chia ngẫu nhiên từng sliding window. Scaler
chỉ được fit trên các clip training. Threshold được calibration từ reconstruction
error của validation normal.

### Đánh giá frame-level trên UCSD Ped2

```bash
python scripts/extract_features.py \
  --input /content/datasets/.../UCSDped2/Test \
  --output-dir /content/drive/MyDrive/SENTINEL/features/ucsd_ped2/test_stride1 \
  --stride 1

python scripts/evaluate_ucsd_ped2.py \
  --features-dir /content/drive/MyDrive/SENTINEL/features/ucsd_ped2/test_stride1 \
  --test-root /content/datasets/.../UCSDped2/Test \
  --model /content/drive/MyDrive/SENTINEL/artifacts/ucsd_ped2/lstm_autoencoder.pt \
  --scaler /content/drive/MyDrive/SENTINEL/artifacts/ucsd_ped2/feature_scaler.npz \
  --threshold /content/drive/MyDrive/SENTINEL/artifacts/ucsd_ped2/threshold.json \
  --output-dir /content/drive/MyDrive/SENTINEL/artifacts/ucsd_ped2/evaluation \
  --device cuda
```

Protocol đánh giá dùng pixel mask `Test*_gt` để tạo nhãn frame-level. Score của một
frame là reconstruction error lớn nhất trong các track kết thúc tại frame đó. Frame
không có score trực tiếp được nội suy trong phạm vi clip; báo cáo luôn ghi
`coverage_before_interpolation` để thể hiện tỷ lệ score quan sát thật. Threshold được
giữ nguyên từ validation normal, không tối ưu lại bằng nhãn test.

Các metrics gồm accuracy, balanced accuracy, precision, recall, specificity, F1,
AUROC, Average Precision và confusion matrix. Accuracy không nên được đọc riêng vì
tập anomaly thường mất cân bằng nhãn.

## Artifact bắt buộc khi inference

Ba file sau phải được deploy cùng nhau:

```text
models/lstm_autoencoder.pt
models/feature_scaler.npz
models/threshold.json
```

Thiếu scaler sẽ làm input inference khác phân phối input lúc train. Thiếu threshold
sẽ khiến hệ thống quay lại giá trị `ANOMALY_THRESHOLD` trong `.env`.

## Khi smoke test không có sequence

Kiểm tra theo thứ tự:

1. Ảnh TIFF có đọc được bằng OpenCV không.
2. `detections` trong manifest có lớn hơn 0 không.
3. Giảm `YOLO_CONF_THRESHOLD` từ `0.40` xuống `0.25`.
4. Thử `YOLO_MODEL=yolov8s.pt` nếu người trong ảnh quá nhỏ.
5. Tăng `--max-frames` vì một track phải đủ 30 frame mới tạo sequence.

Không nên chạy full dataset trước khi smoke test tạo được feature hợp lệ.
