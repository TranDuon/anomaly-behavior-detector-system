# FightWatch — Phát hiện đánh nhau trong video và camera

Ứng dụng local CPU cho **một mục tiêu duy nhất: phát hiện đánh nhau**. Video tải lên, webcam/iPhone qua DroidCam và RTSP/HTTP dùng cùng một runtime.

## Bắt đầu

```powershell
.\run_local.cmd
```

Mở http://localhost:8000. Chưa có checkpoint đánh nhau mới vẫn xem được giao diện và preview camera. LSTM Ped2 và các luật chạy nhanh/lảng vảng cũ không được dùng làm fallback.

## Thử phương pháp trước khi train lớn

Đối chứng tương tác theo thời gian: [notebook PKU + AIRTLab/RLVS](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/fight_interaction_drive.ipynb), [quy trình và giới hạn](docs/FIGHT_INTERACTION_PILOT.md). Hai cấu hình 8fps/12fps, cân bằng nguồn và hiệu chỉnh ngưỡng theo từng tập calibration; chỉ có 3 bản ghi PKU RGB, không tự thay checkpoint đang dùng.

Lượt mở rộng sau audit nhãn UBI: [notebook RLVS + AIRTLab](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/fight_large_drive.ipynb), [quy trình và giới hạn](docs/FIGHT_LARGE_TRAINING.md), [kết quả đã chạy](docs/FIGHT_LARGE_RESULTS.md). Train thực tế 1.550 clip, test riêng 202: accuracy 94,55%, precision 91,51%, recall 97,98%; riêng AIRTLab precision còn 66,67%, chưa duyệt demo/iPhone. Giao diện có **Phát lại theo thời gian thực** để kiểm thử luồng camera bằng video khi chưa nối điện thoại.

[Open in Colab — FightWatch pilot](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/fight_pilot_drive.ipynb)

Notebook tự đồng bộ code, dùng archive UBI-Fights đã có trên Drive, chỉ chọn 48 video từ phần train, lưu cache và chạy trong budget 30 phút. Pipeline thử sanity → probe → fine-tune ngắn → validation theo thời gian. Official test được giữ riêng. Không tự train toàn bộ dataset nhiều giờ.

Kết quả: `MyDrive/SENTINEL/artifacts/fight_pilot_x3d_v1/`. Checkpoint pilot chưa phải model đạt nghiệm thu. Bật **Cho phép checkpoint pilot** trong giao diện khi thử model này.

Theo phạm vi demo có người là chủ thể, dùng [notebook people-focused](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/fight_people_drive.ipynb). Bộ lọc ưu tiên ít nhất hai người đủ rõ, vẫn giữ cảnh nhiều người không đánh nhau và một lượng nhỏ mẫu âm một người. Xem [quy tắc lọc](docs/FIGHT_PEOPLE_FOCUS.md) và [kết quả thực nghiệm](docs/FIGHT_PEOPLE_RESULTS.md); chưa có checkpoint được duyệt demo.

## Gói runtime

Sau audit chất lượng nhãn, dùng [notebook curated nhỏ](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/fight_curated_pilot_drive.ipynb) và đọc [kết quả/giới hạn](docs/FIGHT_CURATED_RESULTS.md). Nhãn AI-assisted được lưu thành phiên bản riêng, không sửa annotation gốc. Có kết quả thì notebook chỉ đọc lại, không tự train. Chưa có model được duyệt demo.

Pilot cho nhóm đông trong lớp học/hành lang/ngoài trời: [mở notebook mở rộng](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/fight_expand_drive.ipynb). Không dùng NTU, không lọc bỏ mẫu đánh nhau vì đếm thiếu người; chọn thêm video UBI, sửa crop và đối chứng baseline trên cùng validation. Xem [quy trình](docs/FIGHT_EXPANDED_PILOT.md) và [kết quả đã chạy](docs/FIGHT_EXPANDED_RESULTS.md): 2.157 clip train, 568 validation, vẫn chưa đạt demo; không tự thay checkpoint.

Đặt `sentinel_fight_only_runtime.zip` tải từ Colab trong Downloads rồi chạy launcher. Gói chứa `models/fight_model.ts` và `models/fight_runtime.json`; không cần ba file LSTM cũ.

## Kiểm thử

```powershell
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Xem [quy trình và giới hạn](docs/FIGHTWATCH_WORKFLOW.md), [kết quả nghiên cứu trước đó](docs/BAO_CAO_HUONG_DI.md). Module nghiên cứu cũ vẫn giữ để đối chiếu, không thuộc đường suy luận ứng dụng mới.
