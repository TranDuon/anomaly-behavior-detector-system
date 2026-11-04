# FightWatch — Phát hiện đánh nhau trong video và camera

Ứng dụng nhận video tải lên, webcam/iPhone qua DroidCam và RTSP/HTTP. Hiện có năm lựa chọn model trong giao diện và API:

| Lựa chọn | Backend API | Trọng số |
|---|---|---|
| FightWatch X3D-S + pose · CPU | `legacy` | `models/fight_model.ts`, `models/yolov8n-pose.pt` |
| MMAction2 gốc · GPU | `mmaction_original` | `action_slowonly_r101.pth`, `person_faster_rcnn_r50.pth` |
| MMAction2 interaction v1 · GPU | `mmaction_interaction_v1` | Hai trọng số MMAction2 gốc cùng `interaction.pt` và `interaction.json` |
| MMAction2 đánh / vật / ngã chung · GPU | `mmaction_fall_wrestle` | Hai trọng số MMAction2 gốc cùng `interaction_fall_wrestle.pt` và `interaction_fall_wrestle.json` |
| MMAction2 ngã sớm + chuyển xanh · Re-DID pilot · GPU | `mmaction_transition_redid` | Hai trọng số MMAction2 gốc cùng `interaction_transition.pt` và `interaction_transition.json` |

Trọng số MMAction2 nằm trong `mmaction2_demo/weights/`. Các nhánh interaction dùng backbone/AVA gốc đóng băng và head riêng; checkpoint mới cảnh báo chung, chưa phân biệt ba hành vi. Xem [danh mục model và nguồn checkpoint](docs/MODELS.md).

## Bắt đầu

```powershell
.\run_local.cmd
```

Mở http://localhost:8000, chọn model trước khi bắt đầu nguồn. Các checkpoint hiện là pilot: bật **Cho phép checkpoint pilot** để thử. Chưa có bằng chứng nghiệm thu camera/điện thoại thực tế. Preview camera vẫn dùng được khi chưa bật pilot.

MMAction2 cần môi trường GPU riêng: xem [hướng dẫn cài và chạy](mmaction2_demo/README_VI.md). CLI cũng cho phép chọn năm backend:

```powershell
.venv\Scripts\python.exe main.py run --source video.mp4 --allow-pilot --backend mmaction_interaction_v1
.venv\Scripts\python.exe main.py run --source video.mp4 --allow-pilot --backend mmaction_fall_wrestle
.venv\Scripts\python.exe main.py run --source video.mp4 --allow-pilot --backend mmaction_transition_redid
```

## Cài interaction v1 từ Drive

Tải gói ZIP trong [run v1](https://drive.google.com/drive/folders/1hk5eNnqToJezDlI2otsPOEFIWIBPzH0R), dừng phiên nhận diện rồi kéo ZIP vào `mmaction2_demo/install_branch.cmd`. Installer chỉ nhận head v1, kiểm tra SHA-256, nguồn backbone, preprocessing, báo cáo hoàn tất và cấu trúc state dict trước khi thay file; lưu bản cũ trong `mmaction2_demo/results/branch_backup_*`.

## Kiểm thử

```powershell
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Các module, notebook và báo cáo nghiên cứu trước đây được giữ để tham khảo lịch sử; chúng không nằm trong danh sách model được hỗ trợ của ứng dụng. LSTM, ST-GCN, interaction v2, refinement v3 và warm-start v4 không phải lựa chọn runtime hiện tại. API từ chối các backend `mmaction_tuned` và `mmaction_tuned_refined` cũ.

Đang chuẩn bị dữ liệu cho vật nhau và ngã nói chung: [tình trạng bước 1](docs/FALL_GENERAL_DATA_STEP1.md). Chưa có checkpoint mới.

Notebook fine-tune trên Colab: [hướng dẫn và giới hạn](docs/MMACTION_GENERAL_FALL_COLAB.md).

Đợt cải thiện ngã sớm và trả xanh sau xô xát: [quy trình review và trace](docs/ACTION_TRANSITION_V2.md), [notebook Colab v2 đang dùng](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/mmaction_action_transition_v2.ipynb), [notebook Re-DID mới](https://colab.research.google.com/github/TranDuon/anomaly-behavior-detector-system/blob/main/colab/mmaction_action_transition_redid_v1.ipynb) và [hướng dẫn dữ liệu Re-DID](docs/REDID_TRANSITION_DATA.md). Vật thể thao là mẫu âm theo phạm vi hiện tại.

Xem thêm [quy trình FightWatch](docs/FIGHTWATCH_WORKFLOW.md) và [nghiên cứu trước đó](docs/BAO_CAO_HUONG_DI.md).
