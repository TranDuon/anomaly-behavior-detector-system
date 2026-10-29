# Tiếp tục train checkpoint MMAction để giảm báo nhầm cảnh đứng gần

Notebook: [mmaction_warmstart_drive.ipynb](../colab/mmaction_warmstart_drive.ipynb).
Chọn GPU trong Colab rồi **Run all**, cấp quyền Drive khi được hỏi.

## Đầu vào đã có trên Drive

- `SENTINEL/artifacts/mmaction_indoor_review_v2/indoor_review_pack.zip`
- `SENTINEL/artifacts/mmaction_interaction_v1/mmaction_interaction_pilot.zip`

Nếu checkpoint nằm cạnh gói indoor, notebook tự dùng bản đó khi đường dẫn v1 không tồn tại.
Notebook kiểm hash checkpoint, metadata và manifest trong gói trước khi train. Trọng số
AVA/detector chính thức được kiểm SHA-256 và lưu lại trên Drive để dùng lần sau.

## Điểm khác so với refinement v3

V4 cập nhật trực tiếp trọng số head tương tác đã học trong `interaction.pt`.
Với head v1, lớp mới được mở rộng để nhận thêm khoảng cách người, chuyển động vùng
người và điểm AVA; các trọng số kế thừa và bộ chuẩn hóa visual được giữ nguyên.
Logits ngay sau khởi tạo khớp model cha trong kiểm thử. Với head v2, nạp strict
toàn bộ trọng số. Không thay head cũ bằng classifier ngẫu nhiên.

Backbone SlowOnly R101/AVA và detector vẫn được khóa. Đây là fine-tune nhánh
tương tác của hệ thống MMAction hiện có, chưa phải fine-tune end-to-end backbone AVA.
Hướng dẫn chính thức cũng khuyến nghị learning rate nhỏ khi
[fine-tune MMAction2](https://mmaction2.readthedocs.io/en/latest/user_guides/finetune.html).

Train tăng trọng số các negative đứng gần, tiếp cận, rời đi, ôm, bắt tay, high-five
và đưa đồ. Từ epoch 4, negative còn bị chấm cao được lấy mẫu nhiều hơn. Loss negative
áp dụng cho mọi ứng viên người/cặp người, tránh chỉ hạ điểm một ứng viên cao nhất.
Không thêm luật cấm báo đánh nhau chỉ vì hai người đứng gần hay ít chuyển động.

## Chạy và chạy tiếp

Mặc định `EPOCHS = 150`, `LEARNING_RATE = 0.0001`, early stopping 20 epoch và
`TRAIN_MINUTES = 240` cho mỗi lượt train/evaluate. Cài môi trường tính riêng.
Không giới hạn dữ liệu chỉ để ép xong trong 45 phút. Colab có thể ngắt runtime
trước ngân sách này; con số trên không bảo đảm thời gian GPU được cấp.

Kết quả được lưu tại:

```text
MyDrive/SENTINEL/artifacts/mmaction_warmstart_v4/
  feature_cache/<extraction-fingerprint>/
  public_weights/
  runs/<source-git-sha>/indoor_close_seed42/
    last.pt
    interaction.pt
    interaction.json
    history.json
    selection_frozen.json
    coverage.json
    parent_exposure.json
    test_predictions.json
    report.json
    mmaction_warmstart_v4_pilot.zip
```

Giữ nguyên notebook, `RUN_TAG`, số epoch và learning rate rồi Run all để tiếp tục.
`last.pt` lưu model, optimizer, trạng thái RNG và best state mỗi epoch. Cache ghi
từng cửa sổ, không phải chờ cuối lượt mới chép ZIP lên Drive. Nếu ngắt giữa epoch,
chạy lại từ đầu epoch đó. Đổi cấu hình thí nghiệm thì đổi `RUN_TAG`.

Checkpoint được chọn trên development; calibration chọn ngưỡng; test chạy sau
khi ghi `selection_frozen.json`. Nếu lỗi khi đánh giá test, chạy lại chỉ tiếp tục
đánh giá, không train/chọn lại checkpoint. `best_epoch = 0` nghĩa là chưa có epoch
nào vượt model cha theo tiêu chí development, không được gọi là đã cải thiện.
Notebook chỉ tự giải phóng GPU sau khi gói kết quả hoàn tất; lỗi giữ log để xem.

## Đọc kết quả

Calibration phải đồng thời đạt precision ≥ 0,75, recall ≥ 0,60, FPR chung ≤ 0,10
và FPR nhóm đứng gần/tiếp cận/rời đi/đi cùng ≤ 0,10. Thiếu negative nhóm này thì
gate không đạt. Đây là mục tiêu kiểm thử trên corpus hiện có, không bảo đảm chất
lượng camera thực tế.

`report.json` so sánh model mới với model cha ở cả ngưỡng cũ và ngưỡng được
calibrate lại trên cùng tập calibration. Báo cáo có lỗi từng hành động negative,
nhóm đứng gần, recall/precision, sự kiện và AUC trong từng video so với motion
baseline. Nếu không có video test chứa cả fight/non-fight, AUC trong video để
`null`, không thay bằng AUC gộp.

512 video indoor AIRTLab/SBU hiện được chia theo nhóm nguồn; nhãn lead-in AIRTLab
có AI hỗ trợ và chưa phải human-gold. Checkpoint cha có thể đã thấy một phần public
corpus trước lần chia này. `parent_exposure.json` đối chiếu manifest cha theo hash
nếu tìm được; nếu không tìm được thì ghi `unknown`. Không coi regression public
hoặc trạng thái unknown là bằng chứng tổng quát hóa trên test hoàn toàn mới.

Gói ZIP chứa `interaction.pt` tương thích installer và nhánh **MMAction2 + nhánh
fine-tuned** hiện tại. Chỉ thử cài khi `report.json` hoàn tất và calibration đạt
gate; `approved_for_demo` vẫn false cho tới khi kiểm thử camera riêng. V4 không
tự thay checkpoint của web. Video camera riêng không được dùng để chọn model/ngưỡng.
