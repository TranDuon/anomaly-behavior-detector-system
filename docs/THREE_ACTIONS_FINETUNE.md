# Fine-tune ba hành vi riêng

Notebook: `colab/mmaction_three_actions_drive.ipynb`.
Trainer: `mmaction2_demo/train_three_actions.py`.

## Nhãn

- `strike`: đấm/đá/tát hoặc đánh trực tiếp vào người khác.
- `grapple`: ghì, kẹp cổ, kéo/vật với tương tác khống chế rõ ràng; không tự coi ôm là vật.
- `fall`: chuyển từ đứng/ngồi sang ngã. Nằm yên sau ngã không tự động là đang ngã.
- Ba đầu sigmoid độc lập; có thể đồng thời `grapple=1, fall=1`.
- Mỗi cửa sổ ghi rõ 0 (không xảy ra), 1 (thấy xảy ra), hoặc null (không xác định).
  Không dùng nhãn nhị phân `fight` cũ làm nhãn đánh, hoặc tự gán mọi clip choke là vật từ đầu đến cuối.

## Dữ liệu

Manifest: `MyDrive/SENTINEL/datasets/three_actions_v1/reviewed_20261110_v1.json`.
Đường dẫn `video` tính từ `MyDrive/SENTINEL/datasets`.
Mỗi nguồn cần SHA-256 video gốc, URL nguồn public, người duyệt, group và split.
Mỗi window có `anchor_seconds` là tâm đoạn 9 mẫu cách nhau 8/30 giây, cùng ba targets.
Các góc quay cùng một lần diễn phải có cùng group; không chia cửa sổ cùng video sang split khác.
Tối thiểu pilot: mỗi nhãn cần 3 group dương/âm train, 2 group dương/âm development và test.
Đây chỉ là chốt chặn thiếu lớp, không phải số lượng đủ để nghiệm thu.
AIRTLab 37/87 và bốn file public đã demo được giữ test-only. Video iPhone không được tải lên.

Phiên bản 20261110_v1 gồm 37 video public AIRTLab và GMDCSA24, 141 cửa sổ.
Nhãn được AI hỗ trợ duyệt bằng contact sheet cách nhau 0,5 giây, không xem điểm model;
chưa phải bộ nhãn gold do người đánh giá độc lập xác nhận. Các ca không rõ để null.
GMDCSA24 chia theo người; AIRTLab chia theo lần diễn, không khẳng định tách người diễn.
Recipe nhãn: `benchmarks/three_actions_visual_review_v1.json`.
Cell dữ liệu xác minh SHA-256 từng video và không tải thêm video về máy local.

## Training

1. Chạy mount + label audit trên CPU. Thiếu nhãn sẽ dừng trước khi tải model GPU.
2. Khi đủ nhãn và file SHA đúng, chọn GPU rồi chạy phần setup và training.
3. Ba `InteractionHead` warm-start từ parent `ddb09180...`; backbone AVA/detector đóng băng.
   Loss BCE bỏ qua targets null. Dừng sớm dựa macro AP trên development.
4. Chọn ngưỡng riêng từng hành vi trên development; chỉ dùng test để báo cáo.
5. Xuất `three_actions.pt`, `three_actions.json`, report, protocol và ZIP riêng.

Không ghi đè checkpoint cũ. Bộ nạp riêng `mmaction2_demo/install_three_actions.py`
xác minh checksum, tensor-only state, SPEC và provenance trước khi cài bằng môi trường MMAction2.
Web có backend `mmaction_three_actions`, ba bộ xác nhận thời gian độc lập và nhãn sự kiện riêng;
không thể nạp checkpoint ba đầu qua bộ nạp checkpoint nhị phân cũ.
Chỉ số báo cáo là cấp cửa sổ,
không thay thế kiểm thử event, độ trễ và false alarm/phút qua upload/camera.
Chưa có bộ nhãn ba hành vi được duyệt thì chưa được tuyên bố đã fine-tune dữ liệu thật.

Tham khảo cơ chế multilabel AVA gốc:
https://github.com/open-mmlab/mmaction2/blob/v1.2.0/mmaction/models/roi_heads/bbox_heads/bbox_head.py
