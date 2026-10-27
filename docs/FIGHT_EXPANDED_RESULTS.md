# Kết quả pilot mở rộng — 27/10/2026

## Đã thực hiện

- Không dùng NTU. Chọn 200 video nguồn UBI official train: 160 train, 40 development validation, không trộn validation nguồn cũ vào train. Kiểm tra nhóm scene hash không phát sinh nhóm xuyên split phải cách ly trong lần chọn này; đây vẫn là phát hiện gần trùng xấp xỉ.
- Tạo 3.052 cửa sổ không chồng: 2.484 train và 568 validation. Sau chọn mẫu train còn 2.157: 756 đánh nhau, 1.294 bình thường có người, 107 nền không người. Không loại positive chỉ vì detector bỏ sót người.
- Trim viền đen, ghép bbox qua ba frame, cùng hàm chuẩn bị input ở train/runtime; fine-tune có mô phỏng rung máy nhẹ. Không coi đó là dữ liệu iPhone thật hoặc bối cảnh lớp học đã được xác minh.
- L4: 4 epoch probe (early stop), 6 epoch fine-tune. Best AP ở fine-tune epoch 3. Cache/train/evaluation/export mất 1.585 giây; cell budget gồm setup/prepare tới export báo 30,28 phút. Không hết budget 45 phút, không tự train thêm.
- Training code `d851f5284a9c97a9e6dfe09bb00f5b829e67a0d1`; diagnostic code `bc8bf1e2bc76acccc80547bbba9a833cd2b0f31a`.

## Cùng 568 clip validation, không so lệch cohort

| Chỉ số | Baseline v2 interaction | Expanded best |
|---|---:|---:|
| AP | 0,4324 | 0,4259 |
| ROC-AUC | 0,6453 | 0,6963 |
| Precision tại ngưỡng chọn trên dev | 41,13% | 39,54% |
| Recall tại ngưỡng chọn trên dev | 70,30% | 83,64% |
| False-positive rate | 41,19% | 52,36% |
| Median in-video AUC, 19 video mixed | 0,5208 | 0,5119 |
| Recall khi precision đạt ít nhất 75% trên dev | 0,61% | 0% |

**Kết luận: không đạt, không thay model đang dùng.** AUC tổng tăng nhưng AP không tăng, báo nhầm cao hơn và khả năng phân biệt đoạn bình thường/đánh nhau trong cùng cảnh vẫn gần ngẫu nhiên. Chỉ tăng lượng data và fine-tune lâu hơn theo cấu hình này không đủ đáp ứng demo. Đây là selected clips development evaluation, chưa phải full-video frame/event evaluation hoặc nghiệm thu iPhone.

## Đối chứng không train: bỏ/xáo thông tin thời gian

Giữ nguyên checkpoint, 568 clip và threshold; chỉ thay thứ tự/nội dung thời gian ở input:

| Input | AP | AUC | In-video AUC | Đồng ý dự đoán với gốc |
|---|---:|---:|---:|---:|
| Gốc | 0,4259 | 0,6963 | 0,5119 | — |
| Lặp frame giữa | 0,3906 | 0,6185 | 0,4167 | 75,70% |
| Xáo frame | 0,4350 | 0,6893 | 0,4306 | 82,75% |

Đối chứng mất 42,91 giây. Model có thay đổi khi bỏ chuyển động, nên không thể nói hoàn toàn không dùng thời gian. Tuy nhiên xáo thứ tự vẫn giữ AP tương đương/tốt hơn và score correlation 0,873: gợi ý nhận dạng diễn tiến hành động chưa vững. Can thiệp là ngoài phân phối, không đủ tự nó chứng minh quan hệ nhân quả.

## Audit nguồn/crop và nhãn

Đã xuất tối đa 6 false positives và 6 false negatives, giữ cả source lẫn crop. Kiểm tra chuỗi frame của hai trường hợp cho thấy:

- `F_21_0_0_0_0`, 18–19,875 giây: score 0,9642, nhãn toàn cửa sổ là 0; hình cho thấy nhóm người chen lấn/động tác ra đòn trong không gian hẹp. Cần người rà định nghĩa nhãn và mốc thời gian.
- `F_175_0_1_0_0`, 53–54,875 giây: score 0,0391, nhãn toàn cửa sổ là 1; chuỗi hình cho thấy người đang di chuyển ra cửa, không thấy rõ động tác đánh trong cửa sổ ngắn này. Cần rà ngữ cảnh dài hơn trước khi kết luận sai nhãn.

Cả hai video là 30fps, số frame annotation khớp số frame video (2.880 và 4.320). Không thấy lỗi chiều dài annotation ở hai ví dụ, nhưng điều đó không đảm bảo nhãn ngữ nghĩa chính xác từng cửa sổ. Chưa sửa bất kỳ ground truth nào; không relabel validation theo model, không xóa clip lỗi để làm đẹp báo cáo.

## Runtime và artifact

Drive: `MyDrive/SENTINEL/artifacts/fight_expand_v1/` giữ manifest, feature table nén, best checkpoint, history, report/predictions, clip lỗi, temporal controls và semantic candidates. RGB cache/video subset chỉ ở `/content`, mất khi xóa runtime và cần tái tạo nếu fine-tune tiếp.

Đã tải ZIP về `logs/expanded-runtime.zip`, load TorchScript/forward CPU thành công; mặc định vẫn chặn checkpoint chưa duyệt. Đo synthetic input classifier khoảng 0,327 giây/region trên máy local, không gồm YOLO/decode/API, không phải FPS camera end-to-end. Không ghi đè model trong thư mục `models/`.

## Bước hợp lý tiếp theo

1. Rà mốc hành vi của các video lỗi ở cả train và validation bằng người, viết quy tắc nhãn cụ thể cho đánh/đá/xô đẩy so với can ngăn, đùa nghịch, chen lấn; giữ nguyên benchmark cũ và tạo phiên bản reviewed riêng có provenance.
2. Kiểm tra riêng các cảnh đông người: crop toàn nhóm vẫn có thể làm người tương tác quá nhỏ; thử ROI cặp/nhóm nhỏ có giữ ngữ cảnh, không bỏ mẫu dương do detector thiếu người.
3. Chỉ sau audit mới chạy pilot đối chứng biểu diễn thời gian/độ phân giải hoặc unfreeze phù hợp, cùng split và budget; không tự chạy nhiều giờ hoặc nâng ngưỡng để giấu báo nhầm.
4. Trước khi công bố đạt phải kiểm thử video/camera cầm tay thật. Không cần NTU để tiếp tục, nhưng cũng không thể dùng kết quả UBI này để hứa mọi cảnh lớp học/hành lang/ngoài trời đều đúng.

Số đầy đủ: `docs/experiments/fight_expand_v1.json`.
