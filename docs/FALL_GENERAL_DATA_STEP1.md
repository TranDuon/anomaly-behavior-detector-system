# Dữ liệu bước 1: đánh nhau, vật nhau và ngã nói chung

Ngày cập nhật: 01/11/2026. Đầu ra dự kiến của lần tinh chỉnh sau là **một nhãn dương chung** cho đánh nhau, vật nhau và ngã bất kỳ nguyên nhân nào. Ôm, ngồi xuống, nằm xuống chủ động và hoạt động thường ngày là nhãn âm. Nguồn train chỉ gồm dataset công khai; video người dùng giữ cho kiểm thử nghiệm thu. Chưa chạy train hoặc thay checkpoint.

## Dữ liệu hiện có

**Khuyến nghị cho Colab:** [GMDCSA24 v2.1](https://zenodo.org/records/13354453). Gói ZIP khoảng 1,1 GB chứa video ngã và hoạt động thường ngày; bản v2.1 bổ sung nhãn lớp và thời điểm bắt đầu/kết thúc trong CSV. Colab có thể tải trực tiếp từ Zenodo. Không cần chép bộ này vào máy local hay Drive trước.

| Nguồn | Nhãn sẵn có | Vai trò |
| --- | --- | --- |
| `benchmarks/mmaction_interaction_public_v1.json` + `benchmarks/mmaction_interaction_indoor_v2.json` | Sau khử trùng có 544 nguồn AIRTLab/SBU/RLVS, gồm fight và các tương tác âm như hug, handshake, stand-close. Không có nhãn riêng cho vật nhau/ngã. | Giữ tập gốc của model và mẫu âm dễ nhầm; khóa split của 64 nguồn v1. |
| [Fall Detection Dataset (State-to-Fall + ADL)](https://github.com/samruddhi-2308/FallDetectionDataset) | 54 MP4: 48 clip chuyển sang ngã, 6 clip hoạt động thường ngày. Có `metadata/clips_index.csv` gán nhãn cấp clip; chỉ 8 clip có CVAT XML, cách ghép XML với video được tác giả ghi cần kiểm tra lại. CC BY-NC 4.0. | Nguồn phụ. Đã dừng tải local theo yêu cầu; chỉ một phần video ở `artifacts/fall_general_v1/raw/state_to_fall/`, chưa được nhập vào tập train. |
| [IKAR](https://agh.rodbuk.pl/dataset.xhtml?persistentId=doi:10.58032/AGH/A1NZPE) | 278 video ngã, 4 kiểu ngã, hộp người theo từng frame; CC BY 4.0. **Không có mẫu âm**. | Nguồn dự phòng. Lượt tải đã dừng; chưa có gói hoàn chỉnh hoặc MD5 đạt. |
| [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) | 30 chuỗi ngã + 40 hoạt động thường ngày dưới dạng ZIP ảnh RGB theo frame; CC BY-NC-SA 4.0 cho nghiên cứu phi thương mại. | Nguồn dự phòng có âm phù hợp, nhưng khoảng 70 ZIP RGB riêng nên cần chuyển ảnh thành video trước khi train. |

Gói `artifacts/assault_extension_v1/review_pack/` vẫn có 544 preview để xem và gán mốc thời gian. Tên thư mục cũ chỉ là tên kỹ thuật; giao diện và quy tắc đã đổi sang **ngã nói chung**. Preview là dữ liệu rà soát, không thay thế video gốc chất lượng train.

## Tình trạng sử dụng để tinh chỉnh

Nguồn Fall Detection Dataset có nhãn **fall / no-fall cấp clip**, đủ để đưa vào bước chuẩn bị train phân loại clip sau khi tải và kiểm tra xong. Để báo động theo **đúng khoảng thời gian**, vẫn cần kiểm tra mốc bắt đầu/kết thúc ngã; không tự coi cả clip là đang ngã. Cần chia train/validation/test theo người hoặc phiên quay khi có thông tin, và không chuyển nguồn đã dùng để train checkpoint v1 sang test.

**Chưa kết luận tập ba hành vi đã sẵn sàng để train:** 544 preview còn chờ duyệt mốc; chưa có nhãn vật nhau riêng đã xác nhận; các nguồn ngã mới chưa được hợp nhất với corpus v1. `ready_for_training=false` ở gói duyệt vẫn là chỉ báo đúng. Khi đó mới lập manifest gộp và chạy fine-tune `mmaction_interaction_v1`.

## Bước 2: manifest cho Colab

Đã lấy **8 CSV gốc** ở tag `v2.1`, kiểm tra hash Git, và đối chiếu **160 tên video** với inventory của nhà phát hành. Chạy `python scripts/prepare_gmdcsa24_v2_1.py` tạo `benchmarks/gmdcsa24_v2_1_manifest.json`: **78 video ngã có mốc dương**, **81 video ADL âm**, **1 video ngã thiếu mốc** (`Subject 4/Fall/15.mp4`, để chờ duyệt). Không lấy giờ ngồi của video này làm giờ ngã. Chỉ các đoạn ngã được nhà phát hành ghi rõ mới là dương; phần còn lại của video ngã chưa tự xem là âm.

Chạy `python scripts/prepare_general_fall_step2.py` tạo `benchmarks/general_fall_step2_manifest.json` gồm **704 nguồn duy nhất**: 64 nguồn checkpoint v1 giữ split, 480 nguồn tương tác bổ sung còn chờ duyệt và 160 nguồn GMDCSA24. Phân GMDCSA24 theo người: **Subject 1–2 train, Subject 3 development, Subject 4 test**. Như vậy một người không xuất hiện ở nhiều split. Chưa có nguồn ngã dành riêng cho calibration. Các nhãn vật nhau của 480 nguồn cũng chưa được xác nhận.

**Đây là preflight metadata, chưa phải tập train đã kiểm tra video.** Để tiếp tục trên Colab cần tải ZIP [GMDCSA24 v2.1](https://zenodo.org/records/13354453) dung lượng **1.107.545.615 byte** từ Zenodo, kiểm tra MD5 `3d36f2c5c1a666b99639e4e9fd843efb`, giải nén, rồi chạy `python scripts/prepare_gmdcsa24_v2_1.py --video-root <thư_mục_chứa_Subject_1_đến_4>`. Lệnh này kiểm tra đủ 160 MP4 theo kích thước công bố; nó không train hoặc đổi checkpoint. **Chưa chạy lượt tải và kiểm tra dài này.**

Notebook `colab/prepare_general_fall_v1.ipynb` chứa metadata 160 nguồn và cell tải/kiểm tra trực tiếp trên Colab. Cell tải để nguyên chưa chạy; chỉ thực hiện sau khi người dùng đồng ý chạy lâu.
