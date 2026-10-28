# Kết quả phục hồi refinement ngày 28/10/2026

## Trạng thái thực tế

Training đã chạy bằng Tesla T4 tại revision `98eabdfb6dede8288107da98d10338a8ab8913f4`, seed 941. Không phải hết token làm mất checkpoint. Tiến trình tiếp tục độc lập rồi dừng vì `Reviewed-event trace budget exhausted` sau 748,83 giây runner, khi đang đánh giá sự kiện development của cấu hình thứ hai. [Thư mục artifact gốc trên Drive](https://drive.google.com/drive/folders/1MhLynjVDerU81GVel-zew_UA-3RcRWE4) được giữ nguyên; không ghi đè run lỗi.

- `rgb_control`: hoàn tất 3 epoch probe và 2 epoch fine-tune; best là fine-tune epoch 1. Có `best.pt`, history, calibration, timeline development và protocol đã đóng băng.
- `rgb_interaction`: hoàn tất 3 epoch probe và 1 epoch fine-tune; có `best.pt` và history. Chưa có protocol sự kiện hoàn chỉnh.
- Chưa chọn xong cấu hình bằng development, chưa chấm 15 cửa sổ test, chưa xuất runtime/QA bundle của lượt này, chưa kiểm chứng iPhone thật.
- Không thay model local: SHA TorchScript đang cài vẫn là `7cd20e23e1590689a1d08bf63a9aa967c831a71dd985fed1a09fae5bd32b7d09`.

## Đã kiểm tra thêm tại local

Đã tải hai checkpoint và cache public về `logs/refinement-resume-v2`, kiểm ZIP, khôi phục state dictionary bằng `weights_only=True`, kiểm strict shape, mode và tính hữu hạn. Chấm lại từ RGB đầy đủ trên CPU, không dùng feature đóng băng thay cho backbone đã fine-tune. Không chạy optimizer, không mở camera, không đọc hai video iPhone và không chấm test. Bước audit 31 cửa sổ development/calibration mất khoảng 45,76 giây; đây không phải thời gian đo pipeline realtime.

| Chỉ số diagnostic | Baseline hiện tại | RGB control | RGB + interaction |
| --- | ---: | ---: | ---: |
| Development BCE trên 15 cửa sổ | 0,343725 | 0,000615 | 0,002291 |
| Development precision/recall từng dataset | 100%/100% | 100%/100% | 100%/100% |
| Calibration AIRTLab, 10 cửa sổ: accuracy | 90% | 90% | 90% |
| Calibration AIRTLab: precision/recall | 80%/100% | 80%/100% | 80%/100% |
| Calibration RLVS, 6 cửa sổ: precision/recall | 100%/100% | 100%/100% | 100%/100% |
| Cửa sổ ôm người báo nhầm / tổng calibration | 1/3 | 1/3 | 1/3 |
| Ngưỡng calibration diagnostic CPU | 0,830048 | 0,072063 | 0,235329 |

Ngưỡng khác nhau vì thang điểm của classifier thay đổi; không so sánh ngưỡng trực tiếp hoặc chép ngưỡng của model này sang model khác. Ngưỡng RGB control CPU khác protocol GPU đã lưu khoảng `2,24e-8`, không chỉnh protocol cũ. Ngưỡng interaction trong audit chỉ là diagnostic cửa sổ, chưa phải protocol triển khai.

Giảm loss chưa chứng minh tăng khả năng phân biệt: development đã đạt precision/recall/AP 100% trước training, và calibration vẫn giữ cùng các quyết định sai. Nhánh tương tác chưa chứng minh tốt hơn control. Các chỉ số này chỉ trên public pilot nhỏ, nhãn `ai_assisted`, không phải nhãn human-gold hay độ chính xác camera 80–100%.

SHA checkpoint đã tải và strict-load thành công:

- RGB control: `e1e04e1ebfbd14ec0da65ef7a80bd044b6f99332eb9518d303661dec11a89951`.
- RGB interaction: `bcd25626aa427efee9f82e9e8daae90826ae901189f4bcc393ccac48a38df900`.

## Sự kiện liên tục: kết quả đã có

Timeline dưới đây là development gốc của RGB control, không phải chấm lại từ preview nén và không phải test độc lập.

| Dataset | Đúng / sự kiện đánh nhau đã duyệt | Báo nhầm | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| AIRTLab | 3/3 | 0 | 100% | 100% |
| RLVS | 3/3 | 1 | 75% | 100% |

Ngưỡng `0,07206281647086143`, xác minh `0,8` giây, đều chọn từ calibration/development. Baseline với ngưỡng/gate đang cài chỉ bắt 2/3 sự kiện AIRTLab và có một cảnh báo lặp. Tuy nhiên baseline được hiệu chỉnh công bằng cũng bắt 3/3 nhưng có một báo nhầm AIRTLab; không quy toàn bộ chênh lệch cho training.

Báo nhầm RLVS là `rlvs/NV_321`: đoạn vung tay ném bowling một mình, cảnh báo 2,36–3,72 giây. Đây là bằng chứng cụ thể rằng model vẫn có thể nhầm chuyển động mạnh với đánh nhau. Tỷ lệ báo nhầm RLVS khoảng 4,08/phút chỉ tính trên 14,72 giây normal đã duyệt; mẫu quá ngắn để suy ra tần suất camera thực tế. Chưa có kết quả sự kiện tương ứng của interaction, không điền số thay thế từ accuracy cửa sổ.

## Camera CPU là một vấn đề riêng

Kiểm thử API baseline trên 5 video public bằng cả upload và paced replay đã hoàn tất. Cùng timestamp thì điểm khớp, nhưng một clip replay có điểm cũ tới khoảng 2,05 giây; freshness không đạt. Thử `torch.jit.freeze` chỉ trong RAM giảm p95 microbenchmark classifier từ khoảng 0,602 xuống 0,430 giây, nhưng kiểm end-to-end vẫn không qua freshness/parity. Không áp dụng tối ưu đó vào runtime và không quảng cáo là đã sửa realtime. Chưa đo pipeline end-to-end với hai checkpoint refinement vì chúng chưa có protocol/bundle được đánh giá đầy đủ.

## Bước tiếp theo theo thứ tự

1. **Hoàn tất đánh giá, không train lại:** tải `best.pt`, annotations và cache đúng fingerprint; chạy phần sự kiện development còn thiếu của interaction. Lưu từng nguồn video ngay khi xong, kèm SHA checkpoint/nguồn và cấu hình; dừng có thể tiếp tục từ nguồn chưa xong. Tách ngân sách evaluation khỏi optimizer, không chạy lại toàn bộ notebook training hoặc ghi đè `fight_refinement_v2`.
2. **Đóng băng lựa chọn trước test:** giữ nguyên protocol control đã có; so sánh hai cấu hình trên development về precision/recall, thời gian báo đỏ nhầm và trễ bắt đầu. Sau khi chốt mới chấm 15 test windows và timeline test. Không lấy video iPhone hoặc test để hạ ngưỡng/chọn epoch.
3. **Kiểm runtime:** xuất TorchScript/runtime và ví dụ tensor QA, đối chiếu CPU/GPU, rồi kiểm upload và paced replay qua cùng API. Nếu inference cộng detector vượt nhịp 0,5 giây, đo từng thành phần rồi chọn tối ưu/model nhẹ có kiểm sai số; không chỉ tăng ngưỡng stale để che trễ.
4. **Chỉ train dài nếu dữ liệu và development có tiến triển:** thêm ví dụ độc lập trong train về ôm, bắt tay, high-five, bowling/vung tay, can ngăn, tương tác đông người và đánh nhau cận cảnh; không chuyển `NV_321` development sang train. Hiện train chỉ có 11 nhóm fight, 3 hug, 2 greeting, 1 solo-motion và không có high-five. Lấy thêm các cửa sổ khó theo thời gian trong nguồn train đã duyệt thay vì chỉ nhân số epoch trên cùng 74 cửa sổ.
5. **Nghiệm thu demo độc lập:** bộ video không tham gia chọn model, đủ cả normal dễ nhầm và fight; thử iPhone ở lớp, hành lang, ngoài trời. Các video iPhone cũ chỉ là regression local, không tải lên hoặc biến thành test độc lập. Mô phỏng an toàn, không cần đánh thật gây thương tích.

Công cụ `python -m scripts.fight_refinement_audit --help` chỉ phục hồi/chấm cửa sổ non-test và kiểm checkpoint. Nó không được gọi là hoàn tất evaluation, không xuất model để thay tự động, và không khởi tạo optimizer.
