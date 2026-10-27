# Kết quả pilot tương tác — 27/10/2026

## Đã chạy thực tế

- Training revision: `469445388cf7e0d48536467adc49a2bdfbefca67`; bản render báo cáo: `da6bfc1b26825b72e53e9d49fd0646ce94403d42`.
- GPU NVIDIA L4, tổng cache/train/chấm/xuất hai cấu hình: **574,82 giây** (9 phút 35 giây), budget 20 phút. Bản 8fps hoàn thành 2 probe + 4 fine-tune, bản 12fps 2 probe + 6 fine-tune. Checkpoint chọn bằng macro AP development AIRTLab/RLVS, không chọn theo PKU chẩn đoán.
- Khởi tạo từ `fight_large_v1/best.pt`, SHA256 `bd3d03cf5bacd5d4b9cc2de6329c2f9a1ca91995ec990374575fcffbc3928cc0`.
- [Artifact trên Drive](https://drive.google.com/drive/folders/1v50zzQy3nJyoeEsCm3mPVYtutjW7oEu3): comparison/summary, hai checkpoint, hai gói CPU, manifest, prediction, history, log, timeline, provenance; cache nén 535,2 MiB cũng đã lưu.
- Không cài đè model app, không duyệt demo, không thử iPhone vật lý. Runtime Colab vẫn giữ kết nối theo yêu cầu trước đó.

## Dữ liệu thực sự có

`datasets/fight (1)/pku_mmd_sample` có **3 RGB nguồn**, không phải toàn bộ PKU-MMD. Thư mục nhãn lớn không có nghĩa RGB tương ứng đã tải đủ. `0005-M` train: 2 occurrence dương tạo 6 cửa sổ và 5 occurrence âm tạo 15 cửa sổ. Lấy mẫu lặp để cân bằng không tạo thêm người diễn hay tình huống độc lập.

`0176-L`, `0360-R` cùng phía chẩn đoán: tổng 4 occurrence đấm/đá và 10 occurrence âm có nhãn; push để vùng chưa chấm. Hai nguồn từng được xem, chưa xác minh độc lập diễn viên, không phải test cuối mới. AIRTLab upload thêm là subset của archive đã có, không tính thành dữ liệu mới độc lập. Không dùng test public v1 để train/chọn checkpoint/chọn ngưỡng trong pilot.

## Cửa sổ đã căn theo annotation

Một cửa sổ kết thúc ở cuối mỗi đoạn hành động. Tổng chỉ **14 cửa sổ từ 2 nguồn**; không suy thành độ chính xác camera.

| Cấu hình | TP | TN | FP | FN | Precision | Recall | Accuracy | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline v1 | 0 | 10 | 0 | 4 | 0%* | 0% | 71,43% | 0,7986 |
| Aligned 8fps | 3 | 8 | 2 | 1 | 60% | 75% | 78,57% | 0,8125 |
| Short 12fps | 2 | 10 | 0 | 2 | 100% | 50% | 85,71% | 0,7750 |

*Không có dự đoán dương, quy ước zero-division=0. Precision 100% ở bản 12fps chỉ từ **2 lần dự đoán dương**, không phải bảo đảm chắc chắn. AP bản 12fps thấp hơn baseline: không thể kết luận khả năng phân biệt tổng thể đã tốt hơn chỉ vì accuracy tăng.

## Video nguồn liên tục, cùng runtime/gate

Giữ confirm 0,8 giây, stride 0,5 giây và phép match đã khai báo: một-một, trễ cuối tối đa 1 giây, IoU tối thiểu 0,1 với interval mở rộng. Không nới phép đo sau khi nhìn kết quả.

| Cấu hình | Event đúng / 4 | Event FP | Bỏ sót | Event recall | FP / phút bình thường |
|---|---:|---:|---:|---:|---:|
| Baseline v1 | 0 | 0 | 4 | 0% | 0 |
| Aligned 8fps | 0 | 4 | 4 | 0% | 2,162 |
| Short 12fps | 1 | 0 | 3 | 25% | 0 |

Tổng thời gian được chấm bình thường chỉ 1,85 phút. Event không match vì quá trễ cũng tính FP; không được diễn giải mọi FP là một hành động bình thường độc lập. Ví dụ 8fps trên `0176-L`, kick kết thúc 59,833 giây nhưng alarm bắt đầu 60,9 giây, ngoài biên đã chốt. Trên `0360-R` còn có nhầm ôm/cử chỉ và đỉnh điểm đấm/đá ngắn. Timeline cho thấy phải sửa cả phân biệt hành động lẫn xác nhận theo thời gian; giảm confirm đơn thuần có thể bắt thêm cả ôm.

## Calibration không phải test

Ngưỡng frozen: 8fps `0.2075291872`; 12fps `0.9371675253`.

| Cấu hình | AIRTLab P/R (24 clip calibration) | RLVS P/R (40 clip calibration) |
|---|---|---|
| 8fps | 81,25% / 92,86% | 100% / 95% |
| 12fps | 85,71% / 85,71% | 100% / 95% |

Đây là các mẫu dùng chọn ngưỡng, không phải bằng chứng nghiệm thu ≥80%. Không có calibration PKU độc lập, cũng chưa có test crowd/handheld/iPhone mới.

## Kiểm chứng CPU

Hai gói CPU được kiểm tra tại local trên 6 tensor RGB đã preprocess (ôm/đá/đấm của `0360-R` ở hai cấu hình), không thay model app.

- So với điểm GPU ban đầu: sai số lớn nhất 0,001030, **không đạt** tolerance nghiêm 0,0001. Không bỏ/ghi đè báo cáo thất bại đó.
- Đối chứng tắt TF32: Colab CPU so GPU FP32 nghiêm sai số tối đa 0,00000143; Windows CPU so Colab CPU tối đa 0,000000715. Cả 6 quyết định tại ngưỡng frozen giống nhau.
- Đây là kiểm tra tính di động số học của classifier, **không** bao gồm YOLO, hàng đợi upload/replay, tốc độ end-to-end hay camera thật. Không phải nguyên nhân chính của các bỏ sót hành động hiện tại.
- Code cho **lần chạy mới** đã tắt TF32, đặt torch seed và chọn ngưỡng giữa các nhóm điểm calibration để tránh đặt đúng trên một điểm GPU. Kết quả pilot ở trên vẫn dùng revision/ngưỡng gốc, không tính lại để báo đẹp hơn. Run gốc chỉ đặt seed sampler Python, không đặt torch seed.

## Kết luận và hướng tiếp theo

**Chưa đạt yêu cầu demo.** Đã có tín hiệu cải thiện cửa sổ/action score nhưng chưa đủ ổn định thành alarm đúng lúc; không tiếp tục train nhiều giờ trên một nguồn PKU rồi hứa ≥80%.

Ưu tiên tiếp theo là thêm RGB tương tác **mới** với người diễn/nguồn khác, gồm đấm/đá/vật/xô ngã và âm khó: ôm, high-five, can ngăn, vung tay một mình, di chuyển nhóm. Giữ cùng người diễn/session/các góc quay cùng phía split, dành riêng development và calibration liên tục để thiết kế gate bắt động tác ngắn mà không tăng báo giả. Một mốc thu thập thực dụng là 20–30 bản ghi mới với nhiều cặp/nhóm người, không phải số bảo đảm đủ hay bảo đảm accuracy. Sau đó mới thử auxiliary action head/đa nhịp thời gian trên cùng đối chứng; không trộn tất cả hành động không liên quan để tăng số lượng.

Nguồn RGB PKU Phase I chính thức hiện cần quyền truy cập; Phase II công khai đã kiểm không có lớp đấm/đá vào người. Cần RGB Phase I thực sự hoặc nguồn tương đương, không chỉ archive nhãn. Khi có candidate qua đánh giá liên tục mới kiểm tra CPU upload/replay và iPhone thật; hiện chưa nối iPhone nên chưa thể xác nhận hai luồng demo.
