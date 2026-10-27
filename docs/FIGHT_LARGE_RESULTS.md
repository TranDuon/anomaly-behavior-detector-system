# Kết quả train lớn — 27/10/2026

## Đã thực hiện

Training revision `64fffb5`, manifest SHA256 `055695b71f0bcffc94e2b2ccfe64d93f432b5f91fd95bda5cec4f713248d693d`. Dữ liệu RLVS + AIRTLab đã lọc vũ khí, không dùng NTU hoặc frame GT UBI. Archive, log, checkpoint và báo cáo ở [Drive fight_large_v1](https://drive.google.com/drive/folders/1vuCG1gnTp-GUm2aRees_3nNpyoV7bA8Z).

- 1.577 cửa sổ train trước lọc; thực train **1.550 video/cửa sổ**, gồm 760 dương, 790 âm. Giữ 31 âm không thấy người, loại 27 âm nền khỏi train; giữ dương dù detector thiếu người.
- Development 216, calibration 228, test 202 (99 dương, 103 âm), tách nhóm trước khi tạo cửa sổ. Hai camera AIRTLab cùng màn diễn luôn chung split. Heuristic không chứng minh sạch mọi trùng nguồn/diễn viên; AIRTLab chung phòng/diễn viên.
- Cách ly trước training: 1 clip ngắn hơn cửa sổ model, 2 clip trùng SHA256 nhưng nhãn xung đột. Không xóa/sửa archive nguồn hoặc lọc theo score test.
- 7 epoch probe + 7 fine-tune đã hoàn tất, early stopping; cache/train/evaluate/export khoảng **19 phút 44 giây** trên L4, không vượt budget 45 phút. Dữ liệu tải/chuẩn bị tính riêng.
- Checkpoint chọn trên development, ngưỡng **0,2713758647** đóng băng bằng calibration trước test. Không chỉnh ngưỡng theo test sau khi thấy điểm AIRTLab.
- Nhãn clip publisher, không phải active-action temporal gold. Audit 40 mẫu AI-assisted còn 9 chưa rõ và 6 nhận xét khác nhãn clip trong 31 đã phân loại; không tự đổi nhãn và không suy ra tỷ lệ nhãn sai toàn tập.

## Test đóng băng, theo clip trung tâm

| Tập | Clips | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Gộp | 202 | 94,55% | 91,51% | 97,98% | 94,63% | 8,74% |
| RLVS | 182 | 95,60% | 93,81% | 97,85% | 95,79% | 6,74% |
| AIRTLab | 20 | 85,00% | **66,67%** | 100% | 80,00% | **21,43%** |

Gộp: TP=97, TN=94, FP=9, FN=2; AP=0,98485, AUC=0,98755. Khoảng Wilson 95% mô tả theo clip: precision [84,65%;95,47%], recall [92,93%;99,44%]; chưa hiệu chỉnh sự phụ thuộc giữa nhóm/camera. AIRTLab chỉ có 6 dương và 14 âm, có 3 báo nhầm nên độ bất định lớn. Mục tiêu P/R ≥80% đạt ở tập gộp nhưng **chưa đạt precision trên nhóm hành vi dễ nhầm**, không được kết luận mọi môi trường demo đã đạt.

Không so trực tiếp 94,55% với kết quả UBI hoặc 23 clip curated trước đây vì khác dữ liệu, nhãn và đơn vị đánh giá. Không gọi điểm clip là độ chính xác event/frame/camera iPhone.

## Kiểm thử checkpoint thật trên CPU

Bundle tải về local, nạp TorchScript CPU và forward thành công; mặc định vẫn chặn nếu chưa chủ động cho phép pilot. Không thay checkpoint trong `models/` một cách âm thầm.

Chọn 8 video test theo thứ tự SHA256, mỗi dataset/lớp 2 video, nhóm khác nhau, dài 3,5–12 giây; không chọn theo score hay lỗi. Chạy toàn bộ video qua API upload thường và phát lại đúng nhịp qua reader/buffer camera; đây là **kiểm tra tích hợp/độ trễ**, không phải bộ đo event gold. Model, ngưỡng và nhãn không đổi giữa hai lượt.

- Trước tối ưu: median thời gian một lượt suy luận camera khoảng 0,674 giây; sau warmup và cache detection trên đúng frame/crop khoảng 0,573 giây. Đây là phép đo trên máy này, không cam kết mọi CPU.
- Sau tối ưu, 8/8 video có cùng số sự kiện giữa upload và replay; những timestamp dự đoán trùng nhau cho score giống nhau (max difference=0). Không có nghĩa tất cả timestamp đều trùng: CPU chậm vẫn có thể bỏ nhịp dự đoán.
- Cả hai luồng nhận ra 4 video dương theo nhãn clip; vẫn có **một video âm AIRTLab báo nhầm ở cả hai luồng**. Không biến 8 video đã dùng kiểm tra thành bằng chứng “đạt demo”.
- Giải mã 1080p/replay vẫn có lúc trễ tối đa khoảng **2,22 giây**; warmup/cache giảm thời gian inference nhưng chưa giảm mọi bottleneck decoder/capture. UI đánh dấu kết quả đã cũ thay vì giữ màu đỏ như bằng chứng mới. Bộ đệm chưa mất mẫu trong lượt đo này, nhưng có bỏ cơ hội suy luận.
- Gate không xác nhận từ hai điểm cao cách xa nhau; clip bằng chứng bám timestamp nguồn thay vì tua nhanh khi CPU bỏ nhịp. Cache chỉ tái dùng detection cùng array owner/crop, được xóa khi reset/ra khỏi cửa sổ.
- Khi không thấy người trong cả ba mốc của cửa sổ, runtime chuyển sang thiếu thông tin và kết thúc cảnh báo đang hoạt động ở nhịp suy luận tiếp theo, thay vì giữ đỏ chỉ vì score cao. Không bắt buộc phải thấy đủ hai người; không nhìn thấy người không đồng nghĩa cảnh đã an toàn.

`logs/large-cpu/cpu_acceptance.json` giữ lượt đầu; `logs/large-cpu-optimized/cpu_acceptance.json` giữ lượt sau. Script tái kiểm: `python -m scripts.fight_cpu_acceptance --bundle ... --sources ... --output ... --budget-minutes 10`.

## Trạng thái và bước tiếp

`approved_for_demo=false`, `real_phone_tested=false`. iPhone chưa nối nên chưa đo rung tay, che khuất, ánh sáng, độ trễ mạng hoặc lớp học đông thực tế. Không tự train dài thêm chỉ vì tổng accuracy cao. Ưu tiên rà đủ ngữ cảnh các false positive (ôm/vỗ tay/can ngăn/đứng sát), rồi bổ sung hard negatives/active-action annotations theo nhóm nguồn và giữ một bộ nghiệm thu mới. Nếu chỉnh threshold sau khi đọc test này, đó là một thí nghiệm phát triển mới, cần holdout khác để xác nhận.
