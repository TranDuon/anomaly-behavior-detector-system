# Rà nhãn và pilot reviewed — 27/10/2026

## Kết luận thực tế

Vấn đề không chỉ là model hoặc video nhiều. Có **nghi vấn nhãn/mốc thời gian và khác biệt tiêu chí hành vi**, cùng với **ngưỡng cảnh báo chưa phù hợp**. Tăng epochs trước khi làm rõ các điểm này có thể làm model học chắc hơn mục tiêu sai. Chưa có checkpoint nào được duyệt demo iPhone/lớp học.

## Tiêu chí người dùng đã chốt

Báo động tác đấm/đá/vật/xô ngã có hướng vào người khác, kể cả diễn mô phỏng an toàn. Không báo chỉ vì đứng sát, tranh cãi, can ngăn, vung tay một mình hoặc chỉ nhìn thấy người nằm đất sau sự việc. Không cần/khuyến khích va chạm thật để demo.

## Audit 60 cửa sổ ban đầu

- 48 cửa sổ phân tầng ngẫu nhiên từ nhiều nguồn train/validation, thêm 12 cửa sổ tăng cường lỗi và báo riêng; ảnh ẩn nhãn và model score.
- Chốt nhận xét trước khi đọc mapping; SHA-256 judgments: `ba85abf20ff09abdfdaf5a6d94ccdf938ba9cefbbbe45fc009e0490ffe4d638c8`.
- 18 cửa sổ chưa rõ, 42 có nhận xét fight/nonfight. Trong cohort ngẫu nhiên: 15/48 chưa rõ, 11 nhận xét khác GT trong 33 đã phân loại. Trong cohort tăng cường lỗi: 3/12 chưa rõ, 9 khác GT trong 9 đã phân loại.
- **Không diễn giải các số này thành tỷ lệ nhãn sai của toàn dataset.** Rater là AI hỗ trợ xem sáu frame, không phải người gán nhãn độc lập; một số cảnh đã thấy ở audit trước. Khác GT có thể do nhãn/mốc lỗi, tiêu chí active-action khác incident, hoặc rater chưa thấy đủ ngữ cảnh.
- Không đảo nhãn toàn bộ, không xóa clip khó khỏi benchmark, không sửa annotation gốc.

## Loại trừ một số lỗi pipeline và đối chiếu nguồn

Kiểm tra `F_21` ở 18s, `F_175` ở 53s, `F_117` ở 53s, `F_54` ở 11s, `F_162` ở 9s: video CRC khớp archive, annotation byte-for-byte khớp archive; seek và đọc tuần tự trả ảnh giống hệt (mean pixel difference = 0). Vì vậy ở năm trường hợp này chưa thấy lỗi copy nhầm file hoặc seek lệch frame. Không suy rộng kiểm tra này thành chứng minh toàn dataset không có lỗi đọc.

Trên repo của tác giả có [issue #20 về nhãn không chính xác](https://github.com/DegardinBruno/human-self-learning-anomaly/issues/20), mở năm 2022, kèm phản ánh tương tự từ người dùng khác. Đây là phản ánh công khai, **không phải lời xác nhận của tác giả rằng toàn bộ nhãn đều sai**. [Luận án tác giả, mục 5.2.2](https://www.di.ubi.pt/~hugomcp/doc/bruno_degardin_2019.pdf) mô tả event từ lúc tiếp xúc đến khi người gây hấn rời đi; tiêu chí đó có thể giữ cả khoảng tạm dừng trong sự kiện, khác yêu cầu chỉ báo active action của demo.

Đã thêm `scripts/fight_source_health.py` để tái kiểm tra CRC/annotation/seek. Đối chứng 2×2 dùng nhãn gốc đã dừng sau khoảng 374 giây cache, **chưa train epoch nào**. `comparison.json` trên Drive ghi `stopped_before_training`; không có kết quả độ chính xác mới từ lần bị dừng. Workflow expanded có gate chặn train lớn mới khi marker nhãn yêu cầu rà; đối chứng unreviewed chỉ chạy bằng cờ xác nhận nghiên cứu rõ ràng.

## Bản curated riêng, không sửa benchmark

Chọn có chủ đích 8 video nguồn, gồm nguồn có cảnh đông người trong phòng/hành lang và ngoài trời; lấy nhiều đoạn cùng nguồn để có đối chứng bình thường/đánh nhau trong cùng cảnh. Rà 62 cửa sổ, giữ 51, bỏ 11 chưa rõ **chỉ trong phiên bản curated**. Nhận xét được chốt trước khi mở mapping; SHA-256 `d76255728afe353f6db2c1b351de0d1892284de97ed8d44cee0581a0fb3a48fd`.

- Train: 28 clip, 16 fight + 12 nonfight, 4 video; 3 video có cả hai nhãn. 17 nhãn khác annotation cũ trong manifest mới.
- Development validation: 23 clip, 8 fight + 15 nonfight, 4 video nguồn khác; cả 4 có hai nhãn. 12 nhãn khác annotation cũ trong manifest mới.
- Không trộn video/scene group qua split, không dùng official test. Tập đã chọn có chủ đích sau audit, validation dùng chọn checkpoint/threshold; không phải test cuối độc lập.
- Tất cả đều ghi `human_verified=False`, `approved_for_demo=False`. Chưa chứng minh đúng với camera iPhone thực tế hoặc lớp học thật.

## Đối chứng nhỏ, cùng 23 clip reviewed

L4, tổng khoảng 91,19 giây cache/train/evaluate/export, trong budget 8 phút. Clip vẫn 16 frame/8fps; chỉ đối chứng vùng nhìn. Cấu hình thời gian native chưa được train/chấm trên bản curated vì review chỉ phủ cửa sổ 1,875 giây, không tự kéo dài cửa sổ rồi dùng lại nhãn.

| Model | AP | AUC | Precision* | Recall* | FPR* | Median in-video AUC |
|---|---:|---:|---:|---:|---:|---:|
| Expanded cũ, reference | 0,8191 | 0,8750 | 75,0% | 75,0% | 13,3% | 1,000 |
| Train nhỏ, crop tubes | 0,7541 | 0,8167 | 77,8% | 87,5% | 13,3% | 0,875 |
| Train nhỏ, pairs + context | 0,8123 | 0,8750 | 61,5% | 100% | 33,3% | 0,900 |

\* Threshold chọn lại trên chính development nhỏ. Reference cũ không cùng budget train; không dùng bảng để kết luận kiến trúc mới thắng. Các điểm là trên AI-curated cohort rất nhỏ, không phải độ chính xác sản phẩm.

Crop tubes: TP=7, FP=2, FN=1, TN=13. Pairs: TP=8, FP=5, FN=0, TN=10. Crop mới giúp thấy người lớn hơn nhưng max score nhiều vùng có thể làm báo nhầm tăng; **không chứng minh crop mới tốt hơn toàn diện**. Train-fit AUC khoảng 0,995 trên 28 clip chỉ cho thấy fit được tập nhỏ, không phải generalization.

**Không nói AP đã tăng từ 0,426 lên 0,812:** hai số đó khác cohort, khác phiên bản nhãn và khác quy mô train. Chỉ so các hàng trong cùng bảng với các giới hạn đã nêu.

## Ngưỡng là vấn đề riêng, không chỉ nhiễu nhãn

Giữ nguyên model expanded và ngưỡng cũ 0,2747956 trên 23 clip reviewed: TP=8, FP=11, FN=0, TN=4, precision 42,1%, recall 100%, FPR 73,3%. Ranking AUC/AP khá tốt nhưng ngưỡng vận hành vẫn báo nhầm nhiều. Ngưỡng tối ưu trên 23 clip không đủ cơ sở thay cấu hình camera thật. Chưa đổi threshold hoặc model local.

## File và thao tác

- Notebook có pull code: `colab/fight_curated_pilot_drive.ipynb`. Đã có report completed thì không train lại; có thể đọc báo cáo không cần GPU.
- Input archive trên Drive giữ nguyên; nếu cần phiên mới, chỉ giải nén 8 video reviewed, không giải nén toàn bộ dataset.
- Audit: `MyDrive/SENTINEL/artifacts/fight_review_v1` và `fight_curated_review_v1`.
- Pilot/bundles/report: `MyDrive/SENTINEL/artifacts/fight_curated_pilot_v1`.
- Marker chất lượng: `MyDrive/SENTINEL/artifacts/ubi_label_review_gate.json`.
- Đã kiểm tra bundle tubes load/forward CPU và bị chặn mặc định khi chưa bật pilot. Bundle kiểm tra ở `logs/curated-runtime.zip`, không ghi đè `models/`.
- `scripts/colab_process.py` dọn tiến trình con khi timeout/ngắt notebook; tránh job GPU chạy nền sau khi cell bị ngắt. Runtime vẫn được giữ theo yêu cầu người dùng; không còn tiến trình train.

## Hướng tiếp theo có căn cứ

1. Mở rộng **dữ liệu được rà theo tiêu chí active action**, có nhiều đoạn âm/dương cùng nguồn và mẫu âm khó như can ngăn/đùa/vung tay/đông người. Giữ uncertain để rà video/ngữ cảnh, không ép nhãn.
2. Xác minh độc lập các nhãn quan trọng; tách train, calibration và test theo nguồn/cụm cảnh. Không tiếp tục chọn threshold/checkpoint trên 23 clip rồi gọi đó là test chuẩn.
3. Thử vùng tương tác và cách gộp score có kiểm soát; kết quả hiện tại không ủng hộ cứ thêm ROI rồi lấy max là đủ.
4. Chỉ sau khi nhãn/calibration đủ tin cậy mới train lâu hơn và nghiệm thu video/camera thật. Không cần NTU để đi tiếp, nhưng không hứa chất lượng mọi cảnh lớp học/hành lang/ngoài trời chỉ từ bộ tiny này.

Số đầy đủ: `docs/experiments/fight_curated_pilot_v1.json`; nhận xét raw và comparison audit được lưu riêng trong `docs/experiments/`.
