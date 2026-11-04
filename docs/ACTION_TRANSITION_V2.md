# MMAction: ngã sớm và kết thúc cảnh báo

## Quyết định của lần thử này

- Giữ checkpoint `mmaction_fall_wrestle` hiện tại làm mốc so sánh. Colab xuất một ZIP pilot riêng; không tự thay trọng số trong ứng dụng.
- Mục tiêu chuyển xanh là dưới 1 giây sau khi hệ thống quan sát rõ xô xát đã dừng. Cổng cảnh báo của pilot kết thúc sau hai dự đoán thấp liên tiếp (khoảng 0,53 giây giữa hai dự đoán), nhưng không thể chuyển xanh khi chính model vẫn cho điểm cao.
- Vật thể thao là mẫu âm. Chỉ xô xát/bạo lực là mẫu dương cho đánh/vật. Ngã được gắn nhãn dương riêng theo thời điểm bắt đầu mất thăng bằng; head hiện tại vẫn xuất một điểm chung.

## 1. Ghi trace và đo

Mỗi phiên web chạy `mmaction_fall_wrestle` mới tạo `logs/action_traces/<session_id>.jsonl`. Header chứa nguồn và cấu hình cổng; mỗi dòng tiếp theo chứa mốc video, điểm, trạng thái, số người, vùng người, thời gian suy luận và số mẫu rơi. `/api/status` trả về `prediction_trace` của phiên hiện tại. Trace không chứa frame ảnh.

Sau khi điền nhãn review, chạy `scripts/evaluate_action_trace.py <trace.jsonl> <review.json>` để đo độ trễ báo đỏ từ `fall_onset`/`fight`/`wrestle`, độ trễ trả xanh từ `after_fight`, và tỷ lệ đỏ sai trong `sports_wrestle`/`ordinary`. Mốc chưa gắn nhãn không tham gia đánh giá. Cần đo thêm cả false alarm trên video mới và độ trễ ngoài đời trước khi nhận phiên bản mới.

## 2. Chuẩn bị nhãn video

Chạy `mmaction2_demo/env/python.exe scripts/prepare_action_review.py --output artifacts/action_review_v1 <video1> <video2> ...`. Script bỏ video trùng theo SHA-256, tạo ảnh tiếp xúc có mốc giây, bản sao video và `review.json`. Các file trong `artifacts/action_review_v1` được Git bỏ qua và không được đẩy cùng source.

Điền `intervals` theo thứ tự thời gian, không chồng nhau. Mỗi đoạn cần `start_seconds`, `end_seconds`, `label`, `action`:

| `action` | `label` | Ý nghĩa |
| --- | ---: | --- |
| `fall_onset`, `fall` | 1 | Bắt đầu mất thăng bằng hoặc đang ngã |
| `fight`, `wrestle` | 1 | Xô xát/bạo lực đang diễn ra |
| `after_fight` | 0 | Hai người đã ngừng tấn công, dù vẫn đứng gần |
| `ordinary`, `sports_wrestle` | 0 | Sinh hoạt bình thường hoặc vật thể thao |

Không gắn nhãn phần thời gian chưa chắc chắn. `group` phải là cùng một người/cảnh hoặc cùng video; không để nhóm trùng ở train và test. Mặc định script đặt `split: test` để giữ video test làm phép đo độc lập. Muốn thêm dữ liệu train, dùng video khác đã duyệt và đặt `split: train`; dùng các cảnh khác cho `development`.

Đã chuẩn bị gói cục bộ từ video test ngày 04/11/2026: 5 lần tải lên gồm 3 video duy nhất. Clip 88,6 giây trên thảm thi đấu được gắn `sports_wrestle` âm theo quyết định của người dùng và giữ ở test. Hai clip còn lại để trống mốc chờ duyệt. Không dùng video test này để tối ưu ngưỡng rồi gọi đó là kết quả test độc lập.

## 3. Fine-tune trên Colab

Notebook `colab/mmaction_action_transition_v2.ipynb` chạy Colab GPU và lưu cache/epoch trên Drive. Nó xác minh ZIP checkpoint cha bằng SHA-256, tải GMDCSA24 v2.1 và SCFD, lấy nhiều cửa sổ gần đầu ngã hơn, rồi fine-tune head từ checkpoint hiện tại. Nếu có gói review đã duyệt, nén **nội dung** `artifacts/action_review_v1` thành `action_review_v1.zip` sao cho `review.json` ở gốc ZIP và đưa tới `MyDrive/SENTINEL/artifacts/action_review_v1/action_review_v1.zip`. Khi đổi nhãn hoặc dữ liệu, đổi `RUN_TAG` trong notebook để tạo cache/protocol mới.

Nếu chưa có gói review, notebook vẫn chạy thử cải thiện ngã bằng dữ liệu công khai. Nó **không thể chứng minh** đã học được thời điểm đánh nhau kết thúc. SCFD chỉ có nhãn fight toàn clip, không có mốc ngừng đánh hoặc nhãn vật bạo lực riêng. Báo cáo gồm so sánh checkpoint cha/mới trên test có người và tỷ lệ báo đỏ trên các loại video review test. Chỉ cài ZIP mới sau khi kiểm tra độ trễ thời gian và báo nhầm trên video chưa dùng để train.
