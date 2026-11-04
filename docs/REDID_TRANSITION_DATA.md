# Dữ liệu chuyển xanh sau xô xát trên Colab

Notebook [mmaction_action_transition_v2.ipynb](../colab/mmaction_action_transition_v2.ipynb) tự tải hai ZIP gốc từ [Re-DID của Đại học Trento](https://loki.disi.unitn.it/~ReDID/) vào `/content/action_transition_v2`. Không cần tải dataset về máy cá nhân. Notebook kiểm tra SHA-256 trước khi cắt clip; video gốc và clip tạo ra không được đẩy lên GitHub hay sao chép lên Drive.

`scripts/build_redid_transitions.py` đọc nhãn frame `fight` và `separating`. Nó chỉ tạo nhãn `after_fight` sau khi mọi hành động xô xát/can ngăn đang chồng nhau đã kết thúc, dừng trước hành động được gắn nhãn tiếp theo, bỏ đoạn quá ngắn hoặc mốc mơ hồ. Mỗi clip có một đoạn `fight` dương trước mốc kết thúc và một đoạn `after_fight` âm ngay sau đó. Toàn bộ clip từ cùng video gốc giữ chung một `group` và cùng tập dữ liệu.

Với hai ZIP gốc đã kiểm tra ngày 04/11/2026, quy tắc tạo 43 clip từ 24 video nguồn: train 21 clip, development 13 clip, test 9 clip. Gói review hiện có vẫn thêm video vật thể thao vào tập test. Hai video test chưa gắn mốc thời gian không tham gia huấn luyện. Các khoảng thời gian khác trong Re-DID không tự được coi là âm.

Các clip Re-DID là nhãn tự suy ra từ mốc của tác giả, chưa phải đánh giá thủ công từng cảnh. Chúng dạy mô hình nhận ra thời điểm xô xát kết thúc, nhưng chưa chứng minh trạng thái xanh trong dưới 1 giây hay giải quyết lỗi báo đỏ với vật thể thao. Sau khi Colab fine-tune xong, cần chạy checkpoint pilot trên video xô xát mới, đo thời gian chuyển xanh bằng `scripts/evaluate_action_trace.py`, rồi mới cân nhắc cài vào ứng dụng.
