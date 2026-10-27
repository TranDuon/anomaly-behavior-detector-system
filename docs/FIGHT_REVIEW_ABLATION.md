# Rà nhãn và đối chứng crop/thời gian

Trạng thái lần chạy: đối chứng nhãn gốc đã dừng trong cache, chưa train epoch nào do audit phát hiện nhiều nghi vấn. Script hiện yêu cầu cờ `--acknowledge-unreviewed-labels` nếu cố ý chạy nghiên cứu trên nhãn cũ; mặc định không chạy. Thay thế bằng notebook curated và xem `docs/FIGHT_CURATED_RESULTS.md`. Không có kết quả thắng/thua của phép 2×2 chưa hoàn tất.

Không tiếp tục train lớn hoặc tự thay model local. Không dùng NTU.

## Audit ẩn nhãn

`scripts/fight_review.py` lấy 48 cửa sổ phân tầng ngẫu nhiên: 12 clip mỗi tổ hợp train/validation × fight/nonfight, mỗi video nguồn tối đa một clip trong cohort này. Thêm 12 cửa sổ validation có sai số score lớn để tìm vấn đề, báo riêng với cohort ngẫu nhiên. Sau đó xáo thứ tự và chỉ hiện ID R001–R060 cùng sáu frame từ cửa sổ 1,875 giây. ZIP không chứa nhãn hoặc score; mapping riêng được giữ để đối chiếu sau khi chốt judgments.

Đánh giá theo ba mức fight/nonfight/uncertain, ghi độ rõ, che khuất, bối cảnh và số người ước lượng. Đây là rà trực quan hỗ trợ bởi AI, không phải nhãn chuẩn được người độc lập xác minh. Cần xem video/ngữ cảnh dài hơn cho trường hợp không rõ. Không tự sửa annotation gốc, không dùng judgments để train hoặc đổi validation trong đối chứng bên dưới. Mẫu phân tầng và tăng cường lỗi không đại diện tỷ lệ nhiễu của toàn dataset.

## Đối chứng 2×2

| Variant | Vùng nhìn | Thời gian |
|---|---|---|
| tubes_legacy | Nhóm người ghép bbox cũ | 16 frame / 8fps |
| pairs_legacy | Tối đa hai vùng cặp người gần nhau + toàn cảnh | 16 frame / 8fps |
| tubes_native | Nhóm người ghép bbox cũ | 13 frame / 5fps |
| pairs_native | Tối đa hai vùng cặp người gần nhau + toàn cảnh | 13 frame / 5fps |

Native là khoảng cách mẫu 6 frame trên nguồn 30fps, theo ví dụ X3D-S chính thức: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/ . Cửa sổ lấy mẫu kéo dài 2,4 giây giữa frame đầu và cuối; cấu hình cũ là 1,875 giây. Không mặc nhiên coi native sẽ tốt hơn sau chuyển miền.

Để không đánh giá trên nhãn sai do cửa sổ dài hơn: giữ cùng video/start/label cho cả bốn nhánh, chỉ nhận cửa sổ có >=70% frame dương với từng cấu hình nếu label=1, hoặc toàn bộ frame âm nếu label=0. Loại cửa sổ vượt cuối video hoặc đổi nhãn; báo số loại. Không so thẳng headline với 568 clip của lần expanded trước vì cohort có thể thay đổi.

Chọn 192 clip mỗi lớp từ train bằng round-robin theo video (384 tổng), giữ toàn bộ validation đủ điều kiện. Cùng seed, cùng head khởi tạo, cùng pretrained features, cùng optimizer/augmentation và tối đa 2 epoch probe + 2 epoch fine-tune mỗi nhánh. Cả train lẫn runtime gọi chung hàm crop. Crop cặp dựa trên hình học bbox, không phải bộ nhận biết “cặp đang đánh”; vùng toàn cảnh giữ lại để không mất hành động khi detector bỏ sót/ROI sai. Việc max score nhiều ROI có thể tăng báo nhầm và phải được đo, không coi crop mới tự động tốt hơn.

Script `scripts/fight_regions_ablation.py` có budget 25 phút, cache phase tối đa 60% và giữ ít nhất 8 phút cho các nhánh còn lại. Hết budget thì ghi incomplete, giữ checkpoint đã hoàn tất, không tự nối thêm. Lưu cả bốn báo cáo và runtime experimental; chỉ “promising_for_review_only” khi AP, in-video AUC và recall@precision75 đồng thời hơn nhánh control theo các mức đã định. Không có nhánh nào tự được duyệt demo.

## Lưu trữ và chạy

- Audit Drive: `MyDrive/SENTINEL/artifacts/fight_review_v1`.
- Input cũ: `artifacts/fight_expand_v1`, source subset `/content/fight_expand_data/UBI_FIGHTS`.
- Cache cũ `/content/fight_expand_cache`, cache đối chứng mới `/content/fight_review_cache`.
- Output: `MyDrive/SENTINEL/artifacts/fight_roi_time_v1`.
- RGB cache chỉ ở runtime. Checkpoint, manifest và báo cáo lưu Drive.

Nếu runtime cũ mất subset/cache, phải chuẩn bị lại bằng notebook expanded trước; không âm thầm giải nén/train lại khi người dùng chỉ muốn đọc báo cáo. Official test vẫn chưa được dùng. Trước khi tuyên bố đạt, cần test độc lập và thử camera cầm tay thực tế.
