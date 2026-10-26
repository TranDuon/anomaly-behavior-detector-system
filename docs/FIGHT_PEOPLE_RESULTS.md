# Kết quả tập trung vào người — 27/10/2026

Nguồn là UBI-Fights trong pilot 48 video, không phải UCSD Ped2. Không training bài toán giao thông; các cảnh xe cộ trong nguồn cũ là một phần nền/bình thường đa dạng. Theo yêu cầu demo, đã lọc theo người đủ rõ thay vì coi mọi cảnh có xe là nhiễu.

Code training: `ffd954a`. GPU: L4, runtime Colab cũ được sử dụng lại, không cần mount quyền mới. Audit + pilot + xuất checkpoint mất 77,56 giây do dùng cache clip/feature và chỉ train subset nhỏ; không phải train toàn dataset. 72 test local đã qua trước khi chạy.

## Clip được giữ và bị loại

| Phần nguồn | Nhãn | Clip nguồn | Nhiều người đủ rõ | Một người đủ rõ | Người nhỏ/không ổn định | Không detect được người |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | Bình thường | 203 | 71 | 32 | 45 | 55 |
| Train | Đánh nhau | 68 | 42 | 11 | 10 | 5 |
| Validation | Bình thường | 84 | 14 | 28 | 20 | 22 |
| Validation | Đánh nhau | 25 | 10 | 3 | 3 | 9 |

- Train mới: **132 clip = 42 đánh nhau + 71 bình thường nhiều người + 19 bình thường một người**. Control một người chiếm 14,4%, lấy riêng từ train; không thay đổi nhãn clip nào.
- Validation mục tiêu: **24 clip = 10 đánh nhau + 14 bình thường nhiều người**, giữ nguyên video/split cũ. Chỉ còn hai video có cả hai nhãn, chưa đủ cổng ít nhất ba video hỗn hợp.
- Bộ lọc cũng loại 15/25 clip dương validation do người nhỏ/che khuất/detector không đủ người. Không coi những clip đó là không có đánh nhau hoặc làm như model đã phát hiện được. Toàn bộ 109 clip gốc vẫn được chấm riêng.
- 91/380 clip nguồn không detect được người; không gọi tất cả 91 clip đó là giao thông vì có cả người bị bỏ sót/che khuất. Quyết định lọc không dùng nhãn hay score.

Ảnh audit đã xác minh: `N_662` giây 30 chủ yếu xe cộ nên bị loại; `N_450` giây 9 có người ra vào bình thường nên được giữ làm mẫu âm; `F_135` giây 1 có người tương tác ở trước xe nên vẫn giữ. Xe ở nền không tự loại một cảnh đánh nhau. YOLO có thể nhầm poster là người; gallery là bằng chứng để kiểm toán, không thay thế hoàn toàn kiểm tra thủ công.

## So sánh trên đúng cùng 24 clip có nhiều người

| Model | AP | ROC AUC | Recall tối đa ở precision >=75% |
| --- | ---: | ---: | ---: |
| Baseline v1 tính lại trên cohort | 0,6162 | 0,6786 | 10% |
| Interaction v2 tính lại trên cohort | **0,6953** | **0,7214** | 60% |
| Full-frame + hard negatives v2 tính lại | 0,5967 | 0,6429 | 60% |
| Train lại people_focus_v1 | 0,5298 | 0,5786 | 10% |

**Train lại sau lọc chưa tốt hơn checkpoint v2 trên cùng cohort. Không thay model demo.** Bộ lọc đưa dữ liệu gần mục tiêu hơn nhưng làm số clip học giảm từ 271 xuống 132; không chứng minh train càng hẹp càng tốt. Cũng không chứng minh xe cộ là nguyên nhân chính gây kém chất lượng.

Số 60% recall tại precision >=75% của baseline v2 chỉ là sweep ngưỡng trên 10 clip dương/14 âm đã dùng để phát triển, với chỉ hai video hỗn hợp; không phải model đã nghiệm thu. Không so AP 0,695 trên 24 clip với AP 0,467 trên 109 clip để tuyên bố chất lượng được nâng lên.

Trên toàn 109 clip gốc ở ngưỡng của model people-focused: AUC 0,6624, AP 0,3840, precision 38,7%, recall 48%. Tất cả checkpoint vẫn `approved_for_demo=false`, official test không dùng.

## Trạng thái và bước tiếp theo

- Artifact: `SENTINEL/artifacts/fight_people_focus_v1/`, gồm audit từng clip, gallery ảnh ba thời điểm, cohort manifest, baseline cùng cohort, history, checkpoint, predictions, clip lỗi và ZIP runtime pilot.
- Số liệu đầy đủ: [experiments/fight_people_focus_v1.json](experiments/fight_people_focus_v1.json).
- Notebook: `colab/fight_people_drive.ipynb`, tự pull code, dùng dữ liệu/cache cũ và có watchdog 12 phút. Script đã hoàn tất, không chạy một phiên train dài hơn.
- Phạm vi các lần thử tiếp theo ưu tiên người là chủ thể chính. Cần mở rộng số cảnh người/nhóm người đa dạng từ phần train, giữ mẫu âm nhiều người, kiểm tra trường hợp người bị che/gần và tạo đánh giá độc lập đủ lớn. Không chỉ lặp epoch trên 132 clip.
- Runtime Colab được giữ mở theo yêu cầu hiện có; không tự ngắt/xóa phiên hoặc đổi checkpoint local.
