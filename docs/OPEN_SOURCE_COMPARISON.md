# So sánh mã nguồn mở trước khi mở rộng

Đối chiếu ngày 28/10/2026. Không dùng số benchmark của dự án khác làm độ chính xác hay tốc độ của FightWatch.

| Phần việc | Nguồn chính thức đã đọc | Quyết định cho dự án |
| --- | --- | --- |
| Timeline/gán nhãn đoạn video | [Oxford VIA3](https://github.com/ox-vgg/via/tree/master/via-3.x.y), [định dạng dữ liệu](https://github.com/ox-vgg/via/blob/master/via-3.x.y/src/js/_via_data.js) | Tích hợp app VIA3 chính thức chạy offline. Chỉ viết adapter mỏng để giữ SHA/split/reviewer của dự án; không viết lại timeline nhiều chức năng. Giao diện tối giản trước đó giữ như lựa chọn phụ. |
| Nhãn bbox/track, nhiều người cộng tác | [CVAT](https://github.com/cvat-ai/cvat) | Thích hợp nếu mở rộng gán nhãn người/track và đội reviewer. Chưa đưa server annotation lớn vào luồng Windows/Drive chỉ cần khoảng 91 video temporal. Không dùng chức năng AI-assistance để tự biến nhãn dự đoán thành ground truth. |
| Fine-tuning/checkpoint/data split | [MMAction2 fine-tuning](https://mmaction2.readthedocs.io/en/latest/user_guides/finetune.html), [X3D README](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/x3d/README.md) | Đối chiếu cách dùng pretrained checkpoint, LR nhỏ và dữ liệu riêng. README X3D hiện nêu hỗ trợ testing cho checkpoint được port; không mặc nhiên khẳng định có thể đổi toàn bộ training sang stack này mà vẫn giữ classifier custom. |
| Backbone và model nhẹ | [PyTorchVideo models](https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/README.md), [X3D preprocessing](https://github.com/pytorch/hub/blob/master/facebookresearch_pytorchvideo_x3d.md) | Tiếp tục dùng triển khai X3D của PyTorchVideo đã có, không tự viết backbone. Giữ toàn bộ classifier RGB-aux đang cài. Không đổi input 16@12fps sang recipe Kinetics 13-frame chỉ vì ví dụ upstream khác; checkpoint hiện tại đã được fine-tune theo cấu hình dự án. Efficient-X3D là lựa chọn nghiên cứu CPU sau khi đo, chưa được cài/đánh giá và không lấy latency Samsung S8 làm latency máy này. |
| Nhận diện bạo lực bằng RGB + flow | [RWF-2000, repo tác giả](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) | Repo có code/weights Keras và đầu vào RGB + hai kênh optical flow. Là đối chiếu cho nhánh chuyển động, không phải checkpoint thay trực tiếp cho PyTorchVideo. Trang tác giả hiện không phát hành video vì riêng tư; không tự lấy mirror không rõ quyền hoặc đổi dataset ở lượt này. Không suy ra accuracy camera iPhone từ benchmark CCTV. |
| Fusion RGB + skeleton có triển khai sẵn | [RGBPoseConv3D trong MMAction2](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/rgbpose_conv3d/README.md) | Đối chiếu hướng hai luồng và fusion sớm. Recipe dùng heatmap skeleton, khác tensor pose/thời gian hiện tại; không ghép checkpoint vào custom head một cách tùy tiện. Nếu refinement hiện tại không tiến triển, dùng triển khai này làm baseline kế tiếp trên cùng split/nhãn và đo CPU trước khi tích hợp; chưa cài, chưa train và không bắt buộc tải NTU. |

## VIA3: tích hợp cụ thể

`scripts/fight_via.py prepare --pack <review_pack> --with-app` lấy nguyên app video annotator từ [Oxford](https://www.robots.ox.ac.uk/~vgg/software/via/app/via_video_annotator.html), khóa phiên bản 3.0.13 và SHA-256 trong code. HTML upstream không bị sửa; giữ nguyên thông báo BSD-2-Clause. Nếu nội dung upstream đổi, dừng để kiểm tra thay vì chạy mã mới âm thầm.

Mở `via_review.html` trong cùng thư mục review qua HTTP localhost, bấm mở project rồi gán nhãn temporal segment. Launcher mỏng dùng `project_load_json` của bản VIA đã khóa checksum, trong iframe cùng origin; không sửa HTML upstream, không tải video lên dịch vụ annotation. Vẫn có thể mở app gốc và load `via_project.json` thủ công. Dự án dùng anchor `FILE1_Z2_XY0`, tọa độ thời gian `z=[start,end]` và `av` chọn một action rõ ràng. Source-view mapping bị khóa với preview đã chuẩn bị; converter từ chối thay file, chồng khoảng, thiếu action hoặc đưa bbox thành nhãn thời gian.

Sau khi lưu project từ VIA:

```text
python -m scripts.fight_via import --pack <review_pack> --manifest <source_manifest.json> --project <via_saved.json> --reviewer <ma_nguoi_duyet> --method human --approve-reviewed --output <annotations.reviewed.json>
```

Nếu AI hỗ trợ duyệt, dùng `--method ai_assisted`. Bỏ `--approve-reviewed` thì dữ liệu nhập vẫn pending. File đã tồn tại không được ghi đè. Import không train, không đổi split và không tự đạt điều kiện preflight. Hai video iPhone local không được đưa vào gói public này.
