# Bàn giao mở rộng hệ thống giám sát

## Bối cảnh và quyết định đã chốt

- Dự án local `D:\digital-surveillance-system`. Nhóm đã làm **đánh, ngã và chạy nhanh**, muốn bổ sung hai cảnh báo an ninh có dấu hiệu đủ khác với ba hành vi cũ. Kiểm tra pipeline và checkpoint thực tế trong bản local trước khi tích hợp.
- Hai hướng đã chốt là **phá hoại, đập phá tài sản** và **phát hiện người cầm vật thể sắc nhọn nhìn thấy được**. Đột nhập, trộm hàng, cướp tài sản, tư thế quỳ và các luật theo vùng từng được cân nhắc.
- Ưu tiên dữ liệu có nhãn sẵn. Với **dao phóng lợn cán dài**, người dùng đã làm rõ yêu cầu là tìm ảnh/video tương tự trên mạng và gán nhãn từ tư liệu đó. Người dùng chưa có video riêng để gán nhãn và không muốn phải tự quay.

## Phá hoại tài sản

Dùng [DCSASS](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset). Mẫu dương là nhóm `Vandalism` có nhãn clip `1`. Nhãn `0` là đoạn bình thường. Rà clip nếu phạm vi chỉ gồm đập phá vì Vandalism rộng hơn. Giữ cả người và tài sản trong khung, dùng cảnh đánh nhau làm đối chứng. [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) là nguồn video gốc.

## Người cầm vật sắc nhọn

- Đầu ra chung là **“Phát hiện người cầm vật sắc nhọn”**. Ưu tiên dao phóng lợn cán dài đúng mẫu Việt Nam, bên cạnh dao và mã tấu. Kiếm, đao và liềm là ứng viên tùy dữ liệu.
- Gán `dao_phong_lon` cho tập con cần thu thập. Giữ nhãn gốc để đánh giá riêng rồi ánh xạ các bounding box đã chọn về một lớp `sharp_object` khi train detector.
- Người dùng đã gửi hai ảnh vật đặt dưới đất và bốn ảnh có người cầm, gồm cảnh đêm, góc nhìn cao và ảnh cận. Các ảnh có thể dùng làm mẫu bổ sung. Chọn ảnh thấy rõ lưỡi và cán. Vẽ một khung bao phần lưỡi và cán nhìn thấy cho mỗi vật. Rà riêng các trường hợp chỉ thấy cán hoặc vật bị che.
- Có thể train detector bằng ảnh tĩnh và chạy trên từng frame. Để cảnh báo đúng **người đang cầm**, cần kiểm thử bước liên kết vật thể với người, nhất là cảnh nhiều người. Giữ riêng video hoặc ảnh từ vụ khác để đánh giá. Chia tập theo vụ việc hoặc video gốc để tránh trùng cảnh.

| Nguồn | Phạm vi |
|---|---|
| [Zenodo Dangerous Items](https://zenodo.org/records/16422779) | Dao và mã tấu có bounding box YOLO. |
| [Roboflow Weapon Detection](https://universe.roboflow.com/keshav-gct9z/weapon-detection-revmd-iqwyl) | Có lớp spear, sword và sickle. Chưa xác minh spear khớp mẫu dao phóng lợn. |
| [VnExpress – Hà Đông](https://vnexpress.net/sau-thieu-nien-cuop-bang-dao-phong-lon-bi-khoi-to-4725625.html) | Tư liệu camera an ninh ban đêm. |
| [VietnamNet – Sóc Sơn](https://vietnamnet.vn/nu-sinh-cam-dao-phong-lon-luon-pho-o-ha-noi-khai-di-danh-nhau-2081245.html) | Tư liệu người ngồi sau xe máy cầm vật cán dài. |
| [VnExpress Video](https://video.vnexpress.net/nhom-thanh-thieu-nien-che-dao-phong-duoi-nhau-tren-duong-4819340.html) | Nguồn ứng viên để trích khung hình. |

## Trạng thái và bước tiếp theo

Chưa tìm thấy bộ công khai có bounding box sẵn cho đúng mẫu dao phóng lợn Việt Nam. Đã tìm được nguồn ứng viên. Lần thử tải VietnamNet gặp lỗi xác minh chứng chỉ. Chưa tải tệp gốc, gán nhãn, đóng gói dữ liệu hay train checkpoint mới.

Agent tiếp theo kiểm tra điều kiện sử dụng của nguồn, thu ảnh hoặc trích frame thấy rõ lưỡi và cán, ghi nguồn và mã vụ việc, lọc ảnh trùng, gán nhãn `dao_phong_lon` và xuất định dạng YOLO. Báo số mẫu thực thu và giữ riêng dữ liệu kiểm thử khác vụ việc để đánh giá trước khi tích hợp.