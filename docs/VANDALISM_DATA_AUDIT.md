# V0 – Audit dữ liệu phá hoại tài sản

Ngày kiểm: 16/11/2026. Mục tiêu hành vi: **đá/đập đồ vật, xô đổ/lật đồ, ném đồ để phá**. Bối cảnh demo: 1 phòng, camera laptop/điện thoại.

Yêu cầu dữ liệu: chỉ dùng nguồn **dùng được ngay, không cần đăng ký / chờ duyệt**. Mặc định tải trong phiên Colab; checkpoint và báo cáo lưu ra Drive.

## Kết luận nhanh

| Nguồn | Vai trò đề xuất | Tải trong phiên | Số clip dùng được | Dung lượng | Giấy phép |
|---|---|---|---|---|---|
| **Kinetics-700-2020** `smashing`, `breaking glass` | **Dương tính chính** | ✅ S3 CVDF, mỗi lớp 1 file tar | 1.281 train + 95 val | ~1.6 GB train, ~0.12 GB val | Nhãn CC BY 4.0; video thuộc chủ YouTube, dùng cho nghiên cứu |
| **Kinetics-700-2020** 14 lớp âm khó (bảng dưới) | **Âm khó chính** | ✅ như trên | ~11.000 train (lấy mẫu con) | ~12 GB nếu tải đủ; lấy mẫu con ~3 GB | như trên |
| AVA v2.2 `hit (an object)`, `push (an object)`, `throw` | **Chỉ làm baseline V1** | ✅ S3 (phim 15 phút/video) | 177 / 387 / 249 lần gán nhãn trong train | 0.2–1.5 GB/phim | Nhãn CC BY 4.0 |
| **ASHEN** `Vandalism` + `normal` (CCTV) | **Dương tính phụ** (đá/lật đồ thật) + âm tính cùng miền CCTV | ✅ Hugging Face, không cần đăng nhập | 82 train + 18 test (có mốc thời gian) | 1.62 GB + ~8 GB normal lấy mẫu + 4.7 GB normal test | CC BY-NC 4.0 |
| UCF-Crime `Vandalism` | Không dùng riêng (50 video, trùng tên với ASHEN) | ⚠️ Dropbox, file lớn | 50 video | cả bộ ~100 GB | Không ghi rõ |
| **Charades** | **Âm khó trong nhà**: ném đồ bình thường, sinh hoạt | ✅ S3 AI2, chỉ lấy file cần bằng HTTP Range | 1.847 đoạn ném + mẫu sinh hoạt khác | ~3–4 GB (zip gốc 16.3 GB) | Phi thương mại |
| NTU RGB+D 120 (`kicking something`) | **Loại** | ❌ phải đăng ký, chờ duyệt, đăng nhập | — | 260 GB | Theo thoả thuận phát hành |
| PKU-MMD | **Loại** | ⚠️ Google Drive/Baidu, chưa xác nhận có lớp đá đồ vật | — | — | Chỉ nghiên cứu |

**Bộ dữ liệu đã chốt (mục 8):**
- **Dương tính:** Kinetics-700 (`smashing` + `breaking glass`) và ASHEN Vandalism.
- **Âm khó:** Kinetics 17 lớp, Charades (ném đồ bình thường trong nhà), ASHEN normal.
- **Baseline V1:** AVA.

Tất cả tải tự động trong phiên Colab được, không cần tài khoản.

## 1. Kinetics-700-2020 (nguồn chính)

- **Host:** `https://s3.amazonaws.com/kinetics/700_2020/{train,val}/k700_{train,val}_NNN.tar.gz`, trong đó `NNN` = số dòng của lớp trong `mmaction2_demo/repo/tools/data/kinetics/label_map_k700.txt` (đã kiểm: `val_557` chứa `smashing/…mp4`).
- **Nhãn:** `https://s3.amazonaws.com/kinetics/700_2020/annotations/{train,val}.csv`.
- **Clip:** dài 10 s, từ YouTube, độ phân giải đa dạng (phần lớn ≥ 360p). Nhiều clip `smashing` quay trong nhà (phòng đập phá, đập TV, máy in, chén đĩa), khá gần bối cảnh demo.

**Dương tính**

| Lớp | Tar | Train | Val | Kích thước train |
|---|---|---|---|---|
| smashing | 557 | 703 | 47 | 928 MB |
| breaking glass | 057 | 578 | 48 | 653 MB |

**Âm khó** (giống chuyển động hoặc tiếp xúc đồ vật nhưng không phá)

| Nhóm | Lớp (tar) | Train mỗi lớp |
|---|---|---|
| Đánh, đá nhưng hợp lệ | breaking boards (056), punching bag (459), chopping wood (102), kicking soccer ball (280) | 846–971 |
| Ném hợp lệ | throwing ball (621), bouncing ball (049) | 560–959 |
| Dọn dẹp, di chuyển đồ | moving furniture (340), opening door (347), closing door (115), sweeping floor (593), mopping floor (334), vacuuming floor (663), unboxing (646), folding clothes (214) | 539–968 |
| Đánh nhau (để không nhầm sang strike) | punching person (460), slapping (553), wrestling (695) | 607–968 |

**Loại khỏi train, cần xem mẫu trước khi quyết:**
- `throwing tantrum` (625): phần lớn là trẻ con khóc lóc, chỉ đôi khi ném đồ, nên nhãn rất nhiễu.
- `using a sledge hammer` (652): phá dỡ công trình hợp pháp, mập mờ.
- `tagging graffiti` (606) và `spray painting` (573): là phá hoại nhưng **ngoài phạm vi demo** (đá/đập/xô/ném trong phòng).

**Chia tập:** train/dev từ split train của Kinetics, nhóm theo `youtube_id` để không lọt cùng video gốc sang hai tập. **Test** = split val của Kinetics (95 clip dương tính + val của các lớp âm), cộng video phòng thật của người dùng (đánh giá local ở V6, không train).

## 2. AVA v2.2 (baseline V1)

- Nhãn: `https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip`. Video: `https://s3.amazonaws.com/ava-dataset/trainval/<id>.mkv` (đã kiểm: tải được, ~263 MB/phim).
- Số lần gán nhãn (train / val), kèm số phim:
  - `hit (an object)` id 34: 177 lần / 61 phim; 53 lần / 18 phim.
  - `push (an object)` id 46: 387 / 92; 92 / 25.
  - `throw` id 58: 249 / 108; 99 / 30.
- **Hạn chế:** `push an object` phần lớn là đẩy cửa, đẩy xe; `throw` gồm cả ném bóng. Nhãn quá rộng để làm dương tính, chỉ dùng làm mốc so sánh.
- **Lưu ý kỹ thuật cho V1:** reply của core chỉ trả top-4 nhãn AVA có điểm ≥ 0.4 cho mỗi người, nên không đủ để đo baseline. V1 sẽ dùng script riêng chạy bằng `mmaction2_demo/env`, *import* `ActionEngine` để lấy đủ 80 điểm. **Không sửa** `action_engine.py` / `worker.py`.

## 3. ASHEN (bật, dương tính phụ)

- `https://huggingface.co/datasets/dead4556/ASHEN-Dataset`, tải từng file bằng `huggingface_hub`, không cần đăng nhập. Giấy phép CC BY-NC 4.0.
- **Split chính thức** `splits_v2_native/`:
  - Vandalism có 82 video train và 18 video test.
  - Danh sách ghi đuôi `.npy` (feature của tác giả), dùng file `.mp4` cùng tên.
- **Nhãn:**
  - Video train chỉ có nhãn chung cho cả video.
  - Video test có mốc khung hình trong `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`, định dạng `video nhãn bắt_đầu1 kết_thúc1 bắt_đầu2 kết_thúc2` (`-1` = không có).
- **Nội dung:** CCTV thật, nhiều cảnh ngoài trời, độ phân giải thấp, video dài. Đây là nguồn duy nhất dùng được ngay có cảnh **đá đồ / lật đồ** thật.
- **Cách dùng:**
  - *Dương tính train:* học kiểu multiple-instance. Mỗi video được cắt thành nhiều cửa sổ, chỉ lấy vài cửa sổ điểm cao nhất làm dương tính, trọng số thấp hơn Kinetics.
  - *Âm tính cùng miền:* lấy mẫu khoảng 80 video `normal/train` (~8 GB), để model không học lối tắt "hình CCTV = phá hoại".
  - *Âm khó tuỳ chọn:* `anomaly/Fighting` (81 video train), để đánh nhau vẫn thuộc strike, không thành phá hoại.
  - *Test:* 18 video Vandalism có mốc thời gian + 150 video `normal/test` (4.7 GB), đánh giá theo sự kiện trên miền CCTV.

## 4. Nguồn đã loại

- **NTU RGB+D 120:** có lớp `kicking something` nhưng phải đăng ký tài khoản ROSE Lab, chờ duyệt và đăng nhập mới tải được (260 GB), nên không tự tải trong phiên Colab được.
- **PKU-MMD:** tải qua Google Drive/Baidu, chưa xác nhận có lớp đá đồ vật; chỉ dùng cho nghiên cứu.
- **UCF-Crime riêng:** 50 video, link Dropbox chia theo phần lớn, không ghi giấy phép; đã có trong ASHEN.
- **Something-Something v2:** cảnh cận tay đẩy đồ rơi, không hợp góc nhìn toàn thân 1–5 m; phải đăng ký Qualcomm.
- **Bộ "614 clip" từng ghi trong kế hoạch:** chưa tìm được nguồn phát hành công khai, nên bỏ.

## 5. Khoảng trống và rủi ro

- **Thiếu cảnh "xô đổ/lật đồ" và "đá đồ" trong nhà**, vì chưa có bộ công khai nào tải tự động được. Kinetics `smashing` phủ phần "đập". Kiểm thử V6 trên camera thật sẽ cho biết có cần bổ sung không.
- **Lệch miền:** clip YouTube khác camera laptop. Bù lại bằng augment (nhoè, nén JPEG, thiếu sáng) và đánh giá trên video phòng thật.
- **Nhầm với strike:** đã đưa `punching person`, `slapping`, `wrestling` vào âm khó; runtime còn hạ điểm vandalism khi core báo strike/grapple cùng vùng.
- **Video Kinetics bị gỡ khỏi YouTube:** không ảnh hưởng, vì CVDF lưu bản sao. SHA của từng tar được ghi vào `protocol.json` ở lần tải đầu.

## 6. Quyết định đã chốt (16/11/2026)

- **Phạm vi:** chỉ đá/đập/xô/ném. Không đưa `tagging graffiti`, `spray painting` vào train.
- **ASHEN:** lúc đầu tắt. Sau khi tra thêm (mục 7) thì **bật**, vì đây là nguồn duy nhất dùng được ngay có cảnh đá/lật đồ.
- **Không dùng nguồn phải đăng ký** (NTU, MSAD, SSv2, Moments in Time), vì chờ duyệt mất thời gian.
- **Thêm Charades** làm âm khó trong nhà.
- **`throwing tantrum`, `using a sledge hammer`:** bỏ cả hai.
- **Khoảng trống "xô đổ/lật đồ", "đá đồ":** đã tra thêm nguồn (kết quả ở mục 7).

## 7. Tra thêm nguồn cho "xô đổ / đá đồ"

| Nguồn | Có gì | Tải trong phiên | Đánh giá |
|---|---|---|---|
| AVA `kick (an object)` (id 35) | 42 lần gán nhãn trong cả train + val | ✅ | Quá ít; head AVA của core cũng không đánh giá lớp này |
| MSAD (Griffith, 720 video, 14 bối cảnh) | Lớp Vandalism: đập kính, phá cửa, phá hoại khác; lớp Object Falling | ⚠️ Phải điền form, chờ email duyệt; sau đó link Google Drive có thể tải bằng `gdown` | Ứng viên bổ sung tốt nhất nếu chấp nhận đăng ký một lần |
| NTU RGB+D 120 A24 `kicking something` | Cảnh trong phòng lab, toàn thân, đúng kiểu "đá đồ" | ❌ Đăng ký + đăng nhập; lớp nằm rải trong 32 file zip lớn (tổng 260 GB) | Không thực tế |
| Something-Something v2 (`Tipping something over`, `Pushing something so that it falls off the table`) | Cảnh cận tay | ❌ Đăng ký Qualcomm | Lệch góc nhìn (cận tay, không toàn thân) |
| Moments in Time (`kicking`, `smashing`…) | Clip 3 s, đa dạng | ❌ Điền form | Không tải tự động |
| HMDB51 `kick` | Phần lớn đá người trong phim | ✅ | Không đúng hành vi |
| Hugging Face (tìm "vandalism", "smashing", "rage room", "property damage") | Chỉ có 1 bộ ảnh cột điện | — | Không có bộ video phù hợp |
| Tải video YouTube bằng `yt-dlp` | — | ⚠️ Colab thường bị YouTube chặn IP; trái điều khoản YouTube | Không đề xuất |

**Kết luận:** không có nguồn công khai nào **tải tự động trong phiên** cho đúng cảnh xô đổ/đá đồ trong nhà. Hai hướng:
1. **Đề xuất:** train bằng Kinetics (`smashing` + `breaking glass`), kiểm ở V6 bằng camera thật; nếu xô/đá bị bỏ sót thì mới bổ sung.
2. Đăng ký **MSAD** một lần (người dùng tự điền form). Có link thì notebook tải bằng `gdown` trong phiên và dùng lớp Vandalism làm dương tính bổ sung. Đây là CCTV nhiều bối cảnh, độ phân giải không cố định, nên cần xem mẫu trước khi dùng.

## 8. Bộ dữ liệu đã chốt

| Vai trò | Nguồn | Số lượng | Tải trong phiên |
|---|---|---|---|
| Dương tính chính | Kinetics-700 `smashing`, `breaking glass` (train) | 1.281 clip × 10 s | ~1.6 GB |
| Dương tính phụ | ASHEN Vandalism train (multiple-instance, trọng số thấp) | 82 video | ~1.3 GB |
| Âm khó | Kinetics-700, 17 lớp (lấy mẫu con, ví dụ 200 clip/lớp) | ~3.400 clip | ~3 GB |
| Âm khó trong nhà | Charades: 11 lớp ném đồ + mẫu sinh hoạt khác | ~1.850 đoạn + ~1.500 đoạn | ~3–4 GB |
| Âm tính cùng miền CCTV | ASHEN normal train (lấy mẫu) | ~80 video | ~8 GB |
| Test | Kinetics val (95 dương + val các lớp âm), ASHEN test (18 Vandalism + 150 normal), video phòng thật (V6, local) | — | ~5.5 GB |

**Chia tập để không lọt dữ liệu giữa train/dev/test:**
- Kinetics: theo `youtube_id`.
- Charades: theo người quay (`subject`).
- ASHEN: giữ split chính thức, dev cắt từ phần train.

**Còn thiếu:** cảnh "xô đổ đồ đạc" quay trong nhà. ASHEN có cảnh lật đồ nhưng là CCTV ngoài trời. Phải kiểm ở V6 bằng camera thật.

## 9. Việc tiếp theo

- **V1:** script baseline AVA 34/46/58, chạy local bằng `mmaction2_demo/env`, đo trên một mẫu nhỏ của tập test: Kinetics val + ASHEN test.
- **V2:** `extensions/vandalism/prepare_vandalism.py` tạo manifest từ `datasets/vandalism/sources.json`:
  - Kinetics: tải tar theo lớp, lấy mẫu con lớp âm.
  - ASHEN: tải theo split chính thức.
  - Charades: dùng HTTP Range lấy đúng đoạn cần.
  - Chia tập theo `youtube_id` / `subject` / split ASHEN.
- **V3:** notebook Colab tải dữ liệu trong phiên, trích feature V-JEPA 2-L / VideoMAE V2-B, train probe, lưu checkpoint + báo cáo ra Drive.

Nguồn đã tra: [Kinetics-700-2020 readme](https://s3.amazonaws.com/kinetics/700_2020/K700_2020_readme.txt), [AVA](https://research.google.com/ava/), [ASHEN](https://huggingface.co/datasets/dead4556/ASHEN-Dataset), [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/), [Charades loader](https://huggingface.co/datasets/HuggingFaceM4/charades/blob/main/charades.py), [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/), [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).
