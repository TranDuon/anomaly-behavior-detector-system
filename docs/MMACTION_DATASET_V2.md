# Đề xuất dữ liệu cho Interaction Fusion v2

## Phạm vi hành vi

Model chỉ cảnh báo xô xát tay chân giữa người với người trong camera cố định hoặc điện thoại dùng như camera: đấm, đá, tát, đẩy/xô mạnh, vật lộn và đánh liên tiếp. Không coi việc hai người xuất hiện, đi lại gần, đứng sát, bắt tay, ôm, high-five, vẫy tay, chạy cùng nhau hoặc rời khỏi khung hình là đánh nhau. Vũ khí, cháy nổ, bạo lực phim ảnh và thể thao đối kháng nằm ngoài phạm vi pilot này.

Log camera cục bộ ngày 29/10 cho thấy hai yêu cầu dữ liệu quan trọng:

- cần giữ cả đoạn trước và sau hành vi; clip chỉ cắt đúng khoảnh khắc đánh sẽ dạy model báo quá sớm;
- cần nhiều hard-negative có hai người trong phòng, vì nhánh v1 tăng điểm mạnh chỉ do người thứ hai bước vào hoặc hai người đứng gần nhau.

## Bộ dữ liệu đề xuất

### 1. Lõi đúng miền: AIRTLab

Nguồn: <https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos>

Đây là nguồn sát log nhất: quay trong cùng một phòng, camera cố định/điện thoại, 30 FPS, 2–4 diễn viên và có hai góc camera cho cùng một màn diễn. Bộ non-violent được thiết kế riêng cho false positive gồm ôm, high-five, vỗ tay, ăn mừng và cử chỉ mạnh.

- dùng toàn bộ 120 clip non-violent;
- dùng các clip violent có đấm, đá, tát và ẩu đả tay chân;
- loại các đoạn chỉ có súng, dao hoặc gậy nếu mục tiêu vẫn là đánh nhau tay không;
- hai góc của cùng một màn diễn phải luôn ở cùng split để tránh rò rỉ.

### 2. Tương tác hai người cận cảnh trong phòng: NTU RGB+D 120

Nguồn chính thức: <https://rose1.ntu.edu.sg/dataset/actionRecognition/>

Chỉ lấy các lớp mutual action quay trong nhà: A50 punch/slap, A51 kick, A52 push làm positive; A53 pat on back, A54 point, A55 hug, A56 give object, A58 handshake, A59 walk towards, A60 walk apart, A112 high-five, A113 cheers, A116 follow, A117 whisper, A118 exchange object và A120 rock-paper-scissors làm hard-negative. Loại hoàn toàn các lớp có vũ khí A106–A110.

Đây là nguồn lớn để học cặp đối chứng trực tiếp trong cùng phòng, cùng thiết bị và nhiều góc quay. Chia split theo subject/setup chính thức; các góc quay đồng thời của cùng một màn diễn phải nằm cùng split.

### 3. Tương tác indoor bổ sung: SBU Kinect Interaction

Nguồn: <https://www3.cs.stonybrook.edu/~cvl/dataset>

SBU có approaching, departing, pushing, kicking, punching, exchanging objects, hugging và shaking hands, đều quay trong môi trường Kinect trong nhà. Dùng push/kick/punch làm positive; các lớp còn lại làm hard-negative. Chia theo cặp người, không chia ngẫu nhiên từng clip.

### 4. Cảnh lớp học: VID classroom subset

Nguồn dữ liệu: <https://doi.org/10.7910/DVN/N4LNZD>

VID có nhóm classroom violence và non-violence, video 3–12 giây do diễn viên quay bằng thiết bị cầm tay, đồng thời cung cấp video RGB và pose. Chỉ lấy classroom subset và negative trong cùng bối cảnh. Gắn cờ camera di chuyển để báo cáo riêng, không trộn điểm với camera cố định.

### 5. Nguồn bị loại khỏi train v2

Không đưa UBI-Fights, NTU CCTV-Fights, RWF-2000, SCFD, RLVS, Hockey Fight, Movie Fight, Crowd Violence hoặc UT-Interaction vào train/development/calibration/test của checkpoint indoor. Chúng khác miền phòng/lớp học hoặc có nhãn quá thô. Script tạo manifest sẽ tự cách ly mọi dataset ngoài AIRTLab, NTU RGB+D 120, SBU và VID-classroom.

## Tỷ lệ mục tiêu cho manifest v2

Đơn vị split là `performance/source group`, không phải clip:

| Nhóm | Tỷ lệ cửa sổ train | Mục đích |
|---|---:|---|
| Đấm/tát, đá, đẩy trong phòng/lớp | 35% | sửa bỏ sót của model gốc |
| Hai người đi vào/tiếp cận/đứng gần/đi cùng/rời đi | 25% | sửa lỗi “vừa bước vào đã báo đỏ” |
| Ôm/bắt tay/high-five/vỗ vai/chỉ tay/đưa đồ | 25% | hard-negative tương tác gần |
| Một người chuyển động mạnh/cảnh bình thường | 15% | chặn cảnh báo khi chỉ còn một người |

Mỗi positive window phải có ít nhất một negative “gần giống” từ cùng loại bối cảnh. Với AIRTLab, khoảng trước hành vi được rà bằng contact sheet ở nhiều mốc thời gian và ghi rõ `ai_assisted_contact_sheet`; các mốc này là nhãn hỗ trợ để thử nghiệm, chưa phải nhãn human-gold. Mỗi nguồn gốc, cặp diễn viên, màn diễn nhiều camera và các clip cắt từ cùng video phải nằm trọn trong một split.

Với clip positive của NTU/SBU/VID, không được gán toàn bộ clip là `fight`. Phải đánh dấu riêng đoạn đi vào/tiếp cận/đứng gần trước va chạm, đoạn đánh thật và đoạn tách ra sau đánh. Ít nhất 25% nguồn positive phải chứa pre-fight context đã gán nhãn negative; manifest indoor phải có tối thiểu 30 hard-negative tương tác gần, gồm `approach`, `stand_close`, `walk_together`, `hug`, `handshake_greet` hoặc `highfive_clap`.

## Chia tập và điều kiện trước khi train

- train 60%, development 15%, calibration 15%, test 10% theo source group;
- thêm một bộ acceptance cục bộ riêng, gồm các video log camera đã chạy; bộ này không được đưa vào train, chọn epoch hoặc chọn threshold;
- gắn nhãn interval `fight`, từng loại hard-negative cụ thể và `uncertain`; bỏ cửa sổ chạm vùng `uncertain`;
- lưu SHA-256, nguồn, giấy phép, participant/performance group, môi trường, camera motion, FPS và start/end gốc;
- chạy exact hash và perceptual-hash grouping trước khi split;
- manifest tối thiểu 200 nguồn, bắt buộc có AIRTLab và ít nhất một nguồn indoor bổ sung trong NTU RGB+D 120, SBU hoặc VID-classroom; chỉ chạy notebook v2 khi `readiness=ready`.

## Gate nghiệm thu phù hợp log

Ngoài precision/recall/FPR tổng, báo cáo riêng:

- false alarm trên `enter/approach/stand-close`: 0 trên bộ acceptance cục bộ;
- false alarm trên `hug/handshake/high-five/gesture`: không quá 5%;
- recall cho punch/kick/push/wrestle trong phòng: tối thiểu 80%;
- median onset delay không quá 1,0 giây; p95 không quá 1,8 giây;
- báo cáo theo từng dataset và từng nhóm môi trường, không chỉ pooled score.

Các ngưỡng này là mục tiêu pilot ban đầu; chỉ điều chỉnh trên calibration công khai, không điều chỉnh theo video acceptance cục bộ.
