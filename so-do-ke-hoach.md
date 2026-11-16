# Sơ đồ mermaid – Kế hoạch mở rộng
Dán từng khối vào https://mermaid.live (chỉ lấy phần bên trong ```mermaid … ```).

## Vòng kiểm tra sau mỗi thay đổi

```mermaid
flowchart LR
    CH["Một thay đổi nhỏ<br/>(1 commit)"] --> PT["pytest<br/>chỉ 12 fail đã biết?"]
    PT -- "không" --> RV["Revert / sửa lại<br/>không commit"]
    PT -- "có" --> GD["compare_behavior_traces<br/>4 nhánh khớp golden 100%?"]
    GD -- "không" --> RV
    GD -- "có" --> PF["Đo timings_ms core<br/>tăng ≤ 10%? VRAM < 5.5 GB?"]
    PF -- "không" --> RV
    PF -- "có" --> OK["Giữ thay đổi<br/>(commit local)"]
    RV --> CH
    OK --> NX["Thay đổi tiếp theo"]
```

## A.1 Luồng hệ thống tổng thể

```mermaid
flowchart LR
    subgraph FE["Trình duyệt<br/>frontend/fight.html"]
        direction TB
        UI["Chọn nguồn<br/>upload / camera / URL"]
        MJ["Hiển thị MJPEG<br/>/api/stream"]
        ST["Poll /api/status<br/>+ /api/events (700 ms)"]
    end

    subgraph SRV["api/fight_server.py · FightProcessor<br/>(.venv · Py3.13 · torch CPU)"]
        direction TB
        RL["read_live<br/>cv2.VideoCapture<br/>mẫu 3.75 fps"]
        RUN["run()<br/>runtime.push()<br/>record / trace / events"]
        PUB["publish_live<br/>box tracking → annotate()<br/>→ JPEG"]
    end

    subgraph RT["fight/mmaction_runtime.py<br/>MMActionRuntime"]
        direction TB
        PS["push_separate<br/>buffer 9 frame<br/>quyết định mỗi 16/30 s"]
        GATE["update_decision<br/>EventGate từng nhãn"]
        VIEW["canonical_posture_view<br/>display_geometry"]
    end

    subgraph WK["mmaction2_demo/worker.py<br/>(env riêng · Py3.10 · torch 2.0.1 cu118 · cuda:0)"]
        direction TB
        BR["4 nhánh hành vi<br/>(xem A.2)"]
    end

    UI --> RL --> RUN --> PS
    PS -->|"stdin JSON +<br/>shared memory"| BR
    BR -->|"reply JSON"| GATE --> VIEW --> PUB --> MJ
    RUN --> ST
```

## A.2 Kiến trúc 4 nhánh hành vi

```mermaid
flowchart LR
    IN[("9 frame BGR<br/>3.75 fps")]

    subgraph SH["Phần dùng chung"]
        direction TB
        DET["Faster R-CNN R50<br/>người trên frames[4]<br/>(≤ 6 người)"]
        BB["SlowOnly R101 AVA<br/>đóng băng · 8 frame · 256 px"]
        OUT["80 điểm AVA / người<br/>feature 2048-d / người<br/>4096-d / cặp + 11 motion"]
        DET --> BB --> OUT
    end

    subgraph B1["Nhánh 1 · STRIKE (đánh)"]
        direction TB
        S1["AVA id 64<br/>fight/hit a person"]
        S2["Ngưỡng<br/>original_calibration.json"]
        S3["EventGate<br/>xác nhận 0.4 s"]
        S1 --> S2 --> S3
    end

    subgraph B2["Nhánh 2 · GRAPPLE (vật lộn)"]
        direction TB
        G1["Head three_actions.pt[1]<br/>trên feature cặp"]
        G2["Chỉ chẩn đoán<br/>alarm tắt"]
        G1 --> G2
    end

    subgraph B3["Nhánh 3 · FAST RUNNING (chạy nhanh)"]
        direction TB
        R1["InteractionDistanceHead<br/>interaction_fixed_running.pt"]
        R2["running_evidence<br/>detect lại frames[-1]<br/>+ recovery"]
        R3["RunningConfirmation<br/>track từng người · 0.8 s"]
        R1 --> R2 --> R3
    end

    subgraph B4["Nhánh 4 · FALL (ngã / nằm)"]
        direction TB
        F1["yolov8n-pose 640 px<br/>trên frames[-1]"]
        F2["LyingPosture<br/>luật hình học khớp COCO"]
        F3["2 lần quan sát<br/>latch tới khi đứng dậy"]
        F1 --> F2 --> F3
    end

    IN --> DET
    OUT --> S1
    OUT --> G1
    OUT --> R1
    IN --> F1

    S3 --> M["Gộp kết quả<br/>action_states<br/>active_behaviors<br/>box hiển thị"]
    G2 --> M
    R3 --> M
    F3 --> M
```

## A.3 Huấn luyện hiện tại

```mermaid
flowchart LR
    D["Dataset gốc<br/>(AIRTLab, RLVS, GMDCSA24, running…)"] --> P["prepare_*.py<br/>manifest + chia theo nguồn/diễn viên<br/>kiểm leakage, SHA"]
    P --> F["train_*.py --stage features<br/>cache .npz (feature AVA)"]
    F --> T["train_*.py --stage train<br/>head nhỏ, chọn ngưỡng trên dev, test cuối"]
    T --> Z["zip: name.pt + name.json<br/>+ report.json + protocol.json"]
    Z --> I["install_*.py<br/>kiểm thành viên zip, SHA, SPEC<br/>copy atomic → weights/"]
    I --> W["Worker nạp checkpoint"]
```

## B.1 Kiến trúc đích (core + extension worker)

```mermaid
flowchart LR
    subgraph SRV["fight_server.py"]
        REG["fight/behaviors.py<br/>BehaviorRegistry (core + extension)"]
        ANN["annotate / record / status<br/>đọc nhãn, màu, gate từ registry"]
    end

    subgraph RT["MMActionRuntime"]
        CORE_G["Gate 4 nhánh core<br/>(KHÔNG ĐỔI)"]
        EXT_G["Gate extension<br/>(mỗi hành vi 1 EventGate riêng)"]
        MERGE["Gộp action_states / active_behaviors<br/>extension chỉ thêm, không ghi đè core"]
        BUD["Theo dõi timings_ms<br/>hạ cadence extension nếu core chậm"]
    end

    SHM[("SharedFrameBuffer<br/>9 frame / bước")]

    subgraph CW["Core worker - mmaction2_demo/env (KHÔNG ĐỔI)"]
        direction TB
        C1["1 · strike"]
        C2["2 · grapple"]
        C3["3 · fast running"]
        C4["4 · fall"]
    end

    subgraph XW["Extension worker MỚI - extensions/env (Py3.11, torch 2.5+, cu12x)"]
        VAN["vandalism<br/>V-JEPA 2 ViT-L (frozen) + attentive probe<br/>hoặc VideoMAE V2-B fine-tune"]
        SHP["sharp_object<br/>RF-DETR (nano/small) fine-tune<br/>+ liên kết cổ tay từ pose"]
    end

    REG --> ANN
    SHM --> CW
    SHM -->|"đọc, cadence 1–2 bước"| XW
    CW -->|reply core| CORE_G
    XW -->|"reply.extensions[name]"| EXT_G
    CORE_G --> MERGE
    EXT_G --> MERGE
    MERGE --> ANN
    CW -. timings .-> BUD
    XW -. timings .-> BUD
    BUD -. throttle .-> XW
```

## E.1 Phá hoại tài sản: dữ liệu → train → tích hợp

```mermaid
flowchart LR
    S1["AVA v2.2 (phim, HQ)<br/>hit/push object, throw"] --> A
    S2["Charades (trong nhà)<br/>Throwing… somewhere"] --> A
    S3["NTU RGB+D 120 / PKU-MMD<br/>kicking something, throw"] --> A
    S4["UCF-Crime / DCSASS<br/>(phụ, ngoài miền)"] --> A
    S5["Âm khó: AIRTLab/RLVS,<br/>running, fall, sinh hoạt"] --> A
    A["V0 Audit<br/>độ phân giải, giấy phép, số clip"] --> P["V2 prepare_vandalism.py<br/>manifest, chia theo video gốc, dedup"]
    P --> F["V3 features: V-JEPA 2-L đóng băng<br/>16 frame, 256 px → cache"]
    F --> T["Attentive probe → sigmoid 'vandalism'<br/>chọn ngưỡng trên dev"]
    T --> Z["zip + install_vandalism.py"]
    Z --> X["extensions/vandalism.py<br/>gate ≥ 0.8 s, chống nhầm strike"]
```

## F.1 Cầm vật sắc nhọn: 3 bước chính + tích hợp

```mermaid
flowchart LR
    subgraph B1["Bước 1 · Thu thập (S1)"]
        direction TB
        I1["Ảnh có nhãn sẵn<br/>Zenodo, Roboflow,<br/>COCO, Open Images"]
        V1["Video tin tức VN<br/>camera an ninh, phóng sự"]
        V2["Video quốc tế<br/>UCF-Crime Robbery/Assault"]
        EX["Trích frame 2 fps<br/>≤ 15 frame/clip, dedup pHash"]
        V1 --> EX
        V2 --> EX
    end

    subgraph B2["Bước 2 · Gán nhãn (S2–S3)"]
        direction TB
        PL["Gán sơ bộ<br/>SAM 3 / Grounding DINO<br/>prompt theo từng loại"]
        HL["Sửa tay trong CVAT<br/>track cho video<br/>thuộc tính held / occluded"]
        QA["QA 10% lần 2<br/>script kiểm lỗi box"]
        SP["Chia train/dev/test<br/>theo vụ việc / clip<br/>giữ clip test nguyên vẹn"]
        PL --> HL --> QA --> SP
    end

    subgraph B3["Bước 3 · Train (S4)"]
        direction TB
        TR["RF-DETR nhiều lớp<br/>Colab Pro"]
        BL["Baseline YOLO26-s<br/>cùng dữ liệu"]
        EV["mAP từng lớp<br/>+ đánh giá theo sự kiện"]
        TR --> EV
        BL --> EV
    end

    subgraph B4["Tích hợp + kiểm thử (S5–S6)"]
        direction TB
        INS["install_sharp_object.py"]
        EXT["extensions/sharp_object.py<br/>liên kết cổ tay + gate"]
        AL["Cảnh báo chung<br/>CẦM VẬT SẮC NHỌN"]
        INS --> EXT --> AL
    end

    I1 --> PL
    EX --> PL
    SP --> TR
    SP --> BL
    EV --> INS
```

## F.2 Nhãn gốc → 7 lớp → 1 cảnh báo

```mermaid
flowchart LR
    subgraph RAW["Nhãn gốc từ nguồn"]
        direction TB
        a1["Roboflow spear<br/>(đã xác minh)"]
        a2["Zenodo machete"]
        a3["COCO knife<br/>OI Knife / Kitchen knife / Dagger"]
        a4["OI Sword<br/>Roboflow sword"]
        a5["Roboflow sickle"]
        a6["OI Axe"]
        a7["Ảnh/frame VN tự gán"]
    end

    subgraph CLS["7 lớp detector (class_map.yaml)"]
        direction TB
        c0["0 · dao_phong_lon"]
        c1["1 · ma_tau"]
        c2["2 · dao_ngan"]
        c3["3 · kiem"]
        c4["4 · liem"]
        c5["5 · riu"]
        c6["6 · vat_sac_khac"]
    end

    ALERT["Cảnh báo duy nhất<br/>CẦM VẬT SẮC NHỌN<br/>event ghi weapon_class"]

    a1 --> c0
    a2 --> c1
    a3 --> c2
    a4 --> c3
    a5 --> c4
    a6 --> c5
    a7 --> c0
    a7 --> c1
    a7 --> c6
    c0 --> ALERT
    c1 --> ALERT
    c2 --> ALERT
    c3 --> ALERT
    c4 --> ALERT
    c5 --> ALERT
    c6 --> ALERT
```

## F.3 Logic cảnh báo vật sắc nhọn lúc chạy

```mermaid
flowchart LR
    FR[("frames[-1]")] --> DT["RF-DETR<br/>box vật + lớp + điểm"]
    CORE[("reply core<br/>box người + pose")] --> AS
    DT --> TH{"Điểm ≥ ngưỡng<br/>của lớp?"}
    TH -- "không" --> DROP["Bỏ qua"]
    TH -- "có" --> AS{"Gần cổ tay kp 9/10<br/>hoặc trong box người<br/>nới 15%?"}
    AS -- "không" --> FLOOR["Vật không được cầm<br/>không báo"]
    AS -- "có" --> TRK["Gắn vào track người"]
    TRK --> GT{"≥ 2 / 3 lần<br/>quan sát liên tiếp?"}
    GT -- "chưa" --> WAIT["Chờ bước sau"]
    GT -- "đủ" --> ALERT["CẦM VẬT SẮC NHỌN<br/>event: weapon_class, score, track_id"]
    ALERT --> REL["Nhả sau 2 s<br/>không còn thấy"]
```

## I.1 Lộ trình thực hiện cả 2 hướng (chiều ngang)

```mermaid
flowchart LR
    START(["Bắt đầu<br/>(local)"]) --> G0
    START --> S0

    subgraph P0["Giai đoạn 0 · Kiến trúc (local)"]
        direction LR
        G0["0.1 Golden<br/>baseline"] --> G1["0.2 Registry"] --> G2["0.3 Extension<br/>worker + env"] --> G3["0.4 Backend<br/>_ext"] --> G4["0.5 Test cô lập<br/>VRAM"] --> G5["0.6 Tài liệu"]
    end

    G5 --> M0{{"M0<br/>core khớp<br/>golden 100%"}}

    subgraph PV["Hướng 1 · Phá hoại tài sản"]
        direction LR
        V0["V0 Audit<br/>dữ liệu"] --> V1["V1 Baseline<br/>AVA"] --> V2["V2 prepare"] --> V3["V3 Train<br/>Colab Pro"] --> V4["V4 Đóng gói<br/>install"] --> V5["V5 Tích hợp<br/>_ext"] --> V6["V6 Kiểm thử"]
    end

    subgraph PSD["Hướng 2 · Dữ liệu vật sắc nhọn (song song, không đụng code)"]
        direction LR
        S0["S0 Taxonomy<br/>giấy phép"] --> S1["S1 Thu thập<br/>video ≥ 20%"] --> S2["S2 Gán nhãn<br/>từng loại + QA"] --> S3["S3 Đóng gói<br/>dataset"]
    end

    subgraph PST["Hướng 2 · Train + tích hợp"]
        direction LR
        S4["S4 Train RF-DETR<br/>Colab Pro"] --> S5["S5 Install +<br/>extension"] --> S6["S6 Kiểm thử"]
    end

    M0 --> V0
    V6 --> M1{{"M1<br/>VANDALISM<br/>ổn trong _ext"}}
    S3 --> S4
    M1 --> S5
    S6 --> M2{{"M2<br/>SHARP OBJECT<br/>ổn trong _ext"}}

    subgraph PZ["Tổng kiểm"]
        direction LR
        Z1["Z1 Chạy cả 2<br/>extension"] --> Z2["Z2 Hồi quy<br/>toàn bộ"] --> Z3["Z3 Kịch bản<br/>demo"] --> Z4["Z4 Duyệt, đổi<br/>backend mặc định"]
    end

    M2 --> Z1
    Z4 --> DONE(["Push GitHub<br/>1 lần"])

    classDef colab stroke:#d97706,stroke-width:3px
    classDef gate stroke:#16a34a,stroke-width:3px
    class V3,S4 colab
    class M0,M1,M2 gate
```
