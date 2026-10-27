import json
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from core.config import BASE_DIR, settings
from fight.runtime import FightRuntime

LABELS = {
    "normal": "NORMAL",
    "fighting": "FIGHT DETECTED",
    "verifying": "VERIFYING",
    "warming_up": "COLLECTING CLIP",
    "unavailable": "MODEL NOT READY",
    "insufficient": "CHECK CAMERA VIEW",
}


def annotate(frame, result):
    image = frame.copy()
    color = (
        (50, 65, 235)
        if result["state"] == "fighting"
        else (
            (40, 195, 215)
            if result["state"] in ("verifying", "insufficient")
            else (175, 195, 90)
        )
    )
    for region in result.get("regions", []):
        left, top, right, bottom = map(int, region)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 36), (25, 30, 28), -1)
    cv2.putText(
        image,
        LABELS.get(result["state"], result["state"]),
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    return image


class FightProcessor:
    def __init__(self):
        self.runtime = FightRuntime(settings.models_dir, settings.fight_allow_pilot)
        self.thread = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.lifecycle = threading.Lock()
        self.jpeg = None
        self.revision = 0
        self.events = []
        self.phase = "idle"
        self.error = None
        self.source = None
        self.replay = False
        self.session_started = None
        self.dropped_samples = 0
        self.frame_count = self.total_frames = 0
        self.timestamp = self.latency = 0.0
        self.output = None
        self.active_event = None
        self.event_frames = []
        self.before = deque(maxlen=24)
        self.evidence_dir = BASE_DIR / "uploads" / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=8)

    def start(self, source, allow_pilot=False, preview=False, replay=False):
        with self.lifecycle:
            self.stop()
            if self.is_running():
                raise RuntimeError("Nguồn trước chưa dừng; vui lòng thử lại.")
            self.runtime = FightRuntime(settings.models_dir, allow_pilot)
            if not self.runtime.model_ready and not preview:
                raise ValueError(self.runtime.message)
            if not preview:
                self.runtime.warmup()
            self.stop_event.clear()
            self.preview = preview
            self.replay = replay
            self.source, self.phase, self.error = source, "opening", None
            self.frame_count = self.total_frames = 0
            self.timestamp = self.latency = 0.0
            self.session_started = time.monotonic()
            self.dropped_samples = 0
            self.output = self.jpeg = None
            self.before.clear()
            self.thread = threading.Thread(target=self.run, args=(source,), daemon=True)
            self.thread.start()

    def status(self):
        prediction = dict(self.runtime.result)
        live_clock = self.replay or not self.total_frames
        current_time = (
            time.monotonic() - self.session_started
            if live_clock and self.session_started is not None and self.is_running()
            else self.timestamp
        )
        observed_at = prediction.get("prediction_timestamp")
        age = max(0.0, current_time - observed_at) if observed_at is not None else None
        stale = (
            age is not None
            and live_clock
            and self.is_running()
            and age > self.runtime.gate.max_observation_gap
        )
        if stale:
            prediction.update(
                state="insufficient",
                event=None,
                message="Kết quả suy luận đã cũ; CPU hoặc nguồn hình đang chậm, chưa xác nhận hành vi hiện tại.",
            )
        prediction["score_stale"] = stale
        return {
            "running": self.is_running(),
            "phase": self.phase,
            "source": str(self.source) if self.source is not None else None,
            "source_kind": (
                "replay"
                if self.replay
                else ("video" if self.total_frames else "camera")
            ),
            "frame_count": self.frame_count,
            "total_frames": self.total_frames,
            "timestamp": self.timestamp,
            "latency_seconds": round(
                max(self.latency, current_time - self.timestamp), 3
            ),
            "prediction_age_seconds": round(age, 3) if age is not None else None,
            "dropped_samples": self.dropped_samples,
            "model_loaded": self.runtime.model_ready,
            "model": {
                "architecture": self.runtime.metadata.get("architecture"),
                "input_kind": self.runtime.input_kind,
                "approved_for_demo": self.runtime.metadata.get(
                    "approved_for_demo", False
                ),
                "message": self.runtime.message,
                "clip_test": self.runtime.metadata.get("test"),
                "hard_case_test": self.runtime.metadata.get("test_by_dataset", {}).get(
                    "airtlab"
                ),
                "real_phone_tested": self.runtime.metadata.get(
                    "real_phone_tested", False
                ),
            },
            "prediction": prediction,
            "error": self.error,
            "event_count": len(self.events),
            "output": self.output,
        }

    def frames(self):
        revision = -1
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.revision != revision or not self.is_running(),
                    timeout=5,
                )
                jpeg, revision = self.jpeg, self.revision
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            if not self.is_running():
                break

    def record(self, frame, result):
        self.before.append((self.timestamp, frame.copy()))
        if result.get("event") == "started" and self.active_event is None:
            identifier = uuid.uuid4().hex
            self.active_event = {
                "id": identifier,
                "type": "fighting",
                "started_at": self.timestamp,
                "ended_at": None,
                "score": result["score"],
                "snapshot": f"/api/evidence/{identifier}.jpg",
                "clip": None,
            }
            self.events.append(self.active_event)
            self.event_frames = list(self.before)
            cv2.imwrite(
                str(self.evidence_dir / f"{identifier}.jpg"), annotate(frame, result)
            )
        elif self.active_event is not None:
            self.event_frames.append((self.timestamp, frame.copy()))
        if self.active_event is not None:
            self.active_event["score"] = max(
                self.active_event["score"], result.get("score") or 0
            )
            if (
                result.get("event") == "ended"
                or self.timestamp - self.event_frames[0][0] >= 30
            ):
                self.finish_event()

    def finish_event(self):
        if self.active_event is None:
            return
        event = self.active_event
        event["ended_at"] = self.timestamp
        height, width = self.event_frames[0][1].shape[:2]
        event["clip_started_at"] = self.event_frames[0][0]
        event["clip_ended_at"] = self.event_frames[-1][0]
        event["source_samples"] = len(self.event_frames)
        event["max_sample_gap_seconds"] = max(
            (
                current[0] - previous[0]
                for previous, current in zip(self.event_frames, self.event_frames[1:])
            ),
            default=0.0,
        )
        path = self.evidence_dir / f"{event['id']}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.runtime.config.fps,
            (width, height),
        )
        if writer.isOpened():
            sample_index = 0
            total = (
                round(
                    (self.event_frames[-1][0] - self.event_frames[0][0])
                    * self.runtime.config.fps
                )
                + 1
            )
            for frame_index in range(min(total, round(self.runtime.config.fps * 35))):
                target_time = (
                    self.event_frames[0][0] + frame_index / self.runtime.config.fps
                )
                while (
                    sample_index + 1 < len(self.event_frames)
                    and self.event_frames[sample_index + 1][0] <= target_time + 1e-6
                ):
                    sample_index += 1
                writer.write(
                    cv2.resize(self.event_frames[sample_index][1], (width, height))
                )
            event["clip"] = f"/api/evidence/{path.name}"
        writer.release()
        (self.evidence_dir / f"{event['id']}.json").write_text(
            json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.active_event, self.event_frames = None, []

    def run(self, source):
        is_file = isinstance(source, str) and Path(source).is_file()
        live_mode = not is_file or self.replay
        capture = cv2.VideoCapture(source)
        writer = reader = None
        pending = deque(maxlen=self.runtime.config.frames * 2)
        live_lock = threading.Lock()
        latest = {"frame": None, "ended": False, "count": 0}
        started = self.session_started or time.monotonic()
        try:
            if not capture.isOpened():
                raise ValueError("Không mở được nguồn hình. Kiểm tra camera hoặc URL.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            fps = fps if 0 < fps <= 240 else 25.0
            self.total_frames = (
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
            )
            self.phase = "processing"
            output_path = self.evidence_dir / f"{uuid.uuid4().hex}.mp4"
            if live_mode:

                def read_live():
                    next_sample = 0.0
                    source_index = 0
                    while not self.stop_event.is_set():
                        ok, frame = capture.read()
                        if not ok:
                            break
                        timestamp = (
                            source_index / fps
                            if is_file
                            else time.monotonic() - started
                        )
                        source_index += 1
                        if is_file and self.stop_event.wait(
                            max(0.0, started + timestamp - time.monotonic())
                        ):
                            break
                        with live_lock:
                            latest.update(frame=frame, count=latest["count"] + 1)
                            if timestamp + 1e-6 >= next_sample:
                                if len(pending) == pending.maxlen:
                                    self.dropped_samples += 1
                                pending.append((frame, timestamp))
                                while next_sample <= timestamp + 1e-6:
                                    next_sample += 1 / self.runtime.config.fps
                    with live_lock:
                        latest["ended"] = True

                reader = threading.Thread(target=read_live, daemon=True)
                reader.start()
            last_evidence = -1.0
            while not self.stop_event.is_set():
                if not live_mode:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    timestamp = self.frame_count / fps
                    samples = [(frame, timestamp)]
                    self.frame_count += 1
                else:
                    with live_lock:
                        frame, samples, ended = (
                            latest["frame"],
                            list(pending),
                            latest["ended"],
                        )
                        pending.clear()
                        self.frame_count = latest["count"]
                    if not samples:
                        if ended:
                            if is_file:
                                break
                            raise ValueError("Camera mất kết nối.")
                        self.stop_event.wait(0.02)
                        continue
                    timestamp = samples[-1][1]
                for index, (sample, sample_time) in enumerate(samples):
                    result = (
                        {
                            "state": "unavailable",
                            "score": None,
                            "regions": [],
                            "people": 0,
                            "event": None,
                            "message": "Chỉ xem trước camera",
                        }
                        if self.preview
                        else self.runtime.push(
                            sample, sample_time, predict=index == len(samples) - 1
                        )
                    )
                    self.timestamp = sample_time
                    if (
                        sample_time - last_evidence + 1e-6
                        >= 1 / self.runtime.config.fps
                        or result.get("event")
                    ):
                        self.record(sample, result)
                        last_evidence = sample_time
                if self.preview:
                    self.runtime.result = result
                self.timestamp = timestamp
                self.latency = (
                    max(0.0, time.monotonic() - started - timestamp)
                    if live_mode
                    else 0.0
                )
                image = annotate(frame, result)
                if not live_mode:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (image.shape[1], image.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("Không tạo được video kết quả.")
                    writer.write(image)
                ok, encoded = cv2.imencode(".jpg", image)
                if ok:
                    with self.condition:
                        self.jpeg = encoded.tobytes()
                        self.revision += 1
                        self.condition.notify_all()
            self.phase = "stopped" if self.stop_event.is_set() else "completed"
            if is_file and writer is not None:
                self.output = f"/api/evidence/{output_path.name}"
        except Exception as error:
            self.phase, self.error = "error", str(error)
        finally:
            self.stop_event.set()
            capture.release()
            if reader:
                reader.join(timeout=2)
            if writer:
                writer.release()
            self.finish_event()
            with self.condition:
                self.condition.notify_all()


app = FastAPI(title="FightWatch", version="2.0.0")
processor = FightProcessor()


@app.get("/", response_class=HTMLResponse)
def frontend():
    return HTMLResponse(
        (BASE_DIR / "frontend" / "fight.html").read_text(encoding="utf-8")
    )


@app.get("/api/status")
def status():
    return processor.status()


@app.get("/api/events")
def events():
    return {"events": [dict(event) for event in processor.events[-100:]]}


def start_source(source, allow_pilot, preview=False, replay=False):
    try:
        if replay:
            processor.start(source, allow_pilot, preview, replay=True)
        else:
            processor.start(source, allow_pilot, preview)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error
    return {"status": "started"}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...), allow_pilot: bool = False, replay: bool = False
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(400, "Chọn một file video hợp lệ.")
    destination = BASE_DIR / "uploads" / f"{uuid.uuid4().hex}{suffix}"
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            handle.write(chunk)
    return start_source(str(destination), allow_pilot, replay=replay)


@app.post("/api/stream/camera")
def camera(index: int = 0, allow_pilot: bool = False, preview: bool = False):
    if not 0 <= index <= 20:
        raise HTTPException(400, "Camera index không hợp lệ.")
    return start_source(index, allow_pilot, preview)


@app.get("/api/cameras")
def cameras():
    found = []
    for index in range(6):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            found.append({"index": index, "label": f"Camera {index}"})
        capture.release()
    return {"cameras": found}


@app.post("/api/stream/url")
def stream_url(url: str, allow_pilot: bool = False):
    if not url.lower().startswith(("rtsp://", "http://", "https://")):
        raise HTTPException(400, "Cần URL RTSP hoặc HTTP(S).")
    return start_source(url, allow_pilot)


@app.post("/api/stop")
def stop():
    processor.stop()
    return {"status": "stopped"}


@app.get("/api/stream")
def stream():
    return StreamingResponse(
        processor.frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/report")
def report():
    return {
        "system": "FightWatch",
        "status": processor.status(),
        "events": processor.events,
    }


@app.get("/api/evidence/{name}")
def evidence(name: str):
    if Path(name).name != name or Path(name).suffix not in {".jpg", ".mp4", ".json"}:
        raise HTTPException(404)
    path = processor.evidence_dir / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)
