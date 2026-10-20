"""
core/config.py — Central configuration for the anomaly detection system.
All tuneable parameters in one place.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────────────────────
    models_dir: Path = BASE_DIR / "models"
    logs_dir: Path = BASE_DIR / "logs"
    anomaly_model_path: Path = BASE_DIR / "models" / "lstm_autoencoder.pt"
    event_log_path: Path = BASE_DIR / "logs" / "events.jsonl"

    # ── Detection (YOLOv8) ─────────────────────────────────────────────────
    yolo_model: str = "yolov8n.pt"           # nano for speed; swap to yolov8s.pt for accuracy
    yolo_conf_threshold: float = 0.40
    yolo_iou_threshold: float = 0.45
    yolo_device: str = "cpu"                 # "cuda" if GPU available
    yolo_imgsz: int = 640
    person_class_id: int = 0                 # COCO class 0 = person

    # ── Tracking (DeepSORT) ────────────────────────────────────────────────
    deepsort_max_age: int = 30               # frames before track is deleted
    deepsort_n_init: int = 3                 # frames needed to confirm track
    deepsort_max_cosine_distance: float = 0.3
    deepsort_nms_max_overlap: float = 1.0
    deepsort_max_iou_distance: float = 0.7

    # ── Anomaly Detection ──────────────────────────────────────────────────
    seq_len: int = 30                        # frames per sequence
    feature_dim: int = 9                     # feature vector size per frame
    lstm_hidden: int = 64
    lstm_layers: int = 2
    anomaly_threshold: float = 0.045        # reconstruction MSE threshold
    min_track_frames: int = 15              # ignore tracks shorter than this

    # Rule-based thresholds (used as fallback + classification hints)
    speed_run_threshold: float = 0.025      # normalised speed (frac of frame diag)
    loiter_frames: int = 90                 # frames in same zone to flag loitering
    loiter_displacement: float = 0.08       # max displacement for loiter check
    fight_proximity: float = 0.15          # normalised distance to consider "close"
    fight_speed_threshold: float = 0.018

    # ── API ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    stream_fps_cap: int = 25               # max FPS sent over MJPEG stream
    jpeg_quality: int = 85

    # ── Visualisation ──────────────────────────────────────────────────────
    normal_color: tuple = (0, 220, 0)       # BGR green
    anomaly_color: tuple = (0, 0, 255)      # BGR red
    text_color: tuple = (255, 255, 255)
    box_thickness: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere
settings = Settings()

# Ensure dirs exist
settings.models_dir.mkdir(exist_ok=True)
settings.logs_dir.mkdir(exist_ok=True)
