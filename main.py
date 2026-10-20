"""
main.py — CLI entry point.

Modes:
  run     : process a video file or camera, display/save annotated output
  serve   : launch the FastAPI server + frontend dashboard
  train   : train the LSTM Autoencoder (delegates to anomaly.train)

Usage:
  python main.py run   --source video.mp4 [--save output.mp4] [--no-display]
  python main.py run   --source 0                         # webcam
  python main.py serve [--host 0.0.0.0] [--port 8000]
  python main.py train [--synthetic] [--source normal.mp4] [--epochs 30]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_source(src: str):
    """Return int camera index or string path."""
    try:
        return int(src)
    except ValueError:
        return src


# ── Mode: run ─────────────────────────────────────────────────────────────────

def run(args):
    from core.config import settings
    from core.logger import event_logger
    from core.utils import draw_bbox, draw_overlay_stats, encode_jpeg
    from detection.detector import PersonDetector
    from tracking.tracker import PersonTracker
    from anomaly.detector import AnomalyDetector
    from core.utils import draw_bbox

    ANOMALY_LABELS = {
        "running":   "RUNNING",
        "fighting":  "FIGHT",
        "loitering": "LOITER",
        "unknown":   "ANOMALY",
    }

    source = parse_source(args.source)
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[run] Source: {source}  {width}×{height} @ {fps:.1f}fps")

    detector = PersonDetector()
    tracker  = PersonTracker()
    anomaly  = AnomalyDetector()

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"[run] Saving annotated video to: {args.save}")

    frame_no    = 0
    alert_count = 0
    t_start     = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            h, w = frame.shape[:2]

            dets   = detector.detect(frame)
            tracks = tracker.update(dets, frame)
            events = anomaly.update(tracks, frame_hw=(h, w), frame_number=frame_no)

            alert_count += len(events)
            for ev in events:
                event_logger.log(ev)
                print(f"  [ALERT] frame={frame_no} track={ev.track_id} "
                      f"type={ev.anomaly_type} conf={ev.confidence:.2f}")

            # Annotate
            anomaly_ids = {e.track_id for e in events}
            for t in tracks:
                color = settings.anomaly_color if t.track_id in anomaly_ids else settings.normal_color
                lbl   = f"#{t.track_id}"
                if t.is_anomaly:
                    lbl += f" {ANOMALY_LABELS.get(t.anomaly_type, '!')}"
                draw_bbox(frame, t.bbox_xyxy, lbl, color)

            if events:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), settings.anomaly_color, 5)

            draw_overlay_stats(frame, {
                "Frame":  frame_no,
                "Tracks": len(tracks),
                "Alerts": alert_count,
            })

            if writer:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("SENTINEL — Anomaly Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - t_start
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        event_logger.close()

        print(f"\n[run] Finished.")
        print(f"  Frames processed : {frame_no}")
        print(f"  Total alerts     : {alert_count}")
        print(f"  Elapsed          : {elapsed:.1f}s")
        print(f"  Event log        : {event_logger._path}")


# ── Mode: serve ───────────────────────────────────────────────────────────────

def serve(args):
    import uvicorn
    from api.server import app

    print(f"[serve] Starting SENTINEL API at http://{args.host}:{args.port}")
    print(f"[serve] Dashboard  → http://localhost:{args.port}/")
    print(f"[serve] API docs   → http://localhost:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


# ── Mode: train ───────────────────────────────────────────────────────────────

def train_model(args):
    import anomaly.train as tr

    if args.synthetic or not args.source:
        print("[train] Using synthetic normal-behaviour dataset …")
        sequences = tr.generate_synthetic_normal(n_samples=args.samples)
    else:
        sequences = tr.collect_sequences_from_video(args.source)

    model = tr.train(sequences, epochs=args.epochs)

    from core.config import settings
    model.save(settings.anomaly_model_path)
    print(f"[train] Saved → {settings.anomaly_model_path}")


# ── CLI parser ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="SENTINEL — Real-time Video Anomaly Detection System",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Process a video file or webcam")
    p_run.add_argument("--source",     required=True, help="Video path or camera index")
    p_run.add_argument("--save",       default=None,  help="Save annotated video here")
    p_run.add_argument("--no-display", action="store_true", help="Headless mode (no window)")

    # --- serve ---
    p_srv = sub.add_parser("serve", help="Launch FastAPI server + dashboard")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)

    # --- train ---
    p_tr = sub.add_parser("train", help="Train the LSTM Autoencoder")
    p_tr.add_argument("--source",    default=None,  help="Normal-behaviour video")
    p_tr.add_argument("--synthetic", action="store_true")
    p_tr.add_argument("--epochs",    type=int, default=30)
    p_tr.add_argument("--samples",   type=int, default=2000)

    args = parser.parse_args()

    if args.mode == "run":
        run(args)
    elif args.mode == "serve":
        serve(args)
    elif args.mode == "train":
        train_model(args)


if __name__ == "__main__":
    main()
