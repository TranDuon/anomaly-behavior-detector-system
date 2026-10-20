"""
api/server.py — FastAPI application.

Endpoints:
  POST /api/upload          → upload a video file
  POST /api/stream/camera   → start webcam (index param)
  POST /api/stop            → stop current processing
  GET  /api/stream          → MJPEG video stream
  GET  /api/events          → recent event log (JSON)
  GET  /api/status          → system status
  WS   /ws/alerts           → real-time WebSocket alert push

Frontend:
  GET  /                    → serves frontend/index.html
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect,
    HTTPException, Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse, StreamingResponse, JSONResponse, FileResponse
)
import aiofiles

from core.config import settings
from core.logger import event_logger
from api.processor import VideoProcessor

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Video Anomaly Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"

# ── WebSocket connection manager ─────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()

# ── Processor singleton ───────────────────────────────────────────────────────

def _alert_cb(event_dict: dict):
    """Sync callback that schedules a coroutine on the event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(event_dict), loop)
    except Exception:
        pass


processor = VideoProcessor(alert_callback=_alert_cb)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if FRONTEND_PATH.exists():
        async with aiofiles.open(FRONTEND_PATH, "r") as f:
            return HTMLResponse(await f.read())
    return HTMLResponse("<h1>Frontend not found</h1><p>Run from project root.</p>")


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and start processing it."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Only video files are accepted.")

    dest = UPLOAD_DIR / file.filename
    async with aiofiles.open(dest, "wb") as out:
        content = await file.read()
        await out.write(content)

    processor.start(str(dest))
    return {"status": "started", "file": file.filename, "path": str(dest)}


@app.post("/api/stream/camera")
async def start_camera(index: int = Query(default=0, description="Camera device index")):
    """Start processing from a webcam."""
    processor.start(index)
    return {"status": "started", "camera": index}


@app.post("/api/stream/url")
async def start_stream_url(url: str = Query(..., description="RTSP / HTTP stream URL")):
    """Start processing from an RTSP or HTTP video stream."""
    processor.start(url)
    return {"status": "started", "url": url}


@app.post("/api/stop")
async def stop_processing():
    processor.stop()
    return {"status": "stopped"}


@app.get("/api/stream")
async def mjpeg_stream():
    """MJPEG video stream — embed in <img src='/api/stream'>."""

    def generate():
        for jpeg_bytes in processor.iter_frames():
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes +
                b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/events")
async def get_events(limit: int = Query(default=100, ge=1, le=500)):
    """Return the most recent N anomaly events."""
    return JSONResponse(content={"events": event_logger.get_recent(limit)})


@app.get("/api/status")
async def get_status():
    return JSONResponse(content=processor.status())


@app.delete("/api/events")
async def clear_events():
    event_logger.clear()
    return {"status": "cleared"}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    Real-time alert push.
    Client receives JSON AnomalyEvent dicts as they occur.
    Sends ping/pong every 20 s to keep the connection alive.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            try:
                # Receive any message (heartbeat ping from client)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=20)
            except asyncio.TimeoutError:
                # Send server ping
                await websocket.send_json({"type": "ping", "ts": time.time()})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
