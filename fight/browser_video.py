"""Finalize generated upload results as browser-compatible H.264 video."""

import os
from pathlib import Path
import subprocess
import time
import uuid


def finalize_browser_video(path, stop_event, timeout=180):
    import imageio_ffmpeg
    path = Path(path)
    temporary = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.h264.mp4')
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-v', 'error', '-i', str(path),
               '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
               '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(temporary)]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if stop_event.is_set():
                raise InterruptedError('Video encoding stopped')
            if time.monotonic() > deadline:
                raise RuntimeError('Browser video encoding timed out')
            try:
                _, error = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode or not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError('Browser video encoding failed: ' + error.decode('utf-8', errors='replace')[-1000:])
        temporary.replace(path)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        temporary.unlink(missing_ok=True)
