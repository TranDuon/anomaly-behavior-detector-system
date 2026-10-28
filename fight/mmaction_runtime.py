"""Use the isolated MMAction2 GPU environment through bounded local IPC."""

from collections import deque
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

import numpy as np

from fight.runtime import EventGate
from fight.vision import ClipConfig

ROOT = Path(__file__).resolve().parents[1] / "mmaction2_demo"


class MMActionRuntime:
    def __init__(self, tuned=False, allow_pilot=False):
        self.backend_name = "mmaction_tuned" if tuned else "mmaction_original"
        self.device = "cuda:0"
        self.tuned = tuned
        self.config = ClipConfig(
            frames=9, fps=3.75, size=256, stride_seconds=16 / 30, region_mode="full"
        )
        self.cpu_threads, self.cpu_optimization, self.batch_detection = (
            4,
            "isolated CUDA worker",
            False,
        )
        self.pose_detector = None
        self.process = self.log = None
        self.replies = queue.Queue(maxsize=4)
        self.metadata = {
            "architecture": "mmaction_slowonly_r101",
            "deployment": {
                "display_name": (
                    "MMAction2 · nhánh tương tác fine-tuned"
                    if tuned
                    else "MMAction2 · checkpoint gốc · RGB chuẩn"
                )
            },
            "approved_for_demo": False,
            "real_phone_tested": False,
        }
        self.input_kind = "rgb_actions"
        self.pilot_required = not allow_pilot
        available = (ROOT / "env/python.exe").is_file() and (
            ROOT / "results/install_verified.json"
        ).is_file()
        if tuned:
            available = (
                available
                and (ROOT / "weights/interaction.json").is_file()
                and (ROOT / "weights/interaction.pt").is_file()
            )
        self.model_ready = available and allow_pilot
        self.message = (
            "Model thử nghiệm; giữ nhãn AVA gốc, không phải độ chính xác camera."
            if self.model_ready
            else (
                "Bật Cho phép checkpoint pilot."
                if self.pilot_required
                else "Chưa có nhánh fine-tuned hoặc môi trường MMAction2 chưa sẵn sàng."
            )
        )
        self.gate = EventGate(0.4, 0.28, confirm_seconds=0.4)
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=9)
        self.last_input = self.next_sample = self.last_prediction = None
        self.gate = self.gate.fresh()
        self.result = {
            "state": "warming_up" if self.model_ready else "unavailable",
            "score": None,
            "regions": [],
            "person_boxes": [],
            "visible_people": None,
            "people": 0,
            "event": None,
            "actions": [],
            "message": self.message,
            "prediction_timestamp": None,
            "inference_seconds": None,
            "view_quality": None,
        }

    def warmup(self):
        if not self.model_ready or self.process is not None:
            return
        (ROOT / "results/ipc").mkdir(parents=True, exist_ok=True)
        self.log = (
            ROOT / "results" / ("web_worker_" + uuid.uuid4().hex + ".log")
        ).open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(ROOT / "repo"),
            CUDA_MODULE_LOADING="LAZY",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            PYTHONUNBUFFERED="1",
            TORCH_HOME=str(ROOT / "weights/torch-cache"),
        )
        command = [str(ROOT / "env/python.exe"), str(ROOT / "worker.py")]
        if self.tuned:
            command.append("--tuned")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            env=environment,
            cwd=ROOT,
        )

        def receive():
            for line in self.process.stdout:
                try:
                    self.replies.put(json.loads(line), timeout=2)
                except (ValueError, queue.Full):
                    break

        threading.Thread(target=receive, daemon=True).start()
        try:
            response = self.replies.get(timeout=90)
            if not response.get("ready"):
                raise RuntimeError(str(response))
            self.metadata.update(response["metadata"])
            threshold = self.metadata["threshold"]
            self.gate = EventGate(
                threshold,
                threshold * 0.7,
                confirm_seconds=self.metadata["confirm_seconds"],
            )
        except (queue.Empty, RuntimeError) as error:
            self.close()
            raise RuntimeError(
                "Không nạp được MMAction2; xem mmaction2_demo/results/web_worker_*.log"
            ) from error

    def push(self, frame, timestamp, predict=True, *, sampled=False):
        if not math.isfinite(timestamp) or (
            self.last_input is not None and timestamp < self.last_input
        ):
            raise ValueError("Finite monotonic timestamps required")
        self.result = {**self.result, "event": None}
        if not self.model_ready:
            return dict(self.result)
        if self.last_input is not None and timestamp - self.last_input > 1:
            self.reset()
        self.last_input = timestamp
        if self.next_sample is None:
            self.next_sample = timestamp
        if not sampled and timestamp + 1e-6 < self.next_sample:
            return dict(self.result)
        while self.next_sample <= timestamp + 1e-6:
            self.next_sample += 1 / self.config.fps
        self.samples.append(frame.copy())
        if (
            len(self.samples) < 9
            or not predict
            or (
                self.last_prediction is not None
                and timestamp - self.last_prediction < self.config.stride_seconds - 1e-6
            )
        ):
            return dict(self.result)
        self.warmup()
        started = time.perf_counter()
        path = ROOT / "results/ipc" / (uuid.uuid4().hex + ".npz")
        try:
            np.savez(path, frames=np.stack(self.samples))
            self.process.stdin.write(json.dumps({"frames": str(path)}) + "\n")
            self.process.stdin.flush()
            response = self.replies.get(timeout=45)
            if "error" in response:
                raise RuntimeError(response["error"])
            output = response["result"]
        except (queue.Empty, BrokenPipeError) as error:
            raise RuntimeError("MMAction2 bị gián đoạn; xem log worker.") from error
        finally:
            path.unlink(missing_ok=True)
        self.last_prediction = timestamp
        state, event = self.gate.update(output["score"], timestamp)
        if not output["people"]:
            state, event = "insufficient", "ended" if self.gate.active else None
            self.gate = self.gate.fresh()
        self.result = {
            **self.result,
            **output,
            "state": state,
            "event": event,
            "prediction_timestamp": timestamp,
            "visible_people": output["people"],
            "inference_seconds": time.perf_counter() - started,
            "message": self.message,
        }
        return dict(self.result)

    def close(self):
        if self.process:
            try:
                if self.process.poll() is None:
                    self.process.stdin.write('{"op":"close"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
            finally:
                self.process = None
        if self.log:
            self.log.close()
            self.log = None
