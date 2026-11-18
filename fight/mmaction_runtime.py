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

from fight import behaviors as behavior_registry
from fight.runtime import EventGate
from fight.vision import ClipConfig
from mmaction2_demo.frame_transport import SharedFrameBuffer

ROOT = Path(__file__).resolve().parents[1] / "mmaction2_demo"
# Extensions enabled by the "_ext" backend: every extension whose checkpoint is installed
# under weights/extensions/<name>/<name>.pt (vandalism from V4, sharp_object from S5).
# Tests pass extension_names explicitly.
EXTENSION_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "extensions"
KNOWN_EXTENSIONS = ("vandalism", "sharp_object", "handgun")
EXTENSION_NAMES = tuple(name for name in KNOWN_EXTENSIONS  # detector .pt or a spec (hand-crop installs)
                        if any((EXTENSION_WEIGHTS / name / f).is_file() for f in (f"{name}.pt", f"{name}_spec.json")))
# Longest input gap that keeps the collected clip (realtime drops under GPU load).
MAX_CONTINUOUS_GAP_SECONDS = 3.0

# Submit one extension step every N core samples; slower when the core step is slow.
# Every step (cadence 1) shared the GPU with the core so much that a core step took
# ~0.26 s of the 0.267 s budget and the live view lagged. Every second step halves the
# load with equivalent vandalism results offline (Kinetics 80 %, ASHEN 67 %, FP 7 %).
EXTENSION_CADENCE = 2
EXTENSION_SLOW_CADENCE = 3
# 1: between full extension steps, run only these light extensions on the newest sample, so their
# boxes refresh every core step (sharp_object ~55 ms on the GPU; vandalism stays on the cadence).
EVERY_STEP_EXTENSIONS = ('sharp_object', 'handgun')
# Hand-crop weapon extensions: full-detail frames, weapon boxes in the overlay, holder-only highlight.
WEAPON_EXTENSIONS = behavior_registry.WEAPON_BEHAVIORS
# Frames sent to the extension worker. The vandalism model resizes every frame to 256x256
# (no crop) anyway, so sending 256x256 gives the same input with ~14x less copying.
EXTENSION_FRAME_SIZE = (256, 256)
# The sharp-object detector needs detail (blades are thin): when it is enabled the frames keep
# their aspect ratio with the long side capped here; vandalism resizes to 256x256 itself.
SHARP_OBJECT_MAX_SIDE = 960
# Time span of the extension input (last 8 of the 9 core samples at 3.75 fps).
EXTENSION_WINDOW_SECONDS = 7 / 3.75
# Per-extension confirmation tuned offline (extensions/vandalism/gate_benchmark.py, 417 test clips):
# mean of the last 3 scores + 0.5 s confirmation (with cadence 2: Kinetics 72% -> 80%,
# ASHEN 56% -> 67%, normal CCTV false alarms 10% -> 7%).
# M1: below this motion (mean absolute grey-level change, 0-255, over ~1 s inside the people
# boxes) nobody is moving, so a vandalism score does not count (and the alarm may end).
# M2: the alarm ends as soon as the mean of the last release_window gated scores is below
# the release threshold (0.7 x threshold), instead of needing 1 s of uninterrupted low scores.
# probe_02_ft (17/11/2026): scores are ~0 or ~1, so averaging 3 scores hid single hits. Raw score +
# 0.5 s confirmation on 299 Kinetics val clips: recall 63% -> 73%, hard-negative FP 3.9% -> 6.9%.
# 18/11/2026: hold the alert 0.6 s (was the spec's 1.0 s) after scores drop, so it clears sooner
# once the striking stops; confirmation is unchanged, so this adds no new alerts.
EXTENSION_GATE_OVERRIDES = {'vandalism': {'confirm_seconds': 0.5, 'smooth_window': 1,
                                          'min_motion': 4.0, 'release_window': 4, 'release_seconds': 0.6,
                                          # 18/11: under GPU load vandalism replies arrive every ~1.6 s, past the
                                          # default 1.5 s gap, so the gate restarted between two high scores
                                          # and never confirmed (yellow/green flicker). Vandalism only.
                                          'max_observation_gap': 2.5},
                            # B: alert on at the spec threshold, off only below 0.5x it for 1.2 s
                            'sharp_object': {'release_ratio': 0.5, 'release_seconds': 1.2,
                                             # C: the gate sees the 3rd-highest of the last 4 scores, so an alert
                                             # needs >= 3 of 4 recent steps above threshold (flickering objects fail)
                                             'k_of_n': (3, 4)},
                            # Handgun: same hand-crop pipeline, same steadying rules as sharp_object.
                            'handgun': {'release_ratio': 0.5, 'release_seconds': 1.2, 'k_of_n': (3, 4)}}
# A/C: sharp-object overlay. Keep the last weapon / holder boxes this long when a step has none
# (alert still on), and blend a new weapon box with the previous one (weight of the old box).
SHARP_HOLD_SECONDS = 1.0
# Vandalism overlay: while the alert is on, keep the last person boxes this long when the core
# person detector misses everyone for a frame (crouching / turned away), so the red box does not
# blink. Display only; vandalism states, scores and every other branch are unchanged.
VANDALISM_BOX_HOLD_SECONDS = 0.8
SHARP_BOX_SMOOTHING = 0.4
SLOW_CORE_STEP_SECONDS = 0.22  # core inference per step (budget 0.267 s at 3.75 fps)


# Vandalism branch context from the (unchanged) strike branch, see MMActionRuntime.vandalism_evidence.
VANDALISM_LONE_STRIKE_FACTOR = 0.7
VANDALISM_DURING_FIGHT_MIN = 0.9


def _iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def has_target_person(striker, people, max_gap=0.5, min_height=0.5, min_area=0.35, same_iou=0.5):
    """True if another full-size person stands next to the striker box (a possible victim)."""
    left, top, right, bottom = striker
    width, height = right - left, bottom - top
    for box in people:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        inter = max(0., min(right, x2) - max(left, x1)) * max(0., min(bottom, y2) - max(top, y1))
        union = width * height + w * h - inter
        if union > 0 and inter / union >= same_iou:
            continue  # the striker itself
        if h < min_height * height or w * h < min_area * width * height:
            continue  # partial / far / spurious box, not a victim
        gap_x = max(0., max(left, x1) - min(right, x2))
        gap_y = max(0., max(top, y1) - min(bottom, y2))
        if gap_x <= max_gap * width and gap_y <= max_gap * height:
            return True
    return False


def prediction_due(last_prediction, timestamp, stride_seconds, *, fall_wrestle=False):
    if last_prediction is None:
        return True
    # 59.94 fps sources can miss the nominal 0.5333 s boundary by <1 ms.
    # A 20 ms tolerance keeps the intended two-sample cadence at 3.75 fps.
    tolerance = 0.02 if fall_wrestle else 1e-6
    return timestamp - last_prediction >= stride_seconds - tolerance


def person_action_regions(actions, threshold):
    return [
        person["box"]
        for person in actions
        if math.isfinite(person.get("fight_score", 0.0))
        and person.get("fight_score", 0.0) >= threshold
    ]


class MMActionRuntime:
    @property
    def result(self):
        context = getattr(self, '_result_context', None)
        if context is not None and hasattr(context, 'draft'):
            return context.draft
        return self._published_result

    @result.setter
    def result(self, value):
        context = getattr(self, '_result_context', None)
        if context is not None and hasattr(context, 'draft'):
            context.draft = value
        else:
            self._published_result = value

    def __init__(self, tuned=False, allow_pilot=False, fall_wrestle=False, transition=False, fixed_running=False, three_actions=False, separate_actions=False,
                 extensions=False, extension_names=None, extension_python=None):
        self._result_context = threading.local()
        self._published_result = {}
        if separate_actions and (tuned or fall_wrestle or transition or fixed_running or three_actions):
            raise ValueError('Select one independent backend')
        if extensions and not separate_actions:
            raise ValueError('Extensions run only beside the separate-actions backend')
        self.separate_actions = separate_actions
        # Extension behaviours (phase 0+): separate process, additive output only.
        self.extensions_enabled = extensions
        self.extension_names = tuple(extension_names if extension_names is not None else EXTENSION_NAMES) if extensions else ()
        self.extension_python = extension_python
        self.extension_client = None
        self.extension_specs = {}
        self.extension_gates = {}
        self.extension_seen = {}
        self.extension_recent = {}
        self.extension_fed = {}
        self.extension_step = 0
        self.extension_cadence = EXTENSION_CADENCE
        three_actions = three_actions or separate_actions
        if three_actions and (tuned or fall_wrestle or transition or fixed_running):
            raise ValueError('Select one pilot backend')
        if fixed_running:
            if fall_wrestle or transition:
                raise ValueError("Select one pilot backend")
            tuned = True
        if (fall_wrestle or transition) and not tuned:
            raise ValueError("Pilot backend requires a tuned head")
        if fall_wrestle and transition:
            raise ValueError("Select one pilot backend")
        self.backend_name = ("mmaction_separate_actions_ext" if extensions else "mmaction_separate_actions" if separate_actions else "mmaction_three_actions" if three_actions else "mmaction_fixed_running" if fixed_running else
                             "mmaction_transition_redid" if transition else
                             "mmaction_fall_wrestle" if fall_wrestle else
                             "mmaction_interaction_v1" if tuned else "mmaction_original")

        self.device = "cuda:0"
        self.tuned = tuned or three_actions
        self.three_actions = three_actions
        self.action_gates = {}
        self.fall_wrestle = fall_wrestle
        self.transition = transition
        self.fixed_running = fixed_running
        self.running_gate = None
        self.config = ClipConfig(
            frames=9, fps=3.75, size=256, stride_seconds=16 / 30, region_mode="full"
        )
        self.cpu_threads, self.cpu_optimization, self.batch_detection = (
            4,
            "isolated CUDA worker",
            False,
        )
        self.pose_detector = None
        self.pose_on_worker = False
        self.process = self.log = None
        self.frame_buffer = SharedFrameBuffer()
        self.replies = queue.Queue(maxsize=4)
        self.metadata = {
            "architecture": "mmaction_three_actions" if three_actions else "mmaction_interaction_v1" if tuned else "mmaction_slowonly_r101",
            "deployment": {
                "display_name": (
                    "MMAction2 · ba nhãn riêng · pilot" if three_actions else "MMAction2 · đánh / vật / ngã + chạy · pilot"
                    if fixed_running else "MMAction2 · đánh / vật / ngã chung"
                    if fall_wrestle else "MMAction2 transition Re-DID pilot"
                    if transition else "MMAction2 · interaction v1"
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
        if tuned or three_actions:
            checkpoint_name = "three_actions" if three_actions else "interaction_fixed_running" if fixed_running else "interaction_transition" if transition else "interaction_fall_wrestle" if fall_wrestle else "interaction"
            available = (
                available
                and (ROOT / f"weights/{checkpoint_name}.json").is_file()
                and (ROOT / f"weights/{checkpoint_name}.pt").is_file()
            )
            if fixed_running:
                available = available and (ROOT / "weights/interaction_fall_wrestle.pt").is_file()
        if separate_actions:
            available = available and all((ROOT / ('weights/' + name)).is_file() for name in
                                          ('separate_actions_calibration.json', 'interaction_fixed_running.pt', 'interaction_fall_wrestle.pt'))
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
        if separate_actions:
            self.message = ('Ngã = tư thế nằm từ khớp cơ thể hiện tại; hai mẫu xác nhận, không chờ clip AVA. '
                            'Đánh/Chạy giữ nguyên; Vật còn yếu. Ngưỡng nằm thử nghiệm, chưa nghiệm thu camera.') if self.model_ready else 'Nhánh tách hành vi chưa sẵn sàng hoặc chưa bật pilot.'
        self.gate = EventGate(0.4, 0.28, confirm_seconds=0.4)
        self.reset()

    def reset(self):
        if self.pose_on_worker and self.process is not None and self.process.poll() is None:
            self.process.stdin.write('{"op":"reset"}\n')
            self.process.stdin.flush()
            if not self.replies.get(timeout=10).get('reset'):
                raise RuntimeError('GPU posture reset failed')
        self.latest_posture = None
        self.latest_posture_time = None
        self.posture_timings = {}
        if getattr(self, 'posture_detector', None) is not None:
            self.posture_detector.reset()
        self.samples = deque(maxlen=self.config.frames)
        self.sample_times = deque(maxlen=self.config.frames)
        self.last_input = self.next_sample = self.last_prediction = None
        self.last_startup_detection = None
        self.lying_support_boxes = []
        self.lying_previous_observation = None
        self.posture_recovery_boxes = []
        self.posture_recovery_time = None
        self.rgb_result = None
        self._last_confirmed_running_time = None
        self.display_detection = None
        self.display_detection_requested = threading.Event()
        from fight.running_confirmation import RunningConfirmation
        self.running_confirmation = RunningConfirmation()
        from fight.box_tracking import DisplayBoxTracker
        self.rgb_box_tracker = DisplayBoxTracker(maximum_age=.8, maximum_gap=.4)
        self.gate = self.gate.fresh()
        self.action_gates = {name: gate.fresh() for name, gate in self.action_gates.items()}
        if self.running_gate is not None:
            self.running_gate = self.running_gate.fresh()
        if getattr(self, 'extensions_enabled', False):
            self.reset_extensions()
        self.alarm_active = False
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
            "behavior_scope": "three_actions" if self.three_actions else "fight_wrestle_fall_running" if self.fixed_running else "fight_wrestle_fall" if (self.fall_wrestle or self.transition) else "fight",
            "action_scores": {},
            "running_score": None,
            "active_behaviors": [],
            "prediction_timestamp": None,
            "inference_seconds": None,
            "view_quality": None,
            "collection": {"samples": 0, "required": self.config.frames,
                           "seconds_required": (self.config.frames - 1) / self.config.fps},
        }

    def warmup(self):
        if not self.model_ready or self.process is not None:
            return
        if os.name != "nt":
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
            MPLBACKEND="Agg",
            PYTHONUNBUFFERED="1",
            TORCH_HOME=str(ROOT / "weights/torch-cache"),
        )
        command = [str(ROOT / "env/python.exe"), str(ROOT / "worker.py")]
        if self.tuned:
            command.append("--tuned")
        if self.fall_wrestle:
            command.append("--fall-wrestle")
        if self.transition:
            command.append("--transition")
        if self.fixed_running:
            command.append("--fixed-running")
        if self.separate_actions:
            command.append('--separate-actions')
        elif self.three_actions:
            command.append('--three-actions')
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
            self.pose_on_worker = self.metadata.get('posture_device') == 'cuda:0'
            if self.separate_actions:
                self.metadata['fall_policy'] = {'version': 5, 'definition': 'lying_or_transition_to_lying',
                                                'source': 'current_frame_coco_pose_geometry',
                                                'confirmation': 'two_observations_same_lying_person',
                                                'accident_inference': False, 'score_kind': 'binary_rule_not_probability'}
            if self.three_actions:
                self.action_gates = {name: EventGate(value, value * .7, confirm_seconds=behavior_registry.gate_timing(name)[0],
                                                     release_seconds=behavior_registry.gate_timing(name)[1])
                                     for name, value in self.metadata['thresholds'].items()}
            threshold = self.metadata["threshold"]
            self.gate = EventGate(
                threshold,
                threshold * 0.7,
                confirm_seconds=self.metadata["confirm_seconds"],
                # The pilot receives a new score about every 0.53 s. Two low
                # observations now end an alert, instead of about three.
                release_seconds=0.25 if (self.fall_wrestle or self.transition or self.fixed_running) else 1.0,
            )
            if self.fixed_running:
                threshold = self.metadata["running_threshold"]
                self.running_gate = EventGate(threshold, threshold * 0.7,
                                              confirm_seconds=0.8, release_seconds=0.25)
        except (queue.Empty, RuntimeError) as error:
            self.close()
            raise RuntimeError(
                "Không nạp được MMAction2; xem mmaction2_demo/results/web_worker_*.log"
            ) from error
        if self.extensions_enabled:
            self.start_extensions()

    # ------------------------------------------------------------ extensions
    @property
    def behavior_names(self):
        return behavior_registry.CORE_NAMES + tuple(self.extension_specs)

    def start_extensions(self):
        """Start the extension worker. Failure only disables extensions."""
        if self.extension_client is not None or not self.extension_names:
            return
        from fight.extension_client import ExtensionClient

        client = ExtensionClient(self.extension_names, python=self.extension_python)
        self.extension_client = client
        if not client.start():
            return
        for name, spec in client.specs.items():
            if name in behavior_registry.CORE_NAMES:
                continue
            try:
                spec = {**spec, **{k: v for k, v in EXTENSION_GATE_OVERRIDES.get(name, {}).items() if k in spec}}
                release_ratio = EXTENSION_GATE_OVERRIDES.get(name, {}).get('release_ratio', .7)
                gate = EventGate(spec['threshold'], spec['threshold'] * release_ratio,
                                 confirm_seconds=spec['confirm_seconds'], release_seconds=spec['release_seconds'],
                                 max_observation_gap=EXTENSION_GATE_OVERRIDES.get(name, {}).get('max_observation_gap', 1.5))
            except (KeyError, TypeError, ValueError):
                continue
            behavior_registry.register(behavior_registry.BehaviorSpec(
                name, spec.get('display_label') or name.upper(), confirm_seconds=spec['confirm_seconds'],
                release_seconds=spec['release_seconds'], alarm_enabled=bool(spec.get('alarm_enabled', True)),
                kind='extension', reset_without_region=False, blocks_small_person_display=False))
            self.extension_specs[name] = spec
            self.extension_gates[name] = gate

    def sharp_display(self, weapons, holders, state, timestamp, name='sharp_object'):
        """A/C: steady weapon boxes while the alert is on (hold short gaps, blend positions)."""
        hold = self.__dict__.setdefault('sharp_hold', {}) if name == 'sharp_object' else             self.__dict__.setdefault('weapon_hold', {}).setdefault(name, {})
        if state == 'normal':
            hold.clear()
            return [], []
        if weapons:
            if hold.get('source') is not weapons:  # a new extension reply, not the same one re-read
                previous, blended = hold.get('weapons', []), []
                for weapon in weapons:
                    match = max(previous, key=lambda old: _iou(old['box'], weapon['box']), default=None)
                    box = weapon['box']
                    if match is not None and _iou(match['box'], box) > 0.2:
                        box = [SHARP_BOX_SMOOTHING * old + (1 - SHARP_BOX_SMOOTHING) * new
                               for old, new in zip(match['box'], box)]
                    blended.append({**weapon, 'box': [round(v, 1) for v in box]})
                hold.update(source=weapons, weapons=blended, at=timestamp)
            hold['holders'] = holders or hold.get('holders', [])
            return hold['weapons'], hold['holders']
        if hold.get('weapons') and timestamp - hold['at'] <= SHARP_HOLD_SECONDS:
            return hold['weapons'], holders or hold.get('holders', [])
        return [], holders

    def extension_status(self):
        if not self.extensions_enabled:
            return None
        client = self.extension_client.status() if self.extension_client is not None else {'enabled': False}
        return {**client, 'requested': list(self.extension_names), 'active': list(self.extension_specs),
                'cadence': self.extension_cadence}

    def apply_extensions(self, output, timestamp):
        """Add extension labels next to the core ones; core keys are never changed."""
        names = tuple(self.extension_specs)
        client = self.extension_client
        if client is not None and client.enabled and output.get('prediction_timestamp') == timestamp \
                and len(self.samples) == self.config.frames:
            self.extension_step += 1
            timings = output.get('timings') or {}
            step = max(timings.get('pipeline_seconds', 0) or 0, output.get('inference_seconds') or 0)
            self.extension_cadence = EXTENSION_SLOW_CADENCE if step > SLOW_CORE_STEP_SECONDS else EXTENSION_CADENCE
            light = [n for n in EVERY_STEP_EXTENSIONS if n in self.extension_specs]
            people = {'person_boxes': output.get('person_boxes', []),
                      'posture_boxes': (output.get('posture') or {}).get('boxes', []),
                      'frame_size': [self.samples[-1].shape[1], self.samples[-1].shape[0]]}
            # 18/11/2026: a full step (vandalism + sharp_object) that found the worker busy with a
            # light sharp_object step used to be dropped until the next cadence slot, so vandalism
            # replies came only every ~1.6 s and the alarm needed ~2 s more to confirm. A due full
            # step now stays due: no light step is sent while it waits, and it goes out on the next
            # idle core step. sharp_object still runs inside every full step.
            if self.extension_step % self.extension_cadence == 0:
                self.extension_full_due = True
            # 18/11/2026: while vandalism is verifying (yellow, not yet red), score it on every core
            # step so the confirming second high score comes 0.27-0.5 s after the first instead of
            # waiting for the next cadence slot. Back to the normal cadence once red or normal.
            vandal_gate = self.extension_gates.get('vandalism')
            if vandal_gate is not None and vandal_gate.pending_since is not None and not vandal_gate.active:
                self.extension_full_due = True
            full_due = getattr(self, 'extension_full_due', False)
            if not full_due and light:
                import cv2
                height, width = self.samples[-1].shape[:2]
                scale = min(1., SHARP_OBJECT_MAX_SIDE / max(width, height))
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                # same 9-frame shape as a full step (the shared buffer is fixed-size); sharp_object reads the last one
                client.submit([cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in self.samples],
                              self.sample_times[-1], people, only=light)
            elif full_due:
                import cv2
                height, width = self.samples[-1].shape[:2]
                size = EXTENSION_FRAME_SIZE
                if any(name in self.extension_specs for name in WEAPON_EXTENSIONS):
                    scale = min(1., SHARP_OBJECT_MAX_SIDE / max(width, height))
                    size = (max(1, round(width * scale)), max(1, round(height * scale)))
                small = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in self.samples]
                if client.submit(small, self.sample_times[-1],
                                 {'person_boxes': output.get('person_boxes', []),
                                  'posture_boxes': (output.get('posture') or {}).get('boxes', []),
                                  'frame_size': [width, height]}):
                    self.extension_full_due = False
        reply = client.latest(timestamp) if client is not None else None
        results = (reply or {}).get('extensions', {})
        active = [name for name in output.get('active_behaviors', []) if name not in names]
        states = {k: v for k, v in output.get('action_states', {}).items() if k not in names}
        scores = {k: v for k, v in output.get('action_scores', {}).items() if k not in names}
        regions = {k: v for k, v in output.get('action_regions', {}).items() if k not in names}
        details = {}
        context = self.vandalism_context(output)
        for name in names:
            gate = self.extension_gates[name]
            data = results.get(name)
            seen = self.extension_seen.get(name)
            if data is not None and 'error' not in data and (seen is None or reply['timestamp'] > seen['timestamp']):
                window = EXTENSION_GATE_OVERRIDES.get(name, {}).get('smooth_window', 1)
                recent = self.extension_recent.setdefault(name, deque(maxlen=window))
                recent.append(data['score'])
                smoothed = sum(recent) / len(recent)
                fed = self.vandalism_evidence(smoothed, context) if name == 'vandalism' else smoothed
                if 'k_of_n' in EXTENSION_GATE_OVERRIDES.get(name, {}):
                    k, n = EXTENSION_GATE_OVERRIDES[name]['k_of_n']
                    history = self.__dict__.setdefault('extension_kofn', {}).setdefault(name, deque(maxlen=n))
                    history.append(data['score'])
                    fed = sorted(history, reverse=True)[k - 1] if len(history) >= k else min(history)
                overrides = EXTENSION_GATE_OVERRIDES.get(name, {})
                motion = None
                if 'min_motion' in overrides:
                    motion = self.people_motion(output)
                    if motion is not None and motion < overrides['min_motion']:
                        fed = 0.  # M1: people standing still -> no evidence
                try:
                    gate.update(fed, reply['timestamp'])
                    released = False
                    if 'release_window' in overrides:
                        fed_history = self.extension_fed.setdefault(name, deque(maxlen=overrides['release_window']))
                        fed_history.append(fed)
                        if (gate.active and len(fed_history) == fed_history.maxlen
                                and sum(fed_history) / len(fed_history) < gate.release_threshold):
                            self.extension_gates[name] = gate = gate.fresh()  # M2: quick release
                            fed_history.clear()
                            released = True
                    seen = {'timestamp': reply['timestamp'], 'score': data['score'], 'fed': fed, 'boxes': data['boxes'],
                            'motion': motion, 'released': released, 'details': data.get('details') or {}}
                except ValueError:
                    self.extension_gates[name] = gate = gate.fresh()
                    seen = None
                self.extension_seen[name] = seen
            elif gate.last_observation is not None and timestamp - gate.last_observation > gate.max_observation_gap:
                self.extension_gates[name] = gate = gate.fresh()
                self.extension_seen[name] = seen = None
                self.extension_recent.pop(name, None)
                self.extension_fed.pop(name, None)
                self.__dict__.get('extension_kofn', {}).pop(name, None)
            state = 'fighting' if gate.active else 'verifying' if gate.pending_since is not None else 'normal'
            states[name] = state
            # D: the first high score comes from a window that ends at pending_since, so the action
            # started at most one window earlier. Kept for the event log only (no effect on alarms).
            onsets = self.__dict__.setdefault('extension_onset', {})
            if state == 'normal':
                onsets.pop(name, None)
            elif name not in onsets and gate.pending_since is not None:
                onsets[name] = max(0., gate.pending_since - EXTENSION_WINDOW_SECONDS)
            scores[name] = seen['score'] if seen else 0.
            regions[name] = seen['boxes'] if seen and state != 'normal' else []
            if state == 'fighting' and self.extension_specs[name].get('alarm_enabled', True):
                active.append(name)
            details[name] = {'state': state, 'score': scores[name],
                             'onset_estimate': onsets.get(name),
                             'observed_at': seen['timestamp'] if seen else None,
                             'error': data.get('error') if isinstance(data, dict) else None}
            if name == 'vandalism':
                details[name].update(context=context, gated_score=seen['fed'] if seen else None,
                                     motion=seen.get('motion') if seen else None,
                                     quick_release=seen.get('released') if seen else None)
            if name in WEAPON_EXTENSIONS:  # weapon boxes for the overlay / event log (source pixels)
                weapons, regions[name] = self.sharp_display(
                    (seen.get('details') or {}).get('weapons', []) if seen else [], regions[name], state, timestamp,
                    name)
                details[name].update(weapons=weapons)
        output.update(active_behaviors=active, action_states=states, action_scores=scores, action_regions=regions)
        if 'vandalism' in names:
            hold = self.__dict__.setdefault('vandalism_box_hold', {})
            boxes = output.get('person_boxes') or []
            if states.get('vandalism', 'normal') == 'normal':
                hold.clear()
            elif boxes:
                hold.update(boxes=[list(box) for box in boxes], at=timestamp)
            elif hold.get('boxes') and timestamp - hold['at'] <= VANDALISM_BOX_HOLD_SECONDS:
                output['vandalism_display_boxes'] = [list(box) for box in hold['boxes']]
        if output.get('prediction_timestamp') == timestamp and 'strike_context' not in output:
            # Group-A context (people / positions) for a strike: information only, the strike
            # branch and all core fields stay exactly as the core produced them.
            if getattr(self, 'strike_context', None) is None:
                from fight.strike_context import StrikeContext
                self.strike_context = StrikeContext()
            output['strike_context'] = self.combine_strike_context(
                self.strike_context.observe(output, timestamp), states, details)
        output['presented_behaviors'], output['presented_regions'] = self.present(output)
        output['extensions'] = {'behaviors': details,
                                'enabled': bool(client is not None and client.enabled),
                                'error': client.error if client is not None else None}

    def people_motion(self, output, lag=4, size=(96, 72)):
        """M1: mean absolute grey change (0-255) over ~1 s (lag samples) inside the people boxes
        (whole frame when nobody is detected). None when there are not enough samples."""
        import cv2

        if len(self.samples) <= lag:
            return None
        now, before = self.samples[-1], self.samples[-1 - lag]
        height, width = now.shape[:2]
        a = cv2.resize(cv2.cvtColor(now, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)
        diff = np.abs(a - b)
        boxes = output.get('person_boxes') or []
        if not boxes:
            return float(diff.mean())
        mask = np.zeros(diff.shape, bool)
        sx, sy = size[0] / width, size[1] / height
        for left, top, right, bottom in boxes:
            pad_x, pad_y = .2 * (right - left), .2 * (bottom - top)
            x1, y1 = max(0, int((left - pad_x) * sx)), max(0, int((top - pad_y) * sy))
            x2, y2 = min(size[0], int((right + pad_x) * sx) + 1), min(size[1], int((bottom + pad_y) * sy) + 1)
            mask[y1:y2, x1:x2] = True
        return float(diff[mask].mean()) if mask.any() else float(diff.mean())

    def combine_strike_context(self, context, states, details):
        """A + D1: 'vandalism' only when group A finds no victim AND the vandalism branch
        itself is high (gate verifying/confirmed, or a raw score >= threshold in the last ~1 s).
        Group A alone mislabels fights where both people share one box or the victim is
        partly out of view; in those the vandalism score stays ~0. Information only."""
        if context is None:
            self.strike_episode_vandalism = False
            self.strike_episode_object_hit = False
            return None
        spec = self.extension_specs.get('vandalism')
        recent = self.extension_recent.get('vandalism') or ()
        support = spec is not None and (states.get('vandalism') in {'fighting', 'verifying'}
                                        or max(recent, default=0.) >= spec.get('threshold', 1.))
        group_a = context['verdict']
        verdict = group_a
        signals = context.get('signals', {})
        # Nobody within reach of the striker: no full-size person next to them AND the
        # nearest other person is far (A3). Two fighters merged into one box do not pass A3.
        out_of_reach = signals.get('A1_alone') or (signals.get('A2_no_target') and signals.get('A3_far'))
        if group_a == 'fight' and support and out_of_reach:
            # A5 "nearest person moves" also fires when several people smash together; with the
            # vandalism branch high and nobody within reach, that motion is not a victim reacting.
            verdict = 'vandalism'
        elif group_a == 'vandalism':
            if support:
                verdict = 'vandalism'
            else:
                # Vandalism branch low -> stays a fight. Hiding the strike here ('object_hit'
                # when nobody seems within reach) hid 24 of 42 strike steps on the 3 golden
                # fight videos: the victim is often missed or merged into the striker's box.
                verdict = 'fight'
        if verdict == 'vandalism':
            # Once judged vandalism, the rest of this strike episode is presented as vandalism
            # (avoids the label flickering between steps). Reset when the strike ends.
            self.strike_episode_vandalism = True
        if verdict == 'object_hit':
            # Latch for the episode too: one extra person box for a frame must not bring the
            # strike back (red flicker). Vandalism support or a reacting person overrides it.
            self.strike_episode_object_hit = True
        elif verdict == 'fight' and signals.get('A5_reacts') and not out_of_reach:
            self.strike_episode_object_hit = False
        if getattr(self, 'strike_episode_vandalism', False):
            episode = 'vandalism'
        elif getattr(self, 'strike_episode_object_hit', False):
            episode = 'object_hit'
        else:
            episode = verdict
        return {**context, 'verdict': verdict, 'episode_verdict': episode, 'group_a_verdict': group_a,
                'D1_vandalism_support': bool(support)}

    @staticmethod
    def present(output):
        """Labels shown to people (overlay, event list, UI) in the extension backend.

        A strike judged vandalism by A + D1 is presented as vandalism instead of a fight.
        Detection is unchanged: active_behaviors / action_states / trace keep the strike.
        """
        active = list(output.get('active_behaviors', []))
        regions = {name: list(boxes) for name, boxes in (output.get('action_regions') or {}).items()}
        context = output.get('strike_context') or {}
        scores = output.get('action_scores') or {}
        vandalism_score, strike_score = scores.get('vandalism'), scores.get('strike')
        # Vandalism outscores the strike -> show only vandalism (user rule, 17/11/2026).
        vandalism_wins = (vandalism_score is not None and strike_score is not None
                          and float(vandalism_score) > float(strike_score))
        if 'strike' in active and (context.get('episode_verdict') == 'vandalism' or vandalism_wins):
            active = [name for name in active if name != 'strike']
            if 'vandalism' not in active:
                active.append('vandalism')
            regions['vandalism'] = regions.get('vandalism', []) + [b for b in regions.get('strike', [])
                                                                    if b not in regions.get('vandalism', [])]
            regions['strike'] = []
        elif 'strike' in active and MMActionRuntime.object_swing(output):
            # AVA (core head, read only): someone holds an object and nobody shows any
            # person-directed action -> a swing at an object (chopping wood), not a fight.
            # 3 golden fight videos: 0/42 strike steps hidden; chopping wood: 8/8 hidden.
            active = [name for name in active if name != 'strike']
            regions['strike'] = []
            if isinstance(output.get('strike_context'), dict):
                output['strike_context']['presented_as'] = 'object_hit'
        return active, regions

    @staticmethod
    def object_swing(output):
        names = {label.get('name', '') for person in output.get('actions') or [] for label in person.get('labels', [])}
        return any('(an object)' in name for name in names) and not any('(a person)' in name for name in names)

    @staticmethod
    def vandalism_context(output):
        """Read-only look at the core strike branch, used as context by the vandalism branch.

        'lone_strike'        strike fires on one person with nobody next to them
                             (AVA "hit a person" firing on someone hitting an object)
        'strike_with_victim' strike fires and another full-size person stands next to them
        'none'               no strike
        The core output is never modified.
        """
        if output.get('action_states', {}).get('strike') not in {'fighting', 'verifying'}:
            return 'none'
        strikes = output.get('action_regions', {}).get('strike') or []
        if not strikes:
            return 'none'
        people = output.get('person_boxes', [])
        return 'strike_with_victim' if any(has_target_person(box, people) for box in strikes) else 'lone_strike'

    def vandalism_evidence(self, score, context):
        """Score fed to the vandalism gate (the raw score is still reported).

        lone_strike:        violent swing with no victim supports vandalism, so the
                            release threshold (0.7 x threshold) is enough to count.
        strike_with_victim: a fight is going on; vandalism must be very clear (>= 0.9)
                            so a fight is not also reported as vandalism.
        """
        threshold = self.extension_specs['vandalism'].get('threshold', 0.65)
        if context == 'lone_strike' and score >= VANDALISM_LONE_STRIKE_FACTOR * threshold:
            return max(score, threshold)
        if context == 'strike_with_victim' and score < VANDALISM_DURING_FIGHT_MIN:
            return min(score, VANDALISM_LONE_STRIKE_FACTOR * threshold * 0.99)
        return score

    def reset_extensions(self):
        self.extension_gates = {name: gate.fresh() for name, gate in self.extension_gates.items()}
        self.extension_seen = {}
        self.extension_recent = {}
        self.extension_fed = {}
        self.extension_kofn = {}
        if getattr(self, 'strike_context', None) is not None:
            self.strike_context.reset()
        self.extension_step = 0
        self.extension_full_due = False
        if self.extension_client is not None:
            self.extension_client.reset()

    def push(self, frame, timestamp, predict=True, *, sampled=False):
        if not self.separate_actions or not self.model_ready or not predict:
            return self.push_actions(frame, timestamp, predict, sampled=sampled)
        self._result_context.draft = dict(self._published_result)
        try:
            output = self.push_separate(frame, timestamp, predict, sampled=sampled)
        finally:
            del self._result_context.draft
        if self.extensions_enabled:
            try:
                self.apply_extensions(output, timestamp)
            except Exception as error:  # an extension problem must never stop the core
                if self.extension_client is not None:
                    self.extension_client.error = f'Extension merge failed: {error}'
        if output.get('state') == 'fighting' and 'fast_running' in output.get('active_behaviors', []):
            confirmed_at = output.get('rgb_prediction_timestamp')
            if confirmed_at is None:
                confirmed_at = output.get('prediction_timestamp')
            if confirmed_at is not None:
                self._last_confirmed_running_time = confirmed_at
        output['last_confirmed_running_at'] = self._last_confirmed_running_time
        self._published_result = output
        return dict(output)

    def push_separate(self, frame, timestamp, predict=True, *, sampled=False):
        pipeline_started = time.perf_counter()
        due = self.next_sample is None or sampled or timestamp + 1e-6 >= self.next_sample
        displayed = self.result
        if self.rgb_result is not None:
            self.result = self.rgb_result
        output = self.push_actions(frame, timestamp, predict, sampled=sampled)
        self.rgb_result = dict(output)
        if not due:
            self.result = {**displayed, 'event': None}
            return dict(self.result)
        output = self.rgb_box_tracker.align(frame, timestamp, output)
        output['box_timestamp'] = timestamp
        if not self.pose_on_worker and getattr(self, 'posture_detector', None) is None:
            from fight.lying_posture import LyingPosture
            self.posture_detector = LyingPosture(ROOT.parent / 'models/yolov8n-pose.pt')
        if output.get('person_boxes'):
            self.posture_recovery_boxes = output['person_boxes']
            self.posture_recovery_time = timestamp
        recovery = self.posture_recovery_boxes if self.posture_recovery_time is not None and timestamp-self.posture_recovery_time <= .8 else []
        if self.pose_on_worker:
            if self.latest_posture_time != timestamp:
                self.observe_worker_posture(frame, timestamp, recovery)
            posture = self.latest_posture
        else:
            posture = self.posture_detector.observe(frame, timestamp, recovery_boxes=recovery)
        active = [name for name in output.get('active_behaviors', []) if name != 'fall']
        if posture['confirmed']:
            active.append('fall')
        states = dict(output.get('action_states', {}))
        states['fall'] = 'fighting' if posture['confirmed'] else 'verifying' if posture['candidates'] else 'normal'
        scores = dict(output.get('action_scores', {}))
        scores['fall'] = 1. if posture['candidates'] else 0.
        from fight.posture_display import canonical_posture_view
        boxes, actions, action_regions = canonical_posture_view(output, posture, timestamp)
        active = [name for name in active if action_regions.get(name)]
        for name in behavior_registry.reset_without_region_names():
            if not action_regions.get(name):
                states[name] = 'normal'
        regions = [region for name, values in action_regions.items() if name in active or states.get(name) == 'verifying' for region in values]
        output.update(active_behaviors=active, action_states=states, action_scores=scores,
                      action_regions=action_regions, regions=regions, person_boxes=boxes, people=len(boxes),
                      actions=actions, fall_policy=self.metadata.get('fall_policy'),
                      visible_people=len(boxes), box_timestamp=timestamp,
                      state='fighting' if active else 'verifying' if 'verifying' in states.values() else
                      'warming_up' if len(self.samples) < self.config.frames else 'insufficient' if not boxes else 'normal',
                      posture=posture, fall_confirmation={'source': 'current_frame_pose', 'confirmed': bool(posture['confirmed'])},
                      message='Ngã = tư thế nằm từ khớp cơ thể hiện tại; xác nhận hai mẫu, không kết luận tai nạn.')
        output['prediction_timestamp'] = timestamp
        output['rgb_prediction_timestamp'] = self.rgb_result.get('prediction_timestamp')
        output['display_geometry'] = None
        display_started = time.perf_counter()
        from fight.display_geometry import needs_small_person_display
        if needs_small_person_display(self.rgb_result, posture, frame.shape, output['action_states']):
            self.prepare_display_geometry(frame, timestamp, output)
        output['timings'] = {**output.get('timings', {}), **self.posture_timings,
                             'display_detection_seconds': time.perf_counter() - display_started,
                             'rgb_inference_ran': self.last_prediction == timestamp,
                             'pipeline_seconds': time.perf_counter() - pipeline_started,
                             'pose_device': 'cuda:0' if self.pose_on_worker else 'cpu'}
        output['score'] = max(scores.values(), default=0.)
        output['action_thresholds'] = {**output.get('action_thresholds', {}), 'fall': .5}
        output['action_evidence'] = {**output.get('action_evidence', {}), 'fall': 'current_frame_pose'}
        output['display_policy'] = 'running_confirmed_geometry_v1'
        self.result = output
        return dict(output)

    def request_display_detection(self):
        self.display_detection_requested.set()

    def prepare_display_geometry(self, frame, timestamp, output):
        raw = self.rgb_result
        anchor = raw.get('box_timestamp')
        if anchor is None or not self.samples:
            return
        reference_time, reference = min(zip(self.sample_times, self.samples), key=lambda item: abs(item[0] - anchor))
        if abs(reference_time - anchor) > .05:
            return
        observed = {'person_boxes': raw.get('person_boxes', []), 'timestamp': anchor, 'frame': reference}
        if self.display_detection is not None and self.display_detection['timestamp'] > anchor:
            observed = self.display_detection
        requested = self.display_detection_requested.is_set()
        if (timestamp - observed['timestamp'] >= .24 and (raw.get('person_boxes') or requested or len(self.samples) < self.config.frames)
                and self.pose_on_worker):
            observed = self.detect_display(frame, timestamp)
            self.display_detection = observed
        self.display_detection_requested.clear()
        from fight.display_geometry import display_geometry
        output['display_geometry'] = display_geometry(raw, observed, reference, observed['frame'],
                                                     output['active_behaviors'], output['action_states'])

    def detect_display(self, frame, timestamp):
        path = ROOT / 'results/ipc' / (uuid.uuid4().hex + '.npz')
        try:
            if os.name == 'nt':
                request = self.frame_buffer.write([frame] * self.config.frames)
            else:
                np.savez(path, frames=np.stack([frame] * self.config.frames))
                request = {'frames': str(path), 'transport': 'file'}
            self.process.stdin.write(json.dumps({**request, 'op': 'display_detect', 'timestamp': timestamp}) + '\n')
            self.process.stdin.flush()
            response = self.replies.get(timeout=15)
            if 'error' in response:
                raise RuntimeError(response['error'])
            return {'person_boxes': response['result']['person_boxes'], 'timestamp': timestamp, 'frame': frame.copy()}
        finally:
            if os.name != 'nt':
                path.unlink(missing_ok=True)

    def accept_worker_posture(self, response, timestamp):
        if self.pose_on_worker:
            if response.get('posture') is None:
                raise RuntimeError('GPU worker returned no posture observation')
            self.latest_posture = response['posture']
            self.latest_posture_time = timestamp
            self.posture_timings = {'pose_seconds': response['timings']['pose_seconds']}

    def observe_worker_posture(self, frame, timestamp, recovery_boxes):
        path = ROOT / 'results/ipc' / (uuid.uuid4().hex + '.npz')
        try:
            if os.name == 'nt':
                request = self.frame_buffer.write([frame] * self.config.frames)
            else:
                np.savez(path, frames=np.stack([frame] * self.config.frames))
                request = {'frames': str(path), 'transport': 'file'}
            self.process.stdin.write(json.dumps({**request, 'op': 'pose', 'timestamp': timestamp,
                                                'recovery_boxes': recovery_boxes}) + '\n')
            self.process.stdin.flush()
            response = self.replies.get(timeout=15)
            if 'error' in response:
                raise RuntimeError(response['error'])
            self.accept_worker_posture(response, timestamp)
        finally:
            if os.name != 'nt':
                path.unlink(missing_ok=True)

    def push_actions(self, frame, timestamp, predict=True, *, sampled=False):
        if not math.isfinite(timestamp) or (
            self.last_input is not None and timestamp < self.last_input
        ):
            raise ValueError("Finite monotonic timestamps required")
        self.result = {**self.result, "event": None}
        if not self.model_ready:
            return dict(self.result)
        # Realtime playback drops samples when the GPU falls behind (gaps of ~1-1.6 s). Keep the
        # collected clip across such gaps instead of clearing it and warming up again for ~2 s;
        # only a long break (paused source, reconnect) starts over.
        if self.last_input is not None and timestamp - self.last_input > MAX_CONTINUOUS_GAP_SECONDS:
            self.reset()
        self.last_input = timestamp
        if self.next_sample is None:
            self.next_sample = timestamp
        if not sampled and timestamp + 1e-6 < self.next_sample:
            return dict(self.result)
        while self.next_sample <= timestamp + 1e-6:
            self.next_sample += 1 / self.config.fps
        self.samples.append(frame.copy())
        self.sample_times.append(timestamp)
        self.result['collection'] = {'samples': len(self.samples), 'required': self.config.frames,
                                     'seconds_required': (self.config.frames - 1) / self.config.fps}
        if len(self.samples) < self.config.frames:
            self.result['message'] = (f'Đang thu chuyển động: {len(self.samples)}/{self.config.frames} mẫu; '
                                      'khung người không phải kết luận hành vi.')
            if predict and (self.last_startup_detection is None or
                            timestamp - self.last_startup_detection >= self.config.stride_seconds - .02):
                self.detect_startup(frame, timestamp)
            return dict(self.result)
        if (
            not predict
            or not prediction_due(self.last_prediction, timestamp,
                                  self.config.stride_seconds,
                                  fall_wrestle=self.fall_wrestle or self.transition or self.fixed_running or self.three_actions)
        ):
            return dict(self.result)
        self.warmup()
        started = time.perf_counter()
        path = ROOT / "results/ipc" / (uuid.uuid4().hex + ".npz")
        try:
            if os.name == "nt":
                request = self.frame_buffer.write(self.samples)
            else:
                np.savez(path, frames=np.stack(self.samples))
                request = {"frames": str(path), "transport": "file"}
            prepared = time.perf_counter()
            self.process.stdin.write(json.dumps({**request, 'timestamp': timestamp}) + "\n")
            self.process.stdin.flush()
            response = self.replies.get(timeout=45)
            if "error" in response:
                raise RuntimeError(response["error"])
            output = response["result"]
            self.accept_worker_posture(response, timestamp)
        except (queue.Empty, BrokenPipeError) as error:
            raise RuntimeError("MMAction2 bị gián đoạn; xem log worker.") from error
        finally:
            if os.name != "nt":
                path.unlink(missing_ok=True)
        self.last_prediction = timestamp
        state, event = self.update_decision(output, timestamp)
        if not self.tuned:
            output["regions"] = person_action_regions(
                output.get("actions", []), self.gate.threshold
            )
        if not output["people"] and not self.fixed_running:
            state, event = "insufficient", "ended" if self.gate.active else None
            self.gate = self.gate.fresh()
        self.result = {
            **self.result,
            **output,
            "state": state,
            "event": event,
            "prediction_timestamp": timestamp,
            "box_timestamp": self.sample_times[-1] if output.get("box_frame_index") == -1 else self.sample_times[4],
            "action_keyframe_timestamp": self.sample_times[4],
            "visible_people": len(output.get("person_boxes", [])),
            "inference_seconds": time.perf_counter() - started,
            "timings": {**response.get("timings", {}),
                        "frame_copy_seconds": prepared - started,
                        "transport": request["transport"]},
            "message": self.message,
        }
        return dict(self.result)

    def detect_startup(self, frame, timestamp):
        self.warmup()
        if os.name == 'nt':
            request = self.frame_buffer.write([frame] * self.config.frames)
            path = None
        else:
            path = ROOT / 'results/ipc' / (uuid.uuid4().hex + '.npz')
            np.savez(path, frames=np.stack([frame] * self.config.frames))
            request = {'frames': str(path), 'transport': 'file'}
        try:
            self.process.stdin.write(json.dumps({**request, 'op': 'detect', 'timestamp': timestamp}) + '\n')
            self.process.stdin.flush()
            response = self.replies.get(timeout=45)
            if 'error' in response:
                raise RuntimeError(response['error'])
            output = response['result']
            self.accept_worker_posture(response, timestamp)
            self.result.update(person_boxes=output['person_boxes'], actions=output['actions'],
                               people=output['people'], visible_people=output['people'],
                               box_timestamp=timestamp)
            self.last_startup_detection = timestamp
        except (queue.Empty, BrokenPipeError) as error:
            raise RuntimeError('Detector khởi đầu bị gián đoạn; xem log worker.') from error
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def update_decision(self, output, timestamp):
        if self.three_actions:
            previous = self.alarm_active
            states, regions, behaviors = {}, [], []
            if not output['people']:
                if self.separate_actions:
                    self.running_confirmation.update([], timestamp, self.action_gates['fast_running'].threshold)
                self.lying_support_boxes = []
                self.lying_previous_observation = None
                self.action_gates = {name: gate.fresh() for name, gate in self.action_gates.items()}
                self.alarm_active = False
                output.update(active_behaviors=[], regions=[], action_states={})
                return 'insufficient', 'ended' if previous else None
            for name, gate in self.action_gates.items():
                if self.separate_actions and name == 'fast_running':
                    running_state, running_regions, diagnostics = self.running_confirmation.update(
                        output.get('actions', []), timestamp, gate.threshold)
                    states[name] = running_state
                    output['action_regions'][name] = running_regions
                    output['running_confirmation'] = diagnostics
                    output.setdefault('action_evidence', {})[name] = f"per_person_running_v{self.metadata.get('running_policy', {}).get('version', 3)}"
                    if running_state in {'fighting', 'verifying'}:
                        regions.extend(running_regions)
                    if running_state == 'fighting':
                        behaviors.append(name)
                    continue
                if self.separate_actions and name == 'grapple':
                    self.action_gates[name] = gate.fresh()
                    states[name] = 'normal'
                    output.setdefault('action_evidence', {})[name] = 'diagnostic_only_unvalidated_grapple_pilot'
                    continue
                continuation = False
                if self.separate_actions and name == 'fall':
                    from mmaction2_demo.lying_policy import continued_lying_boxes
                    current_score = output['action_scores']['fall']
                    prior = self.lying_previous_observation
                    retrospective = False
                    if (gate.pending_since is None and not gate.active and current_score >= gate.threshold
                            and prior is not None and 0 < timestamp - prior['timestamp'] <= gate.max_observation_gap
                            and continued_lying_boxes(output.get('actions', []), prior['boxes'], gate.threshold)):
                        gate.pending_since = prior['timestamp']
                        retrospective = True
                    if gate.pending_since is not None or gate.active:
                        supported = continued_lying_boxes(output.get('actions', []), self.lying_support_boxes,
                                                         gate.release_threshold)
                        continuation = bool(supported)
                        if continuation and not output['action_regions']['fall']:
                            output['action_regions']['fall'] = supported
                            output.setdefault('action_evidence', {})['fall'] = 'tracked_lying_above_release_threshold'
                    self.lying_support_boxes = (output['action_regions']['fall']
                                               if continuation or output['action_scores']['fall'] >= gate.threshold else [])
                    if gate.active and not continuation and output['action_scores']['fall'] < gate.threshold:
                        self.action_gates[name] = gate.fresh()
                        gate = self.action_gates[name]
                    output['fall_confirmation'] = {'continuation': continuation, 'on_threshold': gate.threshold,
                                                   'off_threshold': gate.release_threshold,
                                                   'previous_support_used': retrospective}
                    weak_boxes = continued_lying_boxes(output.get('actions', []), output.get('person_boxes', [])
                                                       or [person['box'] for person in output.get('actions', [])],
                                                       gate.release_threshold)
                    self.lying_previous_observation = {'timestamp': timestamp, 'boxes': weak_boxes} if weak_boxes else None
                if self.separate_actions and name in {'fast_running', 'fall'} and output['action_scores'][name] == 0:
                    self.action_gates[name] = gate.fresh()
                    states[name] = 'normal'
                else:
                    states[name], _ = gate.update(output['action_scores'][name], timestamp, continue_pending=continuation)
                if states[name] in {'fighting', 'verifying'}:
                    regions.extend(output['action_regions'][name])
                if states[name] == 'fighting':
                    behaviors.append(name)
            self.alarm_active = bool(behaviors)
            output.update(active_behaviors=behaviors, regions=regions, action_states=states,
                          action_thresholds={name: gate.threshold for name, gate in self.action_gates.items()})
            state = 'fighting' if behaviors else 'verifying' if 'verifying' in states.values() else 'normal'
            event = 'started' if self.alarm_active and not previous else 'ended' if previous and not self.alarm_active else None
            return state, event
        if not self.fixed_running:
            return self.gate.update(output["score"], timestamp)
        previous = self.alarm_active
        if not output["people"]:
            self.gate = self.gate.fresh()
            self.running_gate = self.running_gate.fresh()
            self.alarm_active = False
            output.update(active_behaviors=[], regions=[])
            return "insufficient", "ended" if previous else None
        fight_state, _ = self.gate.update(output["score"], timestamp)
        if output.get('running_guard_version', 0) >= 2 and output['running_score'] == 0:
            self.running_gate = self.running_gate.fresh()
            running_state = 'normal'
        else:
            running_state, _ = self.running_gate.update(output["running_score"], timestamp)
        behaviors = []
        regions = []
        if fight_state in {"fighting", "verifying"}:
            regions.extend(output["regions"])
        if running_state in {"fighting", "verifying"}:
            regions.extend(output["running_regions"])
        if fight_state == "fighting":
            behaviors.append("fight_wrestle_fall")
        if running_state == "fighting":
            behaviors.append("fast_running")
        self.alarm_active = bool(behaviors)
        output.update(active_behaviors=behaviors, regions=regions,
                      fight_state=fight_state, running_state=running_state)
        state = "fighting" if behaviors else "verifying" if "verifying" in {fight_state, running_state} else "normal"
        event = "started" if self.alarm_active and not previous else "ended" if previous and not self.alarm_active else None
        return state, event

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
        self.frame_buffer.close()
        if getattr(self, 'extension_client', None) is not None:
            self.extension_client.close()
            self.extension_client = None
            for name in self.extension_specs:
                behavior_registry.unregister(name)
            self.extension_specs, self.extension_gates, self.extension_seen = {}, {}, {}
