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
from mmaction2_demo.frame_transport import SharedFrameBuffer

ROOT = Path(__file__).resolve().parents[1] / "mmaction2_demo"


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
    def __init__(self, tuned=False, allow_pilot=False, fall_wrestle=False, transition=False, fixed_running=False, three_actions=False, separate_actions=False):
        if separate_actions and (tuned or fall_wrestle or transition or fixed_running or three_actions):
            raise ValueError('Select one independent backend')
        self.separate_actions = separate_actions
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
        self.backend_name = ("mmaction_separate_actions" if separate_actions else "mmaction_three_actions" if three_actions else "mmaction_fixed_running" if fixed_running else
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
        from fight.box_tracking import DisplayBoxTracker
        self.rgb_box_tracker = DisplayBoxTracker(maximum_age=.8, maximum_gap=.4)
        self.gate = self.gate.fresh()
        self.action_gates = {name: gate.fresh() for name, gate in self.action_gates.items()}
        if self.running_gate is not None:
            self.running_gate = self.running_gate.fresh()
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
                self.metadata['fall_policy'] = {'version': 4, 'definition': 'lying_or_transition_to_lying',
                                                'source': 'current_frame_coco_pose_geometry',
                                                'confirmation': 'two_observations_same_horizontal_person',
                                                'accident_inference': False, 'score_kind': 'binary_rule_not_probability'}
            if self.three_actions:
                self.action_gates = {name: EventGate(value, value * .7, confirm_seconds=.8 if name == 'fast_running' else .4,
                                                     release_seconds=.25)
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

    def push(self, frame, timestamp, predict=True, *, sampled=False):
        pipeline_started = time.perf_counter()
        if not self.separate_actions or not self.model_ready or not predict:
            return self.push_actions(frame, timestamp, predict, sampled=sampled)
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
        for name in ('strike', 'grapple', 'fast_running'):
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
        output['timings'] = {**output.get('timings', {}), **self.posture_timings,
                             'rgb_inference_ran': self.last_prediction == timestamp,
                             'pipeline_seconds': time.perf_counter() - pipeline_started,
                             'pose_device': 'cuda:0' if self.pose_on_worker else 'cpu'}
        output['score'] = max(scores.values(), default=0.)
        output['action_thresholds'] = {**output.get('action_thresholds', {}), 'fall': .5}
        output['action_evidence'] = {**output.get('action_evidence', {}), 'fall': 'current_frame_pose'}
        self.result = output
        return dict(output)

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
                self.lying_support_boxes = []
                self.lying_previous_observation = None
                self.action_gates = {name: gate.fresh() for name, gate in self.action_gates.items()}
                self.alarm_active = False
                output.update(active_behaviors=[], regions=[], action_states={})
                return 'insufficient', 'ended' if previous else None
            for name, gate in self.action_gates.items():
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
