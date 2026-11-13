"""Align delayed person detections to display frames using bounded optical flow."""

from collections import deque
import itertools
import time

import cv2
import numpy as np


class DisplayBoxTracker:
    def __init__(self, width=640, history_seconds=4.0, maximum_age=2.5, maximum_gap=.2):
        self.width = width
        self.history_seconds = history_seconds
        self.maximum_age = maximum_age
        self.maximum_gap = maximum_gap
        self.history = deque(maxlen=160)
        self.prediction_key = None
        self.anchor = None
        self.tracks = []
        self.previous_gray = None
        self.previous_time = None
        self.detector_tracker = None

    def observation_key(self, prediction, anchor, boxes):
        return prediction, anchor

    def corners(self, gray, bounds):
        left, top, right, bottom = np.rint(bounds).astype(int)
        mask = np.zeros_like(gray)
        margin = max(1, round((right - left) * 0.18))
        inset = max(1, round((bottom - top) * 0.08))
        mask[top + inset:bottom - inset, left + margin:right - margin] = 255
        return cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=0.01,
                                       minDistance=3, mask=mask, blockSize=3)

    def seed(self, gray, boxes, scale):
        self.tracks = []
        height, width = gray.shape
        for index, box in enumerate(boxes):
            bounds = np.asarray(box, dtype=np.float32) * scale
            if bounds.shape != (4,) or not np.isfinite(bounds).all():
                continue
            bounds = np.clip(bounds, [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1])
            if min(bounds[2:] - bounds[:2]) < 6:
                continue
            points = self.corners(gray, bounds)
            if points is not None and len(points) >= 4:
                self.tracks.append({"index": index, "box": bounds, "points": points, "age": 0})

    def advance(self, gray, timestamp):
        if not self.tracks:
            self.previous_gray, self.previous_time = gray, timestamp
            return
        if (gray.shape != self.previous_gray.shape or timestamp - self.previous_time > self.maximum_gap
                or timestamp < self.previous_time):
            self.tracks = []
            self.previous_gray, self.previous_time = gray, timestamp
            return
        for track in self.tracks:
            if len(track["points"]) < 12 or track["age"] % 5 == 4:
                refreshed = self.corners(self.previous_gray, track["box"])
                if refreshed is not None and len(refreshed) >= 4:
                    track["points"] = refreshed
            track["age"] += 1
        points = np.concatenate([track["points"] for track in self.tracks])
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None,
                                                  winSize=(21, 21), maxLevel=3)
        if moved is None:
            self.tracks = []
        else:
            back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, moved, None,
                                                      winSize=(21, 21), maxLevel=3)
            valid = status.reshape(-1).astype(bool)
            if back is None:
                valid[:] = False
            else:
                valid &= reverse.reshape(-1).astype(bool)
                valid &= np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 1.5
            valid &= np.isfinite(moved.reshape(-1, 2)).all(axis=1)
            offset = 0
            survivors = []
            height, width = gray.shape
            for track in self.tracks:
                count = len(track["points"])
                selected = valid[offset:offset + count]
                old = points[offset:offset + count][selected].reshape(-1, 2)
                new = moved[offset:offset + count][selected].reshape(-1, 2)
                offset += count
                if len(new) < 4:
                    continue
                flow = new - old
                moving = np.linalg.norm(flow, axis=1) > 0.75
                if moving.sum() >= 4:
                    flow, new = flow[moving], new[moving]
                displacement = np.median(flow, axis=0)
                consistent = np.linalg.norm(flow - displacement, axis=1) <= 3.0
                if consistent.sum() < 4:
                    continue
                displacement = np.median(flow[consistent], axis=0)
                bounds = track["box"] + np.tile(displacement, 2)
                bounds = np.clip(bounds, [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1])
                if min(bounds[2:] - bounds[:2]) < 6:
                    continue
                track.update(box=bounds, points=new[consistent].reshape(-1, 1, 2))
                survivors.append(track)
            self.tracks = survivors
        self.previous_gray, self.previous_time = gray, timestamp

    def align(self, frame, timestamp, result):
        started = time.perf_counter()
        height, width = frame.shape[:2]
        scale = min(1.0, self.width / width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
        if self.history and (self.history[-1][1].shape != gray.shape or timestamp < self.history[-1][0]):
            self.history.clear()
            self.tracks = []
            self.prediction_key = None
        self.history.append((timestamp, gray))
        while self.history and timestamp - self.history[0][0] > self.history_seconds:
            self.history.popleft()
        geometry = result.get('display_geometry')
        if geometry is not None:
            if self.detector_tracker is None:
                self.detector_tracker = DetectionBoxTracker()
            observed = {**result, **geometry, 'display_geometry': None}
            return self.detector_tracker.align(frame, timestamp, observed)
        prediction = result.get("prediction_timestamp")
        anchor = result.get("box_timestamp", prediction)
        boxes = result.get("person_boxes", [])
        key = self.observation_key(prediction, anchor, boxes)
        display = dict(result)
        if anchor is None or anchor > timestamp + 0.05 or timestamp - anchor > self.maximum_age or result.get("score_stale"):
            self.tracks = []
        elif key != self.prediction_key:
            self.prediction_key, self.anchor = key, anchor
            reference_time, reference = min(self.history, key=lambda item: abs(item[0] - anchor))
            self.tracks = []
            if abs(reference_time - anchor) <= 0.1:
                self.seed(reference, boxes, scale)
                self.previous_gray, self.previous_time = reference, reference_time
                for sample_time, sample in self.history:
                    if sample_time > reference_time:
                        self.advance(sample, sample_time)
        elif self.previous_time is not None and timestamp > self.previous_time:
            self.advance(gray, timestamp)
        mapped = {track["index"]: (track["box"] / scale).tolist() for track in self.tracks}
        display["person_boxes"] = list(mapped.values())
        display["visible_people"] = len(mapped) if anchor is not None else None
        display["actions"] = [
            {**action, "box": mapped[index]}
            for index, action in enumerate(result.get("actions", [])) if index in mapped
        ]
        def remap_regions(values):
            regions = []
            for region in values:
                for count in (1, 2):
                    matched = False
                    for indices in itertools.combinations(range(len(boxes)), count):
                        selected = np.asarray([boxes[index] for index in indices])
                        bounds = np.r_[selected[:, :2].min(axis=0), selected[:, 2:].max(axis=0)]
                        if np.allclose(bounds, region, atol=2):
                            if all(index in mapped for index in indices):
                                regions.extend(mapped[index] for index in indices)
                            matched = True
                            break
                    if matched:
                        break
            return regions
        display["regions"] = remap_regions(result.get("regions", []))
        if 'action_regions' in result:
            display['action_regions'] = {name: remap_regions(values) for name, values in result['action_regions'].items()}
        display["box_tracking"] = "optical_flow"
        display["box_tracking_seconds"] = time.perf_counter() - started
        display["display_timestamp"] = timestamp
        confirmed_running = (result.get('display_policy') == 'running_confirmed_geometry_v1'
                             and result.get('state') == 'fighting'
                             and 'fast_running' in result.get('active_behaviors', [])
                             and anchor is not None and -.05 <= timestamp - anchor <= self.maximum_age
                             and not result.get('score_stale'))
        if confirmed_running and not display['regions']:
            display['geometry_status'] = 'running_track_unavailable'
        if boxes and not mapped and not confirmed_running:
            display.update(state="insufficient", event=None, score_stale=True)
        return display


class DetectionBoxTracker(DisplayBoxTracker):
    def __init__(self):
        super().__init__(width=960, history_seconds=1.25, maximum_age=.65)
        self.filters = {}
        self.next_id = 1
        self.measurement_time = None
        self.appearances = []

    def observation_key(self, prediction, anchor, boxes):
        self.measurement_time = anchor
        return anchor

    def align(self, frame, timestamp, result):
        started = time.perf_counter()
        self.frame_timestamp = timestamp
        self.appearances = result.get('appearances', [])
        display = super().align(frame, timestamp, result)
        display['box_tracking'] = 'detector_kalman'
        anchor = result.get('box_timestamp')
        display['tracking_needs_detection'] = (anchor is not None and timestamp - anchor > .35) or bool(result.get('person_boxes') and not display['person_boxes'])
        display['box_tracking_seconds'] = time.perf_counter() - started
        return display

    @staticmethod
    def bounds(state):
        center, size = state[:2], np.maximum(state[2:4], 2.)
        return np.r_[center - size / 2, center + size / 2]

    def projected(self, track, timestamp):
        state = track['filter'].statePost.reshape(-1).copy()
        state[:4] += state[4:] * max(0., timestamp - track['observed'])
        return self.bounds(state)

    def seed(self, gray, boxes, scale):
        from fight.running_confirmation import intersection_over_union
        observed = self.measurement_time
        self.filters = {identity: track for identity, track in self.filters.items()
                        if 0 <= observed - track['observed'] <= self.maximum_age}
        proposals = []
        for index, box in enumerate(boxes):
            bounds = np.asarray(box, np.float32) * scale
            if bounds.shape != (4,) or not np.isfinite(bounds).all() or min(bounds[2:] - bounds[:2]) < 3:
                continue
            candidates = []
            for identity, track in self.filters.items():
                predicted = self.projected(track, observed)
                height = max(bounds[3] - bounds[1], predicted[3] - predicted[1], 1.)
                distance = np.linalg.norm((bounds[:2] + bounds[2:] - predicted[:2] - predicted[2:]) / 2) / height
                ratio = (bounds[2:] - bounds[:2]) / np.maximum(predicted[2:] - predicted[:2], 1.)
                iou = intersection_over_union(bounds, predicted)
                current = self.appearances[index] if index < len(self.appearances) else None
                prior = track.get('appearance')
                affinity = sum(np.sqrt(max(0., first * second)) for first, second in zip(current, prior)) if current and prior else 0.
                unique = len(boxes) == 1 and len(self.filters) == 1
                if distance <= 2. and np.all((ratio >= .4) & (ratio <= 2.5)) and (iou >= .1 or affinity >= .7 or unique):
                    candidates.append((float(iou + affinity - .2 * distance), index, identity))
            candidates.sort(reverse=True)
            if candidates and (len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= .15):
                proposals.append(candidates[0])
        assignments, used = {}, set()
        for _, index, identity in sorted(proposals, reverse=True):
            if identity not in used:
                assignments[index] = identity
                used.add(identity)
        self.tracks = []
        for index, box in enumerate(boxes):
            bounds = np.asarray(box, np.float32) * scale
            if bounds.shape != (4,) or not np.isfinite(bounds).all() or min(bounds[2:] - bounds[:2]) < 3:
                continue
            measurement = np.r_[(bounds[:2] + bounds[2:]) / 2, bounds[2:] - bounds[:2]].astype(np.float32).reshape(4, 1)
            identity = assignments.get(index)
            if identity is None:
                identity = self.next_id
                self.next_id += 1
                kalman = cv2.KalmanFilter(8, 4)
                kalman.measurementMatrix = np.eye(4, 8, dtype=np.float32)
                kalman.measurementNoiseCov = np.diag([4., 4., 16., 16.]).astype(np.float32)
                kalman.errorCovPost = np.diag([25., 25., 25., 25., 4000., 4000., 100., 100.]).astype(np.float32)
                kalman.statePost = np.vstack([measurement, np.zeros((4, 1), np.float32)])
                self.filters[identity] = {'filter': kalman, 'observed': observed, 'observations': 1,
                                          'correction': np.zeros(2), 'correction_time': self.frame_timestamp}
            track = self.filters[identity]
            if observed > track['observed']:
                previous = self.projected(track, self.frame_timestamp)
                delta = observed - track['observed']
                transition = np.eye(8, dtype=np.float32)
                transition[:4, 4:] = np.eye(4, dtype=np.float32) * delta
                track['filter'].transitionMatrix = transition
                track['filter'].processNoiseCov = np.diag([1., 1., 2., 2., 250., 250., 25., 25.]).astype(np.float32) * delta
                track['filter'].predict()
                track['filter'].correct(measurement)
                projected = self.bounds((track['filter'].statePost.reshape(-1)[:4]
                                         + track['filter'].statePost.reshape(-1)[4:] * max(0., self.frame_timestamp - observed)))
                correction = (previous[:2] + previous[2:] - projected[:2] - projected[2:]) / 2
                limit = .25 * max(float(measurement[3, 0]), 1.)
                track['correction'] = np.clip(correction, -limit, limit)
                track['correction_time'] = self.frame_timestamp
                track['observations'] += 1
            track.update(observed=observed, index=index, box=self.bounds(track['filter'].statePost.reshape(-1)),
                         appearance=self.appearances[index] if index < len(self.appearances) else None)
            self.tracks.append(track)
        if len(self.filters) > 24:
            retained = sorted(self.filters, key=lambda identity: self.filters[identity]['observed'], reverse=True)[:24]
            self.filters = {identity: self.filters[identity] for identity in retained}
        self.advance(gray, observed)

    def advance(self, gray, timestamp):
        height, width = gray.shape[:2]
        survivors = []
        for track in self.tracks:
            if 0 <= timestamp - track['observed'] <= self.maximum_age:
                if track['observations'] < 2 and timestamp - track['observed'] > .1:
                    continue
                bounds = self.projected(track, timestamp)
                weight = max(0., 1 - max(0., timestamp - track['correction_time']) / .12)
                bounds += np.tile(track['correction'] * weight, 2)
                bounds = np.clip(bounds, [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1])
                if min(bounds[2:] - bounds[:2]) >= 3:
                    track['box'] = bounds
                    survivors.append(track)
        self.tracks = survivors
        self.previous_gray, self.previous_time = gray, timestamp
