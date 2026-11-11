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
        prediction = result.get("prediction_timestamp")
        anchor = result.get("box_timestamp", prediction)
        key = (prediction, anchor)
        boxes = result.get("person_boxes", [])
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
        if boxes and not mapped:
            display.update(state="insufficient", event=None, score_stale=True)
        return display
