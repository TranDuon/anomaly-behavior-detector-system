"""Current-frame lying evidence from visible COCO joints, not AVA action scores."""

import numpy as np


def reliable_person(points):
    points = np.asarray(points)
    return (points.shape == (17, 3) and np.isfinite(points).all()
            and max(points[5, 2], points[6, 2]) >= .35
            and max(points[11, 2], points[12, 2]) >= .35
            and (points[:, 2] >= .35).sum() >= 6)


def lying_posture(points):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (17, 3) or not np.isfinite(points).all():
        return False, 'invalid_pose'
    for shoulder, hip, knee, ankle in ((5, 11, 13, 15), (6, 12, 14, 16)):
        joints = points[[shoulder, hip, knee, ankle]]
        if (joints[:, 2] < .3).any():
            continue
        torso = joints[1, :2] - joints[0, :2]
        leg = joints[3, :2] - joints[1, :2]
        body = joints[3, :2] - joints[0, :2]
        torso_length, leg_length = np.linalg.norm(torso), np.linalg.norm(leg)
        if min(torso_length, leg_length) < 8:
            continue
        aligned = np.dot(torso, leg) / (torso_length * leg_length)
        if (abs(torso[0]) >= 1.2 * abs(torso[1]) and abs(body[0]) >= 1.5 * abs(body[1])
                and aligned >= .45 and leg_length >= .65 * torso_length):
            return True, 'visible_horizontal_torso_and_extended_leg'
        visible = points[points[:, 2] >= .3, :2]
        extent = np.ptp(visible, axis=0)
        if (joints[:, 2].min() >= .5 and extent[0] >= .95 * extent[1]
                and leg_length >= 2.2 * torso_length and aligned >= .8
                and abs(body[0]) >= .85 * abs(body[1])):
            return True, 'foreshortened_reclined_torso_and_extended_leg'
    if foreshortened_lying(points):
        return True, 'depth_aligned_compressed_torso_and_extended_leg'
    return False, 'insufficient_or_nonlying_pose'


def foreshortened_lying(points):
    upper = points[[5, 6, 11, 12]]
    if upper[:, 2].min() < .5:
        return False
    shoulder_width = np.linalg.norm(upper[0, :2] - upper[1, :2])
    shoulders = upper[:2, :2].mean(axis=0)
    hips = upper[2:, :2].mean(axis=0)
    torso = hips - shoulders
    torso_length = np.linalg.norm(torso)
    if shoulder_width < 12 or not 8 <= torso_length <= .7 * shoulder_width:
        return False
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        joints = points[[hip, knee, ankle]]
        if joints[:, 2].min() < .5:
            continue
        thigh = joints[1, :2] - joints[0, :2]
        shin = joints[2, :2] - joints[1, :2]
        leg = joints[2, :2] - joints[0, :2]
        lengths = np.linalg.norm(thigh) * np.linalg.norm(shin)
        leg_length = np.linalg.norm(leg)
        if lengths < 64 or leg_length < 1.25 * torso_length:
            continue
        extension = np.dot(thigh, shin) / lengths
        alignment = np.dot(torso, leg) / (torso_length * leg_length)
        if extension >= .34 and alignment >= .75:
            return True
    return False


class LyingPosture:
    def __init__(self, model_path, device='cpu'):
        from fight.pose import PoseConfig, PoseDetector
        self.detector = PoseDetector(PoseConfig(image_size=640), model_path=model_path, device=device)
        self.previous = []
        self.previous_time = None

    def reset(self):
        self.previous = []
        self.previous_time = None

    def observe(self, frame, timestamp, recovery_boxes=()):
        from mmaction2_demo.person_recovery import overlap
        records = [record for record in self.detector.poses(frame) if reliable_person(record['points'])]
        if not any(lying_posture(record['points'])[0] for record in records) and frame is not None:
            height, width = frame.shape[:2]
            for bounds in list(recovery_boxes)[:2]:
                left, top, right, bottom = bounds
                if right - left < .8 * (bottom - top):
                    continue
                pad = .15 * max(right - left, bottom - top)
                left, top = max(0, int(left-pad)), max(0, int(top-pad))
                right, bottom = min(width, int(right+pad)), min(height, int(bottom+pad))
                crop = frame[top:bottom, left:right]
                if min(crop.shape[:2]) < 16:
                    continue
                for record in self.detector.poses(crop.copy()):
                    points = record['points'].copy()
                    points[:, :2] += [left, top]
                    box = record['box'] + np.array([left, top, left, top])
                    if not reliable_person(points) or not lying_posture(points)[0]:
                        continue
                    records = [old for old in records if overlap(box, np.asarray([old['box']]))[0] < .3]
                    records.append({'box': box, 'points': points})
        candidates = []
        for record in records:
            valid, reason = lying_posture(record['points'])
            if valid:
                candidates.append({'box': record['box'].tolist(), 'ava_lying_score': 1., 'reason': reason})
        confirmed = []
        if self.previous_time is not None and .15 <= timestamp - self.previous_time <= .8:
            confirmed = [person['box'] for person in candidates if self.previous
                         and overlap(np.asarray(person['box']), np.asarray(self.previous)).max() >= .35]
        self.previous = [person['box'] for person in candidates]
        self.previous_time = timestamp
        return {'boxes': [record['box'].tolist() for record in records],
                'candidates': self.previous, 'confirmed': confirmed,
                'evidence': [person['reason'] for person in candidates]}
