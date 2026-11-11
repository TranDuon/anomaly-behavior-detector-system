"""Demo definition: already lying and moving into lying share the fall label."""

import hashlib
import math


LYING_THRESHOLD = .4
ADDITIVE_FIELD = '                    "ava_lying_score": float(values[8]),'


def compatible_engine_hashes(path):
    content = path.read_bytes()
    hashes = {hashlib.sha256(content).hexdigest()}
    for newline in (b'\r\n', b'\n'):
        addition = ADDITIVE_FIELD.encode() + newline
        if content.count(addition) == 1:
            hashes.add(hashlib.sha256(content.replace(addition, b'', 1)).hexdigest())
    return hashes


def lying_score(person):
    score = person.get('ava_lying_score', 0.)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        return 0.
    return float(score)


def lying_evidence(actions):
    return max((lying_score(person) for person in actions), default=0.), [
        person['box'] for person in actions if lying_score(person) >= LYING_THRESHOLD]


def continued_lying_boxes(actions, previous, minimum_score):
    def horizontal(box):
        return (len(box) == 4 and all(math.isfinite(value) for value in box)
                and box[3] > box[1] and box[2] - box[0] >= 1.5 * (box[3] - box[1]))

    def iou(first, second):
        intersection = max(0., min(first[2], second[2]) - max(first[0], second[0])) * max(
            0., min(first[3], second[3]) - max(first[1], second[1]))
        area = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1])
        return intersection / max(area - intersection, 1.)

    return [person['box'] for person in actions
            if lying_score(person) >= minimum_score and horizontal(person['box'])
            and any(horizontal(old) and iou(person['box'], old) >= .35 for old in previous)]
