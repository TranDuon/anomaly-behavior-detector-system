"""Use current pose boxes as the canonical display geometry for pose alerts."""

import numpy as np


def canonical_posture_view(output, posture, timestamp):
    from mmaction2_demo.person_recovery import overlap
    fresh = (output.get('box_timestamp') is not None and 0 <= timestamp - output['box_timestamp'] <= .8)
    old = output.get('person_boxes', []) if fresh else []
    boxes = [list(box) for box in (posture['confirmed'] or posture['candidates'])]
    assignments, used = {}, set()
    proposals = []
    for index, box in enumerate(old):
        if not boxes:
            continue
        for target, value in enumerate(overlap(np.asarray(box), np.asarray(boxes))):
            if value >= .2:
                proposals.append((float(value), index, target))
    for _, index, target in sorted(proposals, reverse=True):
        if index not in assignments and target not in used:
            assignments[index] = target
            used.add(target)
    pose_used = set()
    for index, box in enumerate(old):
        if index in assignments:
            continue
        bounds = np.asarray(box, dtype=float)
        size = np.maximum(bounds[2:] - bounds[:2], 1.)
        center = (bounds[:2] + bounds[2:]) / 2
        candidates = []
        for target, pose_box in enumerate(posture['boxes']):
            if target in pose_used or any(np.allclose(pose_box, value, atol=2) for value in boxes):
                continue
            current = np.asarray(pose_box, dtype=float)
            current_size = np.maximum(current[2:] - current[:2], 1.)
            ratio = current_size / size
            distance = np.linalg.norm((current[:2] + current[2:]) / 2 - center) / max(size[1], current_size[1])
            if np.all((ratio >= .5) & (ratio <= 2.)) and distance <= .75:
                iou = float(overlap(bounds, np.asarray([current]))[0])
                candidates.append((iou + 1 - distance, target, pose_box))
        candidates.sort(reverse=True)
        if candidates and (len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= .2):
            _, target, pose_box = candidates[0]
            assignments[index] = len(boxes)
            boxes.append(list(pose_box))
            pose_used.add(target)
    for index, box in enumerate(old):
        if index in assignments or not fresh:
            continue
        if boxes and overlap(np.asarray(box), np.asarray(boxes)).max() >= .2:
            continue
        assignments[index] = len(boxes)
        boxes.append(list(box))
    actions = [{'box': box, 'labels': [], 'fight_score': 0.} for box in boxes]
    for index, target in assignments.items():
        if index < len(output.get('actions', [])):
            actions[target] = {**output['actions'][index], 'box': boxes[target]}
    regions = {}
    for name, values in output.get('action_regions', {}).items():
        regions[name] = []
        for index, target in assignments.items():
            if any(np.allclose(old[index], region, atol=2) for region in values):
                regions[name].append(boxes[target])
    regions['fall'] = posture['confirmed'] or posture['candidates']
    return boxes, actions, regions
