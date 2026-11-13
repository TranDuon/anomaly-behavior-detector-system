"""Conservative applicability checks for the outdoor running pilot."""

import math


def running_crop_fragment(inner, outer):
    width, height = inner[2] - inner[0], inner[3] - inner[1]
    if min(width, height) <= 0 or outer[3] - outer[1] < 1.25 * height:
        return False
    return (abs(inner[0] - outer[0]) <= .12 * width
            and abs(inner[2] - outer[2]) <= .12 * width
            and abs(inner[1] - outer[1]) <= .12 * height
            and outer[3] >= inner[3] + .2 * height)


def match_running_step(reference, current_frame, original, current, match):
    assignments = dict(match(reference, current_frame, original, current))
    if not hasattr(reference, 'shape') or not hasattr(current_frame, 'shape'):
        return assignments
    old_appearance = [running_appearance(reference, box) for box in original]
    new_appearance = [running_appearance(current_frame, box) for box in current]
    fragments = {index for index, box in enumerate(current)
                 if any(index != other and running_crop_fragment(box, candidate)
                        for other, candidate in enumerate(current))}
    proposals = []
    for old_index, old in enumerate(original):
        if old_index in assignments or not old_appearance[old_index]:
            continue
        candidates = []
        for new_index, new in enumerate(current):
            if new_index in assignments.values() or new_index in fragments or not new_appearance[new_index]:
                continue
            old_height, new_height = old[3] - old[1], new[3] - new[1]
            if min(old_height, new_height) <= 0 or not .5 <= new_height / old_height <= 2.:
                continue
            distance = math.hypot((old[0] + old[2] - new[0] - new[2]) / 2,
                                  (old[1] + old[3] - new[1] - new[3]) / 2) / max(old_height, new_height)
            affinity = sum(math.sqrt(max(0., first * second))
                           for first, second in zip(old_appearance[old_index], new_appearance[new_index]))
            if distance <= 1. and affinity >= .8:
                candidates.append((1 - affinity + .15 * distance
                                   + .1 * abs(math.log(new_height / old_height)), old_index, new_index))
        candidates.sort()
        if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= .12):
            proposals.append(candidates[0])
    for score, old_index, new_index in sorted(proposals):
        competitors = [value for value in proposals if value[2] == new_index and value[1] != old_index]
        if (new_index not in assignments.values()
                and all(value[0] - score >= .12 for value in competitors)):
            assignments[old_index] = new_index
    return assignments


def depth_motion_path(boxes):
    heights = [box[3] - box[1] for box in boxes]
    if len(boxes) != 5 or min(heights) <= 0:
        return False
    changes = [math.log(new / old) for old, new in zip(heights, heights[1:])]
    total = math.log(heights[-1] / heights[0])
    direction = 1 if total > 0 else -1
    shift = math.hypot((boxes[-1][0] + boxes[-1][2] - boxes[0][0] - boxes[0][2]) / 2,
                       (boxes[-1][1] + boxes[-1][3] - boxes[0][1] - boxes[0][3]) / 2)
    return (abs(total) >= .3 and shift <= .8 * max(heights)
            and all(-.08 <= direction * change <= .6 for change in changes))


def recover_running_matches(frames, original, current, actions, assignments, detect, match,
                            depth_supported=None):
    pending = {index for index, person in enumerate(actions)
               if index < len(original)
               and ((index not in assignments and person.get('running_score', 0.) >= .85
                     and person.get('ava_running_score', 0.) >= .25)
                    or (.75 <= person.get('running_score', 0.) < .85
                        and person.get('ava_running_score', 0.) >= .3))
               and person.get('ava_grab_score', 0.) < person.get('ava_running_score', 0.)}
    if not pending:
        return {}
    path = {index: index for index in range(len(original))}
    observations = {index: [box] for index, box in enumerate(original)}
    previous = original
    for frame_index in range(5, 9):
        observed = current if frame_index == 8 else detect(frames[frame_index]).tolist()
        step = match_running_step(frames[frame_index - 1], frames[frame_index], previous, observed, match)
        path = {origin: step[position] for origin, position in path.items() if position in step}
        for origin, position in path.items():
            observations[origin].append(observed[position])
        if not pending.intersection(path):
            return {}
        previous = observed
    occupied = set(assignments.values())
    recovered = {}
    for origin, position in path.items():
        if origin not in pending or (position in occupied and assignments.get(origin) != position):
            continue
        depth = depth_motion_path(observations[origin])
        if actions[origin].get('running_score', 0.) < .85 and not depth:
            continue
        recovered[origin] = position
        if depth and depth_supported is not None:
            depth_supported.add(origin)
    return recovered


def restore_running_fields(frames, original, actions, display, assignments, detect, match):
    depth_supported = set()
    recovered = recover_running_matches(frames, original, display['person_boxes'], actions,
                                        assignments, detect, match, depth_supported)
    for old_index, new_index in recovered.items():
        source = actions[old_index]
        display['actions'][new_index].update(
            running_score=source['running_score'],
            running_action_box=original[old_index],
            running_ava_score=source['ava_running_score'],
            running_grab_score=source.get('ava_grab_score', 0.),
            running_depth_support=old_index in depth_supported,
            running_match='intermediate_detector_chain')


def running_appearance(frame, box):
    import cv2
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    crop = frame[max(0, int(top + .1 * height)):min(frame.shape[0], int(top + .65 * height)),
                 max(0, int(left + .15 * width)):min(frame.shape[1], int(right - .15 * width))]
    if not crop.size:
        return None
    histogram = cv2.calcHist([cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)], [0, 1], None,
                            [8, 4], [0, 180, 0, 256]).reshape(-1)
    total = float(histogram.sum())
    return (histogram / total).tolist() if total else None


def verified_running_evidence(person, people, shape):
    reference = person.get('action_box')
    if reference is None:
        return False, 'uncertain_person_match'
    valid, reason = running_evidence(reference, person['box'], shape)
    if not valid:
        return valid, reason
    action_score = person.get('ava_running_score', 0.0)
    if not math.isfinite(action_score) or action_score < 0.4:
        return False, 'no_ava_running_confirmation'
    if person.get('ava_grab_score', 0.0) >= action_score:
        return False, 'interaction_not_running'
    box = person['box']
    area = (box[2] - box[0]) * (box[3] - box[1])
    for other in people:
        if other is person:
            continue
        candidate = other['box']
        intersection = max(0, min(box[2], candidate[2]) - max(box[0], candidate[0])) * max(0, min(box[3], candidate[3]) - max(box[1], candidate[1]))
        other_area = max(0, candidate[2] - candidate[0]) * max(0, candidate[3] - candidate[1])
        if intersection / max(min(area, other_area), 1e-6) > 0.35:
            return False, 'overlapping_people_uncertain'
    return True, 'ava_and_motion_supported'


def running_evidence(reference, current, shape, *, allow_clipped=False):
    height, width = shape[:2]
    for box in (reference, current):
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            return False, "invalid_box"
        left, top, right, bottom = box
        box_height, box_width = bottom - top, right - left
        if box_height <= 0 or box_width <= 0:
            return False, "invalid_box"
        clipped = (top <= height * 0.02 or bottom >= height * 0.98
                   or left <= width * 0.01 or right >= width * 0.99)
        if clipped and box_height > height * 0.45 and not allow_clipped:
            return False, "close_up_or_clipped_body"
        if box_height / box_width < 1.25:
            return False, "body_view_unsuitable"
    old_height = reference[3] - reference[1]
    new_height = current[3] - current[1]
    shift = math.hypot((current[0] + current[2] - reference[0] - reference[2]) / 2,
                       (current[1] + current[3] - reference[1] - reference[3]) / 2)
    displacement = shift / max(old_height, new_height)
    scale_change = abs(math.log(new_height / old_height))
    if displacement < 0.12 and scale_change < 0.12:
        return False, "no_sustained_body_displacement"
    return True, "supported_motion"


def adaptive_running_evidence(person, people, shape):
    raw = person.get('running_score', 0.)
    ava = person.get('running_ava_score', person.get('ava_running_score', 0.))
    grab = person.get('running_grab_score', person.get('ava_grab_score', 0.))
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in (raw, ava, grab)):
        return False, 'invalid_running_scores', False
    depth = person.get('running_depth_support') is True and raw >= .75 and ava >= .3
    strong = (raw >= .85 and ava >= .25) or depth
    reference = person.get('running_action_box', person.get('action_box'))
    if reference is None:
        return False, 'uncertain_person_match', False
    valid, reason = running_evidence(reference, person['box'], shape,
                                     allow_clipped=strong and ava >= .6)
    if not valid:
        return False, reason, False
    if ava < .4 and not strong:
        return False, 'no_ava_running_confirmation', False
    if grab >= ava:
        return False, 'interaction_not_running', False
    box = person['box']
    area = (box[2] - box[0]) * (box[3] - box[1])
    for other in people:
        if other is person:
            continue
        candidate = other['box']
        intersection = max(0, min(box[2], candidate[2]) - max(box[0], candidate[0])) * max(0, min(box[3], candidate[3]) - max(box[1], candidate[1]))
        other_area = max(0, candidate[2] - candidate[0]) * max(0, candidate[3] - candidate[1])
        if intersection / max(min(area, other_area), 1e-6) > .35:
            return False, 'overlapping_people_uncertain', False
    return True, ('depth_path_and_model_agreement' if depth else
                  'strong_head_and_ava_motion' if strong else 'ava_and_motion_supported'), strong
