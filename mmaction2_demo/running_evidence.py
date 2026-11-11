"""Conservative applicability checks for the outdoor running pilot."""

import math


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


def running_evidence(reference, current, shape):
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
        if clipped and box_height > height * 0.45:
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
