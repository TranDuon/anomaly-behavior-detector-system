"""Build render-only geometry from observations with authentic source times."""

from mmaction2_demo.action_engine import match_display_people, remap_display_regions
from mmaction2_demo.running_evidence import running_appearance


def needs_small_person_display(raw, posture, shape, states):
    if posture['boxes'] or any(states.get(name) in {'fighting', 'verifying'} for name in ('strike', 'grapple', 'fall')):
        return False
    boxes = raw.get('person_boxes', [])
    height, width = shape[:2]
    return bool(boxes) and all(0 < box[3] - box[1] <= .3 * height
                               and 0 < box[2] - box[0] <= .15 * width for box in boxes)


def display_geometry(raw, observed, reference_frame, observation_frame, active, states):
    boxes = observed['person_boxes']
    original = raw.get('person_boxes', [])
    assignments = match_display_people(reference_frame, observation_frame, original, boxes)
    actions = [{'box': box, 'labels': []} for box in boxes]
    for before, after in assignments.items():
        if before < len(raw.get('actions', [])):
            actions[after] = {**raw['actions'][before], 'box': boxes[after]}
    regions = []
    for name, values in raw.get('action_regions', {}).items():
        if name in active or states.get(name) == 'verifying':
            regions.extend(remap_display_regions(original, boxes, assignments, values))
    return {'person_boxes': boxes, 'actions': actions, 'regions': regions,
            'box_timestamp': observed['timestamp'],
            'appearances': [running_appearance(observation_frame, box) for box in boxes]}
