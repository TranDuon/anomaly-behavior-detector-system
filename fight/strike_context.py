"""Group-A context for a strike: is someone being hit, or is an object being hit?

Read-only: uses the person boxes and strike regions the core already produced and
never changes the strike branch. Each core prediction gets a verdict

    'fight'      a person next to the striker, who reacts (moves away / falls)
    'vandalism'  nobody to hit: alone in view, nobody next to the striker,
                 nearest person far away, or the nearest person does not react
    'uncertain'  someone is next to the striker but there is not enough history

Signals (see the plan, group A):
  A1 only one person in view
  A2 no full-size person next to the striker (has_target_person)
  A3 nearest other person farther than 1.5 striker widths
  A4 nearest other person static while the strike lasts
  A5 nearest other person reacts: moves >= 15 % of their height, or the fall branch
     is active on them
"""

from collections import deque

from fight.mmaction_runtime import has_target_person

FAR_FACTOR = 1.5
STATIC_SHIFT = 0.05
REACT_SHIFT = 0.15
HISTORY = 6  # core predictions (~1.6 s)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def _gap(a, b):
    return max(0., max(a[0], b[0]) - min(a[2], b[2])), max(0., max(a[1], b[1]) - min(a[3], b[3]))


def nearest_other(striker, people):
    others = [box for box in people if _iou(box, striker) < 0.5]
    if not others:
        return None
    return min(others, key=lambda box: sum(v * v for v in _gap(striker, box)) ** .5
               + 1e-3 * abs(_center(box)[0] - _center(striker)[0]))


class StrikeContext:
    """Tracks the nearest other person across core predictions to judge A4/A5."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.track = deque(maxlen=HISTORY)  # (timestamp, nearest box)

    def observe(self, output, timestamp):
        states = output.get('action_states', {})
        strikes = (output.get('action_regions') or {}).get('strike') or []
        if states.get('strike') not in {'fighting', 'verifying'} or not strikes:
            self.reset()
            return None
        people = output.get('person_boxes', [])
        striker = strikes[0]
        width, height = striker[2] - striker[0], striker[3] - striker[1]
        signals = {'A1_alone': len(people) <= 1, 'A2_no_target': not has_target_person(striker, people)}
        near = nearest_other(striker, people)
        if near is not None:
            gx, gy = _gap(striker, near)
            signals['A3_far'] = gx > FAR_FACTOR * width or gy > FAR_FACTOR * height
            self.track.append((timestamp, near))
        else:
            signals['A3_far'] = True
        shift = None
        if len(self.track) >= 2:
            (_, first), (_, last) = self.track[0], self.track[-1]
            ref = max(1., first[3] - first[1])
            dx, dy = _center(last)[0] - _center(first)[0], _center(last)[1] - _center(first)[1]
            shift = (dx * dx + dy * dy) ** .5 / ref
        fall_on_near = near is not None and any(_iou(near, box) > 0.3 for box in
                                                (output.get('action_regions') or {}).get('fall') or [])
        signals['A4_static'] = shift is not None and shift < STATIC_SHIFT
        signals['A5_reacts'] = fall_on_near or (shift is not None and shift >= REACT_SHIFT)
        if signals['A5_reacts']:
            verdict = 'fight'  # strongest evidence: the person next to the striker was pushed / fell
        elif signals['A1_alone'] or signals['A2_no_target'] or signals['A3_far']:
            verdict = 'vandalism'
        elif signals['A4_static'] and len(self.track) >= 3:
            verdict = 'vandalism'
        else:
            verdict = 'uncertain'
        return {'verdict': verdict, 'signals': signals,
                'nearest_shift': None if shift is None else round(shift, 3)}
