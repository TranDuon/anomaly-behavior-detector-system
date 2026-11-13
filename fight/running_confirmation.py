"""Confirm running on the same visible person, not pooled scene scores."""

from collections import deque
import math

from fight.runtime import EventGate


def intersection_over_union(first, second):
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(0, min(first[3], second[3]) - max(first[1], second[1]))
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (first, second)]
    return intersection / max(sum(areas) - intersection, 1e-6)


class RunningConfirmation:
    def __init__(self, maximum_gap=1.15):
        self.maximum_gap = maximum_gap
        self.tracks = {}
        self.next_id = 1
        self.last_time = None

    def update(self, people, timestamp, threshold):
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp < self.last_time):
            raise ValueError('Running confirmation requires monotonic timestamps')
        if timestamp == self.last_time:
            raise ValueError('Repeated observations cannot advance running confirmation')
        self.last_time = timestamp
        self.tracks = {identity: track for identity, track in self.tracks.items()
                       if timestamp - track['time'] <= self.maximum_gap}
        proposals = []
        for index, person in enumerate(people):
            reference = person.get('running_action_box', person.get('action_box'))
            choices = []
            for identity, track in self.tracks.items():
                historical = [box for observed, box in track['history'] if timestamp - observed <= self.maximum_gap]
                reference_match = max((intersection_over_union(reference, box) for box in historical), default=0.) if reference is not None else 0.
                current_match = intersection_over_union(person['box'], track['box'])
                appearance = person.get('running_appearance')
                prior = track.get('appearance')
                affinity = 0.
                if appearance and prior and len(appearance) == len(prior):
                    affinity = sum(math.sqrt(max(0., first * second)) for first, second in zip(appearance, prior))
                box, old = person['box'], track['box']
                scale = max(box[3] - box[1], old[3] - old[1], 1.)
                distance = math.hypot((box[0] + box[2] - old[0] - old[2]) / 2,
                                      (box[1] + box[3] - old[1] - old[3]) / 2) / scale
                appearance_match = affinity >= .8 and distance <= 2.5
                if reference_match >= .35 or current_match >= .35 or appearance_match:
                    choices.append((reference_match * 2 + current_match + (affinity * 2.5 - distance * .2 if appearance_match else 0), index, identity))
            choices.sort(reverse=True)
            if choices and (len(choices) == 1 or choices[0][0] - choices[1][0] >= .2):
                proposals.append(choices[0])
        assignments, used = {}, set()
        for _, index, identity in sorted(proposals, reverse=True):
            if index not in assignments and identity not in used:
                assignments[index] = identity
                used.add(identity)
        regions, confirmed_regions, states, diagnostics = [], [], [], []
        for index, person in enumerate(people):
            supported = person.get('running_supported', False)
            value = person.get('running_effective_score', 0.) if supported else 0.
            identity = assignments.get(index)
            if identity is None:
                identity = self.next_id
                self.next_id += 1
                self.tracks[identity] = {'gate': EventGate(threshold, threshold * .7, confirm_seconds=.8,
                                                         release_seconds=.25, max_observation_gap=self.maximum_gap),
                                         'history': deque(maxlen=6), 'time': timestamp, 'box': person['box'],
                                         'last_supported': None}
            track = self.tracks[identity]
            person['running_track_id'] = identity
            track.update(time=timestamp, box=person['box'], appearance=person.get('running_appearance'))
            track['history'].append((timestamp, person['box']))
            if track['last_supported'] is not None and timestamp - track['last_supported'] > self.maximum_gap:
                track['gate'] = track['gate'].fresh()
            if not supported:
                if person.get('running_evidence') != 'uncertain_person_match':
                    track['gate'] = track['gate'].fresh()
                continue
            track['gate'].confirm_seconds = .4 if person.get('running_fast_support') else .8
            state, _ = track['gate'].update(value, timestamp)
            track['last_supported'] = timestamp
            states.append(state)
            if state in {'fighting', 'verifying'}:
                regions.append(person['box'])
            if state == 'fighting':
                confirmed_regions.append(person['box'])
            diagnostics.append({'track_id': identity, 'state': state, 'score': value,
                                'confirm_seconds': track['gate'].confirm_seconds})
        if len(self.tracks) > 32:
            retained = sorted(self.tracks, key=lambda identity: self.tracks[identity]['time'], reverse=True)[:32]
            self.tracks = {identity: self.tracks[identity] for identity in retained}
        state = 'fighting' if 'fighting' in states else 'verifying' if 'verifying' in states else 'normal'
        return state, confirmed_regions if state == 'fighting' else regions, diagnostics
