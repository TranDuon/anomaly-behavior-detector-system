import copy

import numpy as np

from mmaction2_demo.running_evidence import (
    adaptive_running_evidence,
    depth_motion_path,
    recover_running_matches,
    restore_running_fields,
    match_running_step,
    running_crop_fragment,
)
from fight.running_confirmation import RunningConfirmation


def observations():
    original = [[100, 100, 140, 240]]
    current = [[60, 20, 180, 400]]
    actions = [{'box': original[0], 'running_score': .96, 'ava_running_score': .72,
                'ava_grab_score': .01, 'fight_score': .9, 'labels': ['fight']}]
    return original, current, actions


def detector(frame):
    return np.asarray([[100 - frame * 2, 100 - frame * 3, 140 + frame * 2, 240 + frame * 10]])


def match(previous_frame, current_frame, previous, current):
    return {0: 0}


def test_running_can_recover_only_through_complete_intermediate_chain():
    original, current, actions = observations()
    assert recover_running_matches(list(range(9)), original, current, actions, {}, detector, match) == {0: 0}
    def missing(previous_frame, current_frame, previous, current):
        return {} if current_frame == 6 else {0: 0}
    assert recover_running_matches(list(range(9)), original, current, actions, {}, detector, missing) == {}


def test_no_extra_detection_for_weak_scores_or_already_matched_runner():
    original, current, actions = observations()
    def forbidden(frame):
        raise AssertionError('Unnecessary detector inference')
    assert recover_running_matches(list(range(9)), original, current, actions, {0: 0}, forbidden, match) == {}
    actions[0]['ava_running_score'] = .02
    assert recover_running_matches(list(range(9)), original, current, actions, {}, forbidden, match) == {}


def test_recovery_never_steals_an_existing_person_assignment():
    original, current, actions = observations()
    assert recover_running_matches(list(range(9)), original, current, actions, {1: 0}, detector, match) == {}


def test_recovery_changes_only_running_fields_and_leaves_source_untouched():
    original, current, actions = observations()
    source = copy.deepcopy(actions)
    person = {'box': current[0], 'fight_score': 0., 'labels': [], 'ava_lying_score': .2}
    display = {'person_boxes': current, 'actions': [copy.deepcopy(person)],
               'action_scores': {'strike': .9, 'fall': .2, 'grapple': .3},
               'action_regions': {'strike': [], 'fall': [], 'grapple': []}}
    before = copy.deepcopy(display)
    restore_running_fields(list(range(9)), original, actions, display, {}, detector, match)
    recovered = display['actions'][0]
    assert {key: recovered[key] for key in person} == person
    assert display['action_scores'] == before['action_scores']
    assert display['action_regions'] == before['action_regions']
    assert actions == source
    valid, reason, strong = adaptive_running_evidence(recovered, [recovered], (500, 500, 3))
    assert valid and strong
    recovered.update(running_supported=True, running_fast_support=True, running_effective_score=.96)
    confirmation = RunningConfirmation()
    assert confirmation.update([recovered], 0., .16)[0] == 'verifying'
    assert confirmation.update([recovered], .533, .16)[0] == 'fighting'


def test_depth_path_accepts_approaching_and_receding_not_detector_size_jumps():
    boxes = [[100, 100, 140, 100 + height] for height in (100, 110, 125, 145, 170)]
    assert depth_motion_path(boxes)
    assert depth_motion_path(boxes[::-1])
    assert not depth_motion_path([boxes[0]] * 5)
    assert not depth_motion_path([boxes[0], boxes[-1], boxes[0], boxes[-1], boxes[0]])
    assert not depth_motion_path(boxes[:3])


def test_depth_support_requires_continuous_path_and_both_models_not_walking_motion_alone():
    original, current, actions = observations()
    actions[0]['running_score'] = .8
    actions[0]['ava_running_score'] = .35
    display = {'person_boxes': current, 'actions': [{'box': current[0]}]}
    restore_running_fields(list(range(9)), original, actions, display, {0: 0}, detector, match)
    person = display['actions'][0]
    assert person['running_depth_support']
    assert adaptive_running_evidence(person, [person], (500, 500, 3))[0]
    person['running_ava_score'] = .02
    assert not adaptive_running_evidence(person, [person], (500, 500, 3))[0]
    person['running_ava_score'] = .35
    person['running_depth_support'] = False
    assert not adaptive_running_evidence(person, [person], (500, 500, 3))[0]


def test_duplicate_torso_proposal_does_not_break_running_identity_or_change_detector_boxes():
    original = [[100, 50, 150, 200]]
    current = [[100, 40, 150, 240], [100, 41, 150, 160]]
    first = np.zeros((300, 300, 3), np.uint8)
    second = first.copy()
    first[50:200, 100:150] = [0, 180, 0]
    second[40:240, 100:150] = [0, 180, 0]
    before = copy.deepcopy(current)
    assert running_crop_fragment(current[1], current[0])
    assert not running_crop_fragment([170, 40, 220, 160], current[0])
    assert match_running_step(first, second, original, current, lambda *args: {}) == {0: 0}
    assert current == before


def test_similar_clothed_distinct_people_remain_ambiguous_not_forced_to_match():
    frame = np.zeros((300, 300, 3), np.uint8)
    frame[:, :] = [0, 180, 0]
    assert match_running_step(frame, frame, [[100, 50, 150, 200]],
                              [[70, 50, 120, 200], [130, 50, 180, 200]], lambda *args: {}) == {}
