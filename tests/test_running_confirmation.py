from fight.running_confirmation import RunningConfirmation
from mmaction2_demo.running_evidence import adaptive_running_evidence


def person(left=0, strong=True, supported=True, appearance=None):
    return {'box': [left, 20, left + 40, 140], 'action_box': [left, 20, left + 40, 140],
            'running_supported': supported, 'running_fast_support': strong,
            'running_effective_score': .95 if supported else 0.,
            'running_evidence': 'strong_head_and_ava_motion' if supported else 'uncertain_person_match',
            'running_appearance': appearance}


def test_same_strong_runner_confirms_with_two_real_observations():
    policy = RunningConfirmation()
    assert policy.update([person()], 0., .16)[0] == 'verifying'
    assert policy.update([person()], .533, .16)[0] == 'fighting'


def test_alternating_people_cannot_pool_their_scores():
    policy = RunningConfirmation()
    policy.update([person()], 0., .16)
    assert policy.update([person(left=800)], .533, .16)[0] == 'verifying'


def test_motion_and_appearance_keep_fast_runner_without_box_overlap():
    policy = RunningConfirmation()
    policy.update([person(0, appearance=[1., 0.])], 0., .16)
    next_person = person(120, appearance=[1., 0.])
    assert policy.update([next_person], .533, .16)[0] == 'fighting'
    assert next_person['running_track_id'] == 1


def test_single_missing_observation_does_not_draw_a_ghost_or_erase_pending():
    policy = RunningConfirmation()
    policy.update([person()], 0., .16)
    assert policy.update([], .533, .16)[:2] == ('normal', [])
    assert policy.update([person()], 1.067, .16)[0] == 'fighting'
    assert policy.update([person()], 2.4, .16)[0] == 'verifying'


def test_walking_evidence_resets_pending_and_weak_candidates_stay_slower():
    policy = RunningConfirmation()
    policy.update([person()], 0., .16)
    walking = person(supported=False)
    walking['running_evidence'] = 'no_sustained_body_displacement'
    policy.update([walking], .533, .16)
    assert policy.update([person()], 1.067, .16)[0] == 'verifying'
    policy = RunningConfirmation()
    policy.update([person(strong=False)], 0., .16)
    assert policy.update([person(strong=False)], .533, .16)[0] == 'verifying'


def test_adaptive_evidence_accepts_agreement_not_pilot_score_alone():
    candidate = {'box': [233, 259, 281, 363], 'action_box': [450, 266, 508, 384],
                 'running_score': .97, 'ava_running_score': .399, 'ava_grab_score': .01}
    assert adaptive_running_evidence(candidate, [candidate], (540, 960, 3)) == (True, 'strong_head_and_ava_motion', True)
    candidate['ava_running_score'] = .02
    assert not adaptive_running_evidence(candidate, [candidate], (540, 960, 3))[0]


def test_clipped_moving_runner_requires_strong_ava_and_stationary_pose_is_rejected():
    candidate = {'box': [649, 208, 948, 707], 'action_box': [182, 299, 367, 648],
                 'running_score': .93, 'ava_running_score': .72, 'ava_grab_score': .01}
    assert adaptive_running_evidence(candidate, [candidate], (720, 1280, 3))[0]
    candidate['action_box'] = candidate['box']
    assert not adaptive_running_evidence(candidate, [candidate], (720, 1280, 3))[0]
