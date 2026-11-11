from mmaction2_demo.running_evidence import running_evidence, verified_running_evidence


def test_stationary_full_body_is_not_running():
    box = [200, 100, 260, 300]
    assert running_evidence(box, box, (540, 960, 3)) == (False, "no_sustained_body_displacement")
    assert not running_evidence(box, [203, 102, 264, 304], (540, 960, 3))[0]


def test_close_up_standing_and_gestures_are_not_running():
    assert running_evidence([330, 116, 670, 716], [350, 110, 710, 720], (720, 1280, 3)) == (False, "close_up_or_clipped_body")


def test_visible_translating_runner_or_approaching_runner_is_supported():
    assert running_evidence([450, 266, 508, 384], [233, 259, 281, 363], (540, 960, 3))[0]
    assert running_evidence([400, 200, 440, 300], [395, 180, 447, 310], (540, 960, 3))[0]


def test_invalid_or_upper_body_boxes_are_not_running():
    assert not running_evidence([0, 0, float("nan"), 20], [1, 1, 5, 20], (100, 100))[0]
    assert not running_evidence([20, 20, 180, 120], [30, 20, 190, 120], (300, 400))[0]


def runner(ava_score=0.8):
    return {'action_box': [450, 266, 508, 384], 'box': [233, 259, 281, 363],
            'ava_running_score': ava_score, 'ava_grab_score': 0.01}


def test_motion_alone_does_not_confirm_running():
    person = runner(0.2)
    assert verified_running_evidence(person, [person], (540, 960, 3)) == (False, 'no_ava_running_confirmation')


def test_ava_run_with_clear_motion_is_supported():
    person = runner()
    assert verified_running_evidence(person, [person], (540, 960, 3))[0]


def test_overlapping_people_and_grabbing_are_not_confirmed_as_running():
    person = runner()
    assert verified_running_evidence(person, [person, {'box': [240, 260, 290, 370]}], (540, 960, 3)) == (False, 'overlapping_people_uncertain')
    person['ava_grab_score'] = 0.9
    assert verified_running_evidence(person, [person], (540, 960, 3)) == (False, 'interaction_not_running')
