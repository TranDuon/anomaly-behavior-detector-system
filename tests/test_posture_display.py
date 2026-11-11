from fight.posture_display import canonical_posture_view


def test_pose_alert_uses_exact_display_box_and_no_duplicate_detector():
    old = [20, 313, 400, 460]
    pose = [25, 304, 395, 452]
    output = {'person_boxes': [old], 'actions': [{'box': old, 'labels': []}],
              'box_timestamp': 2., 'action_regions': {'strike': [], 'fall': [old]}}
    posture = {'boxes': [pose], 'confirmed': [pose], 'candidates': [pose]}
    boxes, actions, regions = canonical_posture_view(output, posture, 2.)
    assert boxes == [pose]
    assert regions['fall'] == boxes
    assert actions[0]['box'] == pose


def test_expired_detector_location_does_not_count_as_an_extra_person():
    output = {'person_boxes': [[0, 0, 40, 100]], 'box_timestamp': 0., 'action_regions': {}}
    posture = {'boxes': [[100, 0, 140, 100]], 'confirmed': [], 'candidates': []}
    assert canonical_posture_view(output, posture, 1.)[0] == []


def test_pose_inventory_does_not_erase_an_rgb_person_with_hidden_hips():
    output = {'person_boxes': [[0, 0, 40, 100], [60, 0, 160, 30]], 'box_timestamp': 0., 'action_regions': {}}
    posture = {'boxes': [[60, 0, 160, 30]], 'confirmed': [], 'candidates': []}
    assert len(canonical_posture_view(output, posture, 0.)[0]) == 2


def test_empty_full_body_pose_keeps_close_up_rgb_strike_boxes():
    box = [0, 0, 100, 100]
    output = {'person_boxes': [box], 'box_timestamp': 2.,
              'actions': [{'box': box, 'fight_score': .8}], 'action_regions': {'strike': [box]}}
    posture = {'boxes': [], 'confirmed': [], 'candidates': []}
    boxes, actions, regions = canonical_posture_view(output, posture, 2.27)
    assert boxes == [box] and regions['strike'] == [box]
    assert actions[0]['fight_score'] == .8


def test_bystander_does_not_receive_lying_highlight():
    person, bystander = [0, 0, 100, 30], [150, 0, 180, 100]
    posture = {'boxes': [person, bystander], 'confirmed': [person], 'candidates': [person]}
    boxes, _, regions = canonical_posture_view({'person_boxes': [bystander], 'box_timestamp': 0.,
                                                'action_regions': {}}, posture, 0.)
    assert len(boxes) == 2
    assert regions['fall'] == [person]


def test_upright_pose_does_not_add_a_second_box_behind_running_person():
    current, trailing = [40, 0, 70, 90], [85, 0, 115, 90]
    posture = {'boxes': [current], 'candidates': [], 'confirmed': []}
    output = {'person_boxes': [trailing], 'box_timestamp': 0., 'action_regions': {}}
    assert canonical_posture_view(output, posture, 0.)[0] == [current]


def test_pose_relocation_keeps_running_region_on_matched_person_only():
    runner, bystander = [90, 0, 120, 90], [250, 0, 280, 90]
    current = [45, 0, 75, 90]
    output = {'person_boxes': [runner, bystander], 'box_timestamp': 0.,
              'actions': [{'box': runner, 'labels': []}, {'box': bystander, 'labels': []}],
              'action_regions': {'fast_running': [runner]}}
    posture = {'boxes': [current, bystander], 'candidates': [], 'confirmed': []}
    boxes, _, regions = canonical_posture_view(output, posture, 0.)
    assert boxes == [current, bystander]
    assert regions['fast_running'] == [current]


def test_confirmed_lying_box_is_drawn_red_not_green():
    import numpy as np
    from api.fight_server import annotate
    box = [20, 70, 150, 110]
    result = {'state': 'fighting', 'person_boxes': [box], 'regions': [box],
              'actions': [], 'active_behaviors': ['fall'], 'behavior_scope': 'three_actions'}
    image = annotate(np.zeros((140, 200, 3), np.uint8), result)
    assert image[90, 20].tolist() == [50, 65, 235]
