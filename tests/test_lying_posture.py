import numpy as np

from fight.lying_posture import lying_posture, LyingPosture


def pose(shoulder, hip, knee, ankle):
    points = np.zeros((17, 3), np.float32)
    points[[5, 11, 13, 15], :2] = [shoulder, hip, knee, ankle]
    points[[5, 11, 13, 15], 2] = .9
    points[6] = points[5] + [0, 2, 0]
    points[12] = points[11] + [0, 2, 0]
    return points


def test_horizontal_body_is_lying_but_upright_sitting_and_bending_are_not():
    assert lying_posture(pose((10, 10), (50, 12), (80, 14), (110, 14)))[0]
    assert not lying_posture(pose((10, 10), (10, 50), (10, 80), (10, 110)))[0]
    assert not lying_posture(pose((10, 10), (10, 50), (50, 50), (50, 90)))[0]
    assert not lying_posture(pose((10, 50), (50, 50), (50, 80), (50, 110)))[0]


def test_missing_joints_and_nonfinite_pose_do_not_invent_lying():
    points = pose((10, 10), (50, 12), (80, 14), (110, 14))
    points[15, 2] = .05
    assert not lying_posture(points)[0]
    points[5, 0] = np.nan
    assert not lying_posture(points)[0]


def test_same_person_needs_two_actual_pose_observations():
    detector = LyingPosture.__new__(LyingPosture)
    record = {'box': np.array([0, 0, 130, 30]),
              'points': pose((10, 10), (50, 12), (80, 14), (110, 14))}
    detector.detector = type('Detector', (), {'poses': lambda self, frame: [record]})()
    detector.reset()
    assert not detector.observe(None, 0.)['confirmed']
    assert detector.observe(None, .267)['confirmed']
    assert not detector.observe(None, 2.)['confirmed']


def test_missing_full_frame_pose_recovers_from_detector_crop_without_using_box_as_lying():
    detector = LyingPosture.__new__(LyingPosture)
    points = pose((10, 10), (50, 12), (80, 14), (110, 14))
    class Detector:
        def poses(self, frame):
            if frame.shape[:2] == (200, 300):
                return []
            return [{'box': np.array([0., 0., 120., 30.]), 'points': points}]
    detector.detector = Detector()
    detector.reset()
    image = np.zeros((200, 300, 3), np.uint8)
    assert not detector.observe(image, 0.)['candidates']
    first = detector.observe(image, .27, [[20, 20, 160, 60]])
    assert first['candidates'] and not first['confirmed']
    assert detector.observe(image, .54, [[20, 20, 160, 60]])['confirmed']


def test_perspective_reclined_body_differs_from_extended_leg_sitting():
    lying = pose((679, 354), (658, 398), (609, 410), (550, 461))
    sitting = pose((794, 314), (753, 396), (712, 430), (663, 502))
    assert lying_posture(lying)[0]
    assert not lying_posture(sitting)[0]


def test_border_phantom_without_visible_shoulders_is_filtered():
    from fight.lying_posture import reliable_person
    points = pose((1278, 210), (1267, 293), (1254, 355), (1222, 418))
    points[[5, 6], 2] = .2
    assert not reliable_person(points)


def frontal_pose():
    points = np.zeros((17, 3), np.float32)
    points[[5, 6, 11, 12, 13, 14, 15, 16]] = [
        [583.52, 404.89, .99], [443.74, 422.60, .98],
        [578.05, 479.37, .99], [481.22, 483.07, .99],
        [561.45, 516.26, .99], [415.72, 497.39, .99],
        [623.14, 592.17, .97], [511.44, 546.28, .97],
    ]
    return points


def test_depth_aligned_lying_accepts_mirrored_scaled_and_oblique_views():
    for angle in (-25, 0, 25, 180):
        radians = np.deg2rad(angle)
        rotation = np.array([[np.cos(radians), -np.sin(radians)],
                             [np.sin(radians), np.cos(radians)]])
        for scale in (.5, 1., 2.):
            for mirror in (-1, 1):
                points = frontal_pose()
                points[:, :2] = (points[:, :2] * [mirror, 1]) @ rotation.T * scale + 30
                assert lying_posture(points)[0]


def test_front_sitting_missing_joints_and_folded_legs_are_not_lying():
    points = frontal_pose()
    points[[5, 6], :2] = [[565.22, 331.02], [444.44, 357.42]]
    assert not lying_posture(points)[0]
    points = frontal_pose()
    points[6, 2] = .1
    assert not lying_posture(points)[0]
    points = frontal_pose()
    points[[15, 16], :2] = points[[11, 12], :2] + [0, 5]
    assert not lying_posture(points)[0]


def test_frontal_lying_confirms_same_person_and_clears_after_sitting_up():
    policy = LyingPosture.__new__(LyingPosture)
    record = {'box': np.array([400., 380., 660., 620.]), 'points': frontal_pose()}
    policy.detector = type('Detector', (), {'poses': lambda self, frame: [record]})()
    policy.reset()
    assert policy.observe(None, 0.)['candidates']
    assert policy.observe(None, .267)['confirmed']
    record['points'][[5, 6], :2] = [[565.22, 331.02], [444.44, 357.42]]
    assert not policy.observe(None, .534)['confirmed']
