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
