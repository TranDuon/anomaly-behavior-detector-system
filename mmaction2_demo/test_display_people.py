import numpy as np

from action_engine import match_display_people, remap_display_regions


def test_current_boxes_match_by_appearance_without_coloring_bystanders():
    old = np.zeros((100, 200, 3), np.uint8)
    new = np.zeros_like(old)
    boxes = [[10, 20, 40, 80], [100, 20, 130, 80]]
    current = [[80, 20, 110, 80], [20, 20, 50, 80]]
    old[20:80, 10:40] = (0, 0, 255)
    old[20:80, 100:130] = (255, 0, 0)
    new[20:80, 80:110] = (0, 0, 255)
    new[20:80, 20:50] = (255, 0, 0)
    assignments = match_display_people(old, new, boxes, current)
    assert assignments == {0: 0, 1: 1}
    assert remap_display_regions(boxes, current, assignments, [boxes[0]]) == [current[0]]


def test_ambiguous_people_do_not_inherit_somebody_elses_action():
    frame = np.full((100, 200, 3), 100, np.uint8)
    assert match_display_people(frame, frame, [[80, 20, 110, 80]],
                                [[60, 20, 90, 80], [100, 20, 130, 80]]) == {}
    assert remap_display_regions([[10, 10, 20, 20]], [], {}, [[10, 10, 20, 20]]) == []
