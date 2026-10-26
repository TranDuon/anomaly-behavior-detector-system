import numpy as np

from scripts.fight_ablation import row_id
from scripts.fight_people import (
    PeoplePolicy,
    coverage,
    decide_domain,
    frame_people,
    select_people_rows,
)


def test_people_policy_requires_two_visible_people_in_two_distinct_frames():
    policy = PeoplePolicy()
    assert decide_domain([2, 0, 0], [2, 0, 0], policy) == "small_or_unstable_people"
    assert decide_domain([2, 3, 1], [2, 3, 1], policy) == "people_interaction"
    assert decide_domain([1, 1, 1], [1, 1, 1], policy) == "single_person_control"
    assert decide_domain([0, 0, 0], [0, 0, 0], policy) == "no_detected_people"


def test_distant_people_and_invalid_boxes_do_not_satisfy_demo_domain():
    boxes = [
        [5, 5, 15, 15],
        [10, 10, 25, 70],
        [-20, 0, 12, 80],
        [1, 2, np.nan, 70],
        [40, 50, 20, 10],
    ]
    assert len(frame_people(boxes, 100, 100, PeoplePolicy())) == 2


def test_domain_decision_has_no_label_or_prediction_input():
    policy = PeoplePolicy()
    assert decide_domain([3, 2, 2], [3, 2, 2], policy) == "people_interaction"
    assert decide_domain([0, 0, 0], [7, 8, 9], policy) == "small_or_unstable_people"


def test_people_cohort_retains_normal_crowds_and_keeps_split():
    rows = [
        {
            "video": f"video{index}",
            "start": 0,
            "label": index % 2,
            "split": "train" if index < 20 else "validation",
        }
        for index in range(28)
    ]
    audits = {row_id(row): {"domain": "people_interaction"} for row in rows}
    train, validation = select_people_rows(rows, audits, PeoplePolicy())
    assert len(train) == 20 and len(validation) == 8
    assert sum(row["label"] == 0 for row in validation) == 4
    assert all(row["split"] == "train" for row in train)
    assert all(row["split"] == "validation" for row in validation)
    result = coverage(rows, audits)
    assert result["validation"]["0"]["domains"]["people_interaction"] == 4


def test_single_person_control_quota_is_train_only_and_never_positive():
    main = [
        {"video": f"main{index}", "start": 0, "label": index % 2, "split": "train"}
        for index in range(40)
    ]
    candidates = [
        {
            "video": f"control{index}",
            "start": 0,
            "label": index % 2,
            "split": "validation" if index < 4 else "train",
        }
        for index in range(30)
    ]
    audits = {
        row_id(row): {
            "domain": "people_interaction" if row in main else "single_person_control"
        }
        for row in main + candidates
    }
    train, validation = select_people_rows(main + candidates, audits, PeoplePolicy())
    added = [row for row in train if row["video"].startswith("control")]
    assert added and len(added) / len(train) <= 0.15
    assert all(row["label"] == 0 and row["split"] == "train" for row in added)
    assert not validation
