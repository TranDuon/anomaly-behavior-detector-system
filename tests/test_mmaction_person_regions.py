from fight.mmaction_runtime import person_action_regions


def test_alarm_highlights_only_people_with_fight_action():
    people = [
        {"box": [0, 0, 10, 20], "fight_score": 0.8},
        {"box": [20, 0, 30, 20], "fight_score": 0.01},
        {"box": [40, 0, 50, 20], "fight_score": 0.02},
    ]
    assert person_action_regions(people, 0.4) == [[0, 0, 10, 20]]
    assert person_action_regions(people, 0.9) == []
    assert person_action_regions(people, 0.015) == [
        [0, 0, 10, 20], [40, 0, 50, 20]
    ]


def test_no_person_or_invalid_score_does_not_assign_a_victim():
    assert person_action_regions([], 0.4) == []
    assert person_action_regions([
        {"box": [0, 0, 10, 20], "fight_score": float("nan")},
        {"box": [20, 0, 30, 20]},
    ], 0.4) == []
