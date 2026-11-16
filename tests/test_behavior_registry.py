"""Pin the core registry to the values that used to be hard-coded."""

import numpy as np
import pytest

from fight import behaviors
from fight.behaviors import BehaviorSpec


def test_core_values_match_previous_hard_coded_constants():
    # api/fight_server.annotate label map
    assert {name: behaviors.display_label(name) for name in behaviors.CORE_NAMES} == {
        "strike": "STRIKE", "grapple": "GRAPPLE", "fall": "FALL", "fast_running": "RUNNING"}
    # api/fight_server.record label tuple (order matters for event order)
    assert behaviors.recorded_names()[:4] == ("strike", "grapple", "fall", "fast_running")
    # fight/mmaction_runtime.push_separate reset loop
    assert behaviors.reset_without_region_names() == ("strike", "grapple", "fast_running")
    # fight/display_geometry.needs_small_person_display
    assert behaviors.small_person_blocking_names() == ("strike", "grapple", "fall")
    # fight/mmaction_runtime.warmup gate timings
    assert behaviors.gate_timing("fast_running") == (0.8, 0.25)
    for name in ("strike", "grapple", "fall"):
        assert behaviors.gate_timing(name) == (0.4, 0.25)


def test_extensions_append_after_core_and_cannot_override_core():
    spec = BehaviorSpec("test_ext", "TEST EXT", kind="extension")
    try:
        behaviors.register(spec)
        assert behaviors.recorded_names()[:4] == behaviors.CORE_NAMES
        assert behaviors.recorded_names()[-1] == "test_ext"
        # Core helper sets are computed from core entries only.
        assert "test_ext" not in behaviors.reset_without_region_names()
        assert "test_ext" not in behaviors.small_person_blocking_names()
        with pytest.raises(ValueError):
            behaviors.register(BehaviorSpec("strike", "X", kind="extension"))
        with pytest.raises(ValueError):
            behaviors.register(BehaviorSpec("new_core", "X"))
        with pytest.raises(ValueError):
            behaviors.unregister("fall")
    finally:
        behaviors.unregister("test_ext")
    assert behaviors.recorded_names() == behaviors.CORE_NAMES


def test_unknown_label_is_displayed_instead_of_crashing():
    assert behaviors.display_label("someday_behavior") == "SOMEDAY_BEHAVIOR"
    from api.fight_server import annotate

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = {"state": "fighting", "behavior_scope": "three_actions",
              "active_behaviors": ["strike", "someday_behavior"], "person_boxes": [], "actions": []}
    assert annotate(frame, result).shape == frame.shape
