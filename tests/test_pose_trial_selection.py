import copy
import importlib.util
from pathlib import Path
import sys

import pytest


def selector(monkeypatch):
    root = Path(__file__).resolve().parents[1] / 'mmaction2_demo'
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location('review_pose_trial', root / 'review_pose_specialists.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.select_per_label


def reports():
    result = {}
    for stage, grapple, fall in [('frozen', .7, .8), ('tuned', .95, .5)]:
        result[stage] = {'status': 'completed', 'development': {
            label: {'ap': value, 'f1': .5, 'precision': .8, 'recall': .4}
            for label, value in [('grapple', grapple), ('fall', fall)]}}
    return result


def test_independent_selection_never_sacrifices_fall_for_macro_ap(monkeypatch):
    select = selector(monkeypatch)
    assert select(reports()) == {'grapple': 'tuned', 'fall': 'frozen'}


def test_selection_does_not_read_test_results(monkeypatch):
    select = selector(monkeypatch)
    original = reports()
    changed = copy.deepcopy(original)
    changed['frozen']['test'] = {'grapple': {'ap': 1.}, 'fall': {'ap': 0.}}
    assert select(original) == select(changed)
    changed['tuned']['development']['grapple']['ap'] = float('nan')
    with pytest.raises(ValueError, match='Invalid'):
        select(changed)
