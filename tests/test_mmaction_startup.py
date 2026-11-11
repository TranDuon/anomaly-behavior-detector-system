import io
import json
from types import SimpleNamespace

import numpy as np

from fight.mmaction_runtime import MMActionRuntime


def test_startup_detects_people_without_fabricating_action_scores(monkeypatch):
    runtime = MMActionRuntime(three_actions=True, allow_pilot=True)
    runtime.model_ready = True
    calls = []

    def detect(frame, timestamp):
        calls.append(timestamp)
        runtime.last_startup_detection = timestamp
        runtime.result.update(people=1, person_boxes=[[1, 2, 20, 40]], box_timestamp=timestamp)

    monkeypatch.setattr(runtime, 'detect_startup', detect)
    frame = np.zeros((48, 64, 3), np.uint8)
    output = runtime.push(frame, 0.)
    assert output['people'] == 1
    assert output['state'] == 'warming_up'
    assert output['action_scores'] == {} and output['score'] is None
    assert output['prediction_timestamp'] is None
    runtime.push(frame, .27)
    runtime.push(frame, .54)
    assert calls == [0., .54]
    assert output['collection']['samples'] == 1
    runtime.reset()
    assert runtime.last_startup_detection is None


def test_detector_ipc_is_explicit_and_does_not_run_action_gate(monkeypatch):
    runtime = MMActionRuntime(allow_pilot=True)
    runtime.process = type('Process', (), {'stdin': io.StringIO()})()
    runtime.replies.put({'result': {'people': 1, 'person_boxes': [[0, 0, 4, 8]],
                                   'actions': [{'box': [0, 0, 4, 8], 'labels': []}]}})
    monkeypatch.setattr(runtime, 'warmup', lambda: None)
    try:
        runtime.detect_startup(np.zeros((12, 16, 3), np.uint8), 0.)
        assert json.loads(runtime.process.stdin.getvalue())['op'] == 'detect'
        assert runtime.result['prediction_timestamp'] is None
        assert runtime.result['people'] == 1
        assert not runtime.gate.active
    finally:
        runtime.frame_buffer.close()


def test_completed_runtime_is_reused_but_session_state_is_reset(monkeypatch):
    from api import fight_server
    from fight.runtime import EventGate
    calls = []
    runtime = SimpleNamespace(backend_name='mmaction_separate_actions', model_ready=True,
                              process=SimpleNamespace(poll=lambda: None),
                              gate=EventGate(.5, .35), metadata={}, message='pilot',
                              reset=lambda: calls.append('reset'), warmup=lambda: calls.append('warmup'),
                              close=lambda: calls.append('close'))
    monkeypatch.setattr(fight_server, 'FightRuntime', lambda *args, **kwargs: runtime)
    processor = fight_server.FightProcessor()
    processor.events = [{'id': 'previous-session'}]

    class Thread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            calls.append('thread')

        def is_alive(self):
            return False

        def join(self, **kwargs):
            pass

    monkeypatch.setattr(fight_server.threading, 'Thread', Thread)
    processor.start('public-fixture', allow_pilot=True, backend='mmaction_separate_actions')
    assert processor.runtime is runtime
    assert calls == ['reset', 'warmup', 'thread']
    assert processor.events == [] and processor.session_id
    processor.stop()
    assert calls[-1] == 'close'
