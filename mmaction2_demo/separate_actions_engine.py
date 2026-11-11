"""Compare independent original AVA actions with the separate grapple pilot."""

import json
import math

import torch

from action_engine import ActionEngine, SOURCE_SHA, SPEC, digest, match_display_people, remap_display_regions
from fixed_running_engine import FixedRunningEngine
from three_actions_engine import load_three_actions
from lying_policy import LYING_THRESHOLD, compatible_engine_hashes, lying_evidence


LABELS = ('strike', 'grapple', 'fall', 'fast_running')


class SeparateActionsEngine(FixedRunningEngine):
    def __init__(self, weights=None, device='cuda:0', person_recovery=True):
        super().__init__(weights, device, person_recovery)
        _, heads = load_three_actions(self.weights, device)
        self.grapple_head = heads[1]
        path = self.weights / 'separate_actions_calibration.json'
        if not path.is_file():
            raise ValueError('Separate-action calibration has not completed')
        calibration = json.loads(path.read_text(encoding='utf-8'))
        expected = {'source_sha256': SOURCE_SHA, 'spec': SPEC,
                    'grapple_checkpoint_sha256': digest(self.weights / 'three_actions.pt'),
                    'running_checkpoint_sha256': digest(self.weights / 'interaction_fixed_running.pt')}
        if any(calibration.get(key) != value for key, value in expected.items()):
            raise ValueError('Incompatible separate-action calibration')
        if calibration.get('engine_sha256') not in compatible_engine_hashes(self.weights.parent / 'action_engine.py'):
            raise ValueError('Incompatible action-feature engine')
        thresholds = calibration.get('thresholds', {})
        if set(thresholds) != set(LABELS) or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                                               or not math.isfinite(value) or not 0 < value < 1
                                               for value in thresholds.values()):
            raise ValueError('Invalid separate-action thresholds')
        self.metadata = {**self.metadata, **calibration, 'architecture': 'independent_ava_strike_lying_grapple_pilot_running',
                         'threshold': min(calibration['thresholds'].values()), 'confirm_seconds': .4,
                         'deployment': {'display_name': 'Bốn nhánh độc lập · AVA gốc + pilot'},
                         'approved_for_demo': False, 'real_phone_tested': False}
        self.metadata['thresholds'] = {**thresholds, 'fall': LYING_THRESHOLD}
        original = json.loads((self.weights.parent / 'config/original_calibration.json').read_text(encoding='utf-8'))
        if (original.get('source_sha256') != SOURCE_SHA or original.get('spec') != SPEC
                or not isinstance(original.get('threshold'), (int, float))
                or not 0 < original['threshold'] < 1):
            raise ValueError('Invalid legacy original AVA strike calibration')
        self.metadata['thresholds']['strike'] = original['threshold']
        self.metadata['strike_policy'] = {'source': 'legacy_original_public_calibration',
                                          'calibration': original, 'revalidated_on_current_detector': False}
        self.metadata['grapple_policy'] = {'alarm_enabled': False, 'reason': 'unvalidated_pilot_diagnostic_only'}
        self.metadata['threshold'] = min(self.metadata['thresholds'].values())
        self.metadata['fall_policy'] = {'version': 3, 'definition': 'lying_or_transition_to_lying',
                                       'source': 'AVA 8 lie/sleep', 'threshold': LYING_THRESHOLD,
                                       'threshold_calibrated': False, 'accident_inference': False,
                                       'confirmation': 'same_horizontal_person_above_release',
                                       'continuation_min_iou': .35, 'continuation_min_aspect': 1.5}
        self.metadata['archived_calibration_test'] = calibration.get('test', {})
        self.metadata['test'] = {}

    @torch.inference_mode()
    def infer(self, frames):
        bag, auxiliary, output = self.extract_features(frames)
        scores = {'strike': max((person['fight_score'] for person in output['actions']), default=0.),
                  'fall': lying_evidence(output['actions'])[0],
                  'grapple': 0., 'fast_running': 0.}
        regions = {label: [] for label in LABELS}
        for label, field in [('strike', 'fight_score'), ('fall', 'ava_lying_score')]:
            regions[label] = [person['box'] for person in output['actions']
                              if person[field] >= self.metadata['thresholds'][label]]
        if len(bag):
            features = torch.from_numpy(bag).to(self.device)
            values = self.grapple_head(features).sigmoid().cpu().numpy()
            selected = int(values.argmax())
            scores['grapple'] = float(values[selected])
            regions['grapple'] = [output['candidate_regions'][selected]]
            running = self.running_head(features[:output['people']],
                                        torch.from_numpy(auxiliary[:output['people']]).to(self.device)).sigmoid().cpu().numpy()
            for person, value in zip(output['actions'], running):
                person['running_score'] = float(value)
        return {**output, 'score': max(scores.values()), 'regions': [],
                'action_scores': scores, 'action_regions': regions}

    @torch.inference_mode()
    def current_view(self, frames, output):
        from running_evidence import verified_running_evidence
        display = ActionEngine.current_view(self, frames, output)
        display['action_scores'] = dict(output['action_scores'])
        original, current = output['person_boxes'], display['person_boxes']
        assignments = match_display_people(frames[4], frames[-1], original, current)
        display['action_regions'] = {label: remap_display_regions(original, current, assignments, regions)
                                     for label, regions in output['action_regions'].items()}
        display['people'] = len(current)
        display['action_evidence'] = {}
        for label in ('strike', 'fall', 'grapple'):
            display['action_evidence'][label] = 'model_only' if display['action_regions'][label] else 'no_localized_support'
        display['action_scores']['fall'], display['action_regions']['fall'] = lying_evidence(display['actions'])
        display['action_evidence']['fall'] = ('ava_lie_sleep' if display['action_regions']['fall'] else 'no_lying_support')
        display['fall_policy'] = self.metadata['fall_policy']
        supported = []
        for person in display['actions']:
            valid, reason = verified_running_evidence(person, display['actions'], frames[-1].shape)
            value = person.get('running_score', 0.) if valid else 0.
            person['running_evidence'] = reason
            supported.append(value)
            if value >= self.metadata['thresholds']['fast_running']:
                display['action_regions']['fast_running'].append(person['box'])
        display['action_scores']['fast_running'] = max(supported, default=0.)
        display['score'] = max(display['action_scores'].values())
        return display
