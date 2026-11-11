"""Independent strike, grapple and fall pilot heads; original AVA labels stay intact."""

import json
import math
from pathlib import Path

import torch

from action_engine import ActionEngine, InteractionHead, SOURCE_SHA, SPEC, digest, match_display_people, remap_display_regions


LABELS = ('strike', 'grapple', 'fall')
PARENT_SHA = 'ddb09180284c4093189c53b7decbfcc5b51fd2d9b7700536d104524bceb5c344'


def load_three_actions(weights, device='cpu'):
    weights = Path(weights)
    metadata = json.loads((weights / 'three_actions.json').read_text(encoding='utf-8'))
    if (metadata.get('labels') != list(LABELS) or metadata.get('output') != 'independent_sigmoid'
            or metadata.get('source_sha256') != SOURCE_SHA or metadata.get('spec') != SPEC
            or metadata.get('parent_checkpoint_sha256') != PARENT_SHA
            or metadata.get('checkpoint_sha256') != digest(weights / 'three_actions.pt')):
        raise ValueError('Incompatible three-action checkpoint')
    thresholds = metadata.get('thresholds', {})
    if set(thresholds) != set(LABELS) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value < 1 for value in thresholds.values()):
        raise ValueError('Invalid per-action thresholds')
    state = torch.load(weights / 'three_actions.pt', map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor)
                                        or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Expected finite tensor-only three-action state')
    heads = torch.nn.ModuleList([InteractionHead() for _ in LABELS])
    heads.load_state_dict(state, strict=True)
    return metadata, heads.to(device).eval()


class ThreeActionsEngine(ActionEngine):
    def __init__(self, weights=None, device='cuda:0', person_recovery=True):
        super().__init__(weights=weights, device=device, person_recovery=person_recovery)
        metadata, self.action_heads = load_three_actions(self.weights, device)
        self.metadata = {**self.metadata, **metadata, 'threshold': min(metadata['thresholds'].values()),
                         'confirm_seconds': .4, 'scope': 'Independent strike / grapple / fall pilot'}

    @torch.inference_mode()
    def infer(self, frames):
        bag, _, output = self.extract_features(frames)
        action_scores = dict.fromkeys(LABELS, 0.)
        action_regions = {label: [] for label in LABELS}
        if len(bag):
            features = torch.from_numpy(bag).to(self.device)
            for label, head in zip(LABELS, self.action_heads):
                scores = head(features).sigmoid().cpu().numpy()
                selected = int(scores.argmax())
                action_scores[label] = float(scores[selected])
                action_regions[label] = [output['candidate_regions'][selected]]
        return {**output, 'score': max(action_scores.values()), 'regions': [],
                'action_scores': action_scores, 'action_regions': action_regions}

    @torch.inference_mode()
    def current_view(self, frames, output):
        display = super().current_view(frames, output)
        original, current = output['person_boxes'], display['person_boxes']
        assignments = match_display_people(frames[4], frames[-1], original, current)
        display['action_regions'] = {
            label: remap_display_regions(original, current, assignments, regions)
            for label, regions in output['action_regions'].items()
        }
        display['people'] = len(current)
        return display
