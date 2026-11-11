"""Temporal pose specialists; independent from the installed RGB action branches."""

import itertools

import numpy as np
import torch
from torch import nn


LABELS = ('grapple', 'fall')
SPEC = {'version': 1, 'frames': 32, 'fps': 15., 'joints': 17, 'channels': 3,
        'normalization': 'full_image_minus_one_one', 'minimum_track_coverage': .8,
        'maximum_people': 6, 'prediction_offset_seconds': 16 / 15,
        'pose_estimator': 'yolov8n-pose', 'pose_image_size': 640}
SOURCE_URL = ('https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/'
              'stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/'
              'stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth')
SOURCE_SHA = '484a394a50246032fef9694396096902909784e8618ccc845d2859dc2b952fea'


def overlap(first, second):
    intersection = np.maximum(np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]), 0).prod()
    areas = np.maximum(first[2:] - first[:2], 0).prod() + np.maximum(second[2:] - second[:2], 0).prod()
    return float(intersection / max(areas - intersection, 1.))


def track_window(detections, image_shape):
    if len(detections) != SPEC['frames']:
        raise ValueError('Dense real-frame pose window required')
    height, width = image_shape[:2]
    center = len(detections) // 2
    seeds = detections[center][:SPEC['maximum_people']]
    tracks = np.zeros((len(seeds), len(detections), 17, 3), np.float32)
    masks = np.zeros((len(seeds), len(detections)), bool)
    for person, seed in enumerate(seeds):
        tracks[person, center] = seed['points']
        masks[person, center] = True
    for direction in (-1, 1):
        boxes = [np.asarray(seed['box']).copy() for seed in seeds]
        gaps = [0] * len(seeds)
        for index in range(center + direction, len(detections) if direction == 1 else -1, direction):
            proposals = []
            for person, previous in enumerate(boxes):
                if gaps[person] > 3:
                    continue
                for target, candidate in enumerate(detections[index]):
                    box = np.asarray(candidate['box'])
                    distance = np.linalg.norm((previous[:2] + previous[2:] - box[:2] - box[2:]) / 2)
                    scale = max(np.linalg.norm(previous[2:] - previous[:2]), 1.)
                    similarity = overlap(previous, box)
                    if similarity >= .15 or distance / scale <= .18:
                        proposals.append((similarity - distance / scale, person, target))
            assigned, used = set(), set()
            for _, person, target in sorted(proposals, reverse=True):
                if person in assigned or target in used:
                    continue
                candidate = detections[index][target]
                tracks[person, index] = candidate['points']
                masks[person, index] = True
                boxes[person] = np.asarray(candidate['box'])
                gaps[person] = 0
                assigned.add(person)
                used.add(target)
            for person in set(range(len(seeds))) - assigned:
                gaps[person] += 1
    coverage = masks.mean(axis=1)
    keep = coverage >= SPEC['minimum_track_coverage']
    tracks, masks = tracks[keep], masks[keep]
    boxes = [seed['box'] for seed, valid in zip(seeds, keep) if valid]
    tracks[..., 0] = (tracks[..., 0] - width / 2) / (width / 2)
    tracks[..., 1] = (tracks[..., 1] - height / 2) / (height / 2)
    tracks[~masks] = 0
    return tracks, boxes, coverage.tolist()


def candidate_bags(tracks):
    candidates = [(index,) for index in range(len(tracks))]
    candidates += list(itertools.combinations(range(len(tracks)), 2))
    bags = np.zeros((len(candidates), 2, SPEC['frames'], 17, 3), np.float32)
    for number, indexes in enumerate(candidates):
        for slot, person in enumerate(indexes):
            bags[number, slot] = tracks[person]
    return bags, np.asarray([len(indexes) == 2 for indexes in candidates]), candidates


class PoseSpecialists(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        from mmaction.models import STGCN
        self.backbone = STGCN(graph_cfg={'layout': 'coco', 'mode': 'stgcn_spatial'})
        self.relation = nn.GRU(51, 32, batch_first=True)
        self.grapple = nn.Sequential(nn.Linear(544, 64), nn.ReLU(), nn.Dropout(.4), nn.Linear(64, 1))
        self.fall = nn.Linear(256, 1)
        if pretrained is not None:
            state = torch.load(pretrained, map_location='cpu', weights_only=True)['state_dict']
            backbone = {key.removeprefix('backbone.'): value for key, value in state.items() if key.startswith('backbone.')}
            self.backbone.load_state_dict(backbone, strict=True)
            weight, bias = state['cls_head.fc.weight'], state['cls_head.fc.bias']
            self.fall.weight.data.copy_(weight[42:43] - weight.mean(dim=0, keepdim=True))
            self.fall.bias.data.copy_(bias[42:43] - bias.mean())

    def configure_training(self, unfreeze=False):
        self.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if unfreeze:
            for layer in self.backbone.gcn[-2:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        self.relation.train()
        self.grapple.train()
        self.fall.train()

    def forward(self, bags, pairs):
        features = self.backbone(bags).mean(dim=(-1, -2))
        pooled = torch.cat((features.mean(dim=1), features.max(dim=1).values), dim=1)
        relative = (bags[:, 0, ..., :2] - bags[:, 1, ..., :2]).abs()
        confidence = bags[:, 0, ..., 2:] * bags[:, 1, ..., 2:]
        relation = torch.cat((relative, confidence), dim=-1).flatten(2)
        _, hidden = self.relation(relation)
        grapple = self.grapple(torch.cat((pooled, hidden[-1]), dim=1)).flatten()
        grapple = grapple.masked_fill(~pairs, -20.)
        present = bags[..., 2].sum(dim=(-1, -2)) > 0
        fall = self.fall(features).squeeze(-1).masked_fill(~present, -20.).max(dim=1).values
        return torch.stack((grapple, fall), dim=1)
