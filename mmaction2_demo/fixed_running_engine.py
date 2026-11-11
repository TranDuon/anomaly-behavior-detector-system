"""Load the separate running pilot without replacing the fall/wrestle head."""

import json
import math
from pathlib import Path

import torch

from action_engine import AUX_NAMES, ActionEngine, InteractionDistanceHead, InteractionHead, SOURCE_SHA, SPEC, digest
from running_evidence import verified_running_evidence


PARENT_SHA = "ddb09180284c4093189c53b7decbfcc5b51fd2d9b7700536d104524bceb5c344"


def load_running_checkpoint(weights, device="cpu"):
    weights = Path(weights)
    metadata = json.loads((weights / "interaction_fixed_running.json").read_text(encoding="utf-8"))
    calibration = json.loads((Path(__file__).parent / "config/fall_wrestle_calibration.json").read_text())
    if (metadata.get("source_sha256") != SOURCE_SHA or metadata.get("spec") != SPEC
            or metadata.get("parent_checkpoint_sha256") != PARENT_SHA
            or metadata.get("checkpoint_sha256") != digest(weights / "interaction_fixed_running.pt")
            or digest(weights / "interaction_fall_wrestle.pt") != PARENT_SHA
            or metadata.get("motion_features") != list(AUX_NAMES)
            or metadata.get("running_head_version") != 2
            or metadata.get("person_threshold") != 0.8
            or metadata.get("labels") != ["fight_wrestle_fall", "fast_running"]
            or metadata.get("parent_threshold") != calibration["threshold"]
            or calibration["checkpoint_sha256"] != PARENT_SHA):
        raise ValueError("Incompatible running checkpoint or parent")
    threshold = metadata.get("running_threshold", float("nan"))
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid running threshold")
    state = torch.load(weights / "interaction_fixed_running.pt", map_location="cpu", weights_only=True)
    parent_state = torch.load(weights / "interaction_fall_wrestle.pt", map_location="cpu", weights_only=True)
    if set(state) != {"parent", "running"} or set(state["parent"]) != set(parent_state):
        raise ValueError("Invalid running package state")
    if any(not torch.equal(state["parent"][key], value) for key, value in parent_state.items()):
        raise ValueError("Running package changed the parent weights")
    parent = InteractionHead()
    parent.load_state_dict(state["parent"], strict=True)
    running = InteractionDistanceHead()
    running.load_state_dict(state["running"], strict=True)
    if any(not torch.isfinite(value).all() for value in running.state_dict().values()):
        raise ValueError("Non-finite running weights")
    return metadata, running.to(device).eval()


class FixedRunningEngine(ActionEngine):
    def __init__(self, weights=None, device="cuda:0", person_recovery=False):
        super().__init__(weights=weights, device=device, tuned=True, fall_wrestle=True,
                         person_threshold=0.8, person_recovery=person_recovery)
        metadata, self.running_head = load_running_checkpoint(self.weights, device)
        self.metadata = {**self.metadata, **metadata, "threshold": metadata["parent_threshold"],
                         "confirm_seconds": 0.4, "scope": "Separate fight/wrestle/fall and running pilots"}

    @torch.inference_mode()
    def current_view(self, frames, output):
        display = super().current_view(frames, output)
        evidence = []
        supported = []
        regions = []
        for person in display["actions"]:
            raw = person.get("running_score", 0.0)
            valid, reason = verified_running_evidence(person, display['actions'], frames[-1].shape)
            person["running_raw_score"] = raw
            person["running_score"] = raw if valid else 0.0
            person["running_evidence"] = reason
            supported.append(person["running_score"])
            evidence.append(reason)
            if valid and raw >= self.metadata["running_threshold"]:
                regions.append(person["box"])
        return {**display, "running_raw_score": output["running_score"],
                "running_score": max(supported, default=0.0), "running_regions": regions,
                "running_evidence": evidence, "running_guard_version": 2}

    @torch.inference_mode()
    def infer(self, frames):
        bag, auxiliary, output = self.extract_features(frames)
        if not len(bag):
            return {**output, "score": 0.0, "regions": [], "running_score": 0.0,
                    "running_regions": []}
        features = torch.from_numpy(bag).to(self.device)
        parent_scores = self.head(features).sigmoid().cpu().numpy()
        selected = int(parent_scores.argmax())
        people = output["people"]
        running_scores = self.running_head(
            features[:people], torch.from_numpy(auxiliary[:people]).to(self.device)
        ).sigmoid().cpu().numpy()
        running_regions = [box for box, score in zip(output["person_boxes"], running_scores)
                           if float(score) >= self.metadata["running_threshold"]]
        for person, score in zip(output["actions"], running_scores):
            person["running_score"] = float(score)
        return {**output, "score": float(parent_scores[selected]),
                "regions": [output["candidate_regions"][selected]],
                "running_score": float(running_scores.max()), "running_regions": running_regions}
