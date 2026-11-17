"""V5: runtime vandalism extension (runs inside extensions/worker.py, extensions/env).

Per step: last 8 of the 9 core samples -> frozen V-JEPA 2 (fp16 on GPU) ->
attentive probe -> probability. Scene-level: the overlay highlights the people
the core already detected. Threshold / gate timings come from vandalism.json.
"""

import json

import torch

from extensions.base import WEIGHTS, Extension, register
from extensions.vandalism.model import FRAMES, AttentiveProbe, Backbone, preprocess


@register
class VandalismExtension(Extension):
    name = "vandalism"
    display_label = "VANDALISM"
    confirm_seconds = 0.8
    release_seconds = 1.0
    threshold = 0.65

    def load(self, weights_dir=WEIGHTS):
        folder = weights_dir / "vandalism"
        spec = json.loads((folder / "vandalism.json").read_text(encoding="utf-8"))
        self.threshold = float(spec["threshold"])
        self.confirm_seconds = float(spec.get("confirm_seconds", self.confirm_seconds))
        self.release_seconds = float(spec.get("release_seconds", self.release_seconds))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tail = spec.get("backbone_tail_file")  # fine-tuned last encoder blocks (probe_02 finetune)
        tail_state = torch.load(folder / tail, map_location="cpu", weights_only=True) if tail else None
        self.backbone = Backbone(device=self.device, model_id=spec["backbone"], revision=spec["backbone_revision"],
                                 tail_state=tail_state)
        self.probe = AttentiveProbe()
        self.probe.load_state_dict(torch.load(folder / "vandalism.pt", map_location="cpu", weights_only=True), strict=True)
        self.probe.to(self.device).eval()
        # Warm up so the first real step is not slowed by CUDA kernel setup.
        dummy = torch.zeros(1, FRAMES, 3, 256, 256)
        self.probe(self.backbone.encode(dummy).to(self.device))

    @torch.inference_mode()
    def observe(self, frames, timestamp, people=None):
        clip = preprocess(list(frames[-FRAMES:])).unsqueeze(0)
        tokens = self.backbone.encode(clip).to(self.device)
        score = float(torch.sigmoid(self.probe(tokens)).item())
        boxes = list((people or {}).get("person_boxes", []))
        return {"score": score, "boxes": boxes, "details": {"threshold": self.threshold}}
