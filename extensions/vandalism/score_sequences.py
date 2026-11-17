"""Score every runtime window (stride 1 sample = 0.27 s) of the local test sample.

Produces the per-clip vandalism score sequence the runtime would see with
cadence 1, so confirmation rules can be compared offline (see gate_benchmark.py)
without re-running the whole core pipeline. Runs in extensions/env (GPU).

    extensions\\env\\Scripts\\python.exe -m extensions.vandalism.score_sequences
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from extensions.vandalism.model import FRAMES, AttentiveProbe, Backbone, preprocess
from extensions.vandalism.train_vandalism import sample_frames

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "datasets" / "vandalism" / "raw" / "eval_v1"
WEIGHTS = ROOT / "weights" / "extensions" / "vandalism"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=EVAL / "manifest.jsonl")
    parser.add_argument("--output", type=Path, default=EVAL / "score_sequences.jsonl")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    spec = json.loads((WEIGHTS / "vandalism.json").read_text(encoding="utf-8"))
    tail = spec.get("backbone_tail_file")  # fine-tuned encoder tail (probe_02_ft), same as the runtime
    tail_state = torch.load(WEIGHTS / tail, map_location="cpu", weights_only=True) if tail else None
    backbone = Backbone(device="cuda", model_id=spec["backbone"], revision=spec["backbone_revision"],
                        tail_state=tail_state)
    probe = AttentiveProbe()
    probe.load_state_dict(torch.load(WEIGHTS / "vandalism.pt", map_location="cpu", weights_only=True))
    probe.cuda().eval()
    items = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l]
    done = set()
    if args.output.is_file():
        done = {json.loads(l)["path"] + str(json.loads(l).get("segment")) for l in args.output.read_text(encoding="utf-8").splitlines() if l}
    started = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as stream, torch.inference_mode():
        for number, item in enumerate(items, 1):
            key = item["path"] + str(item.get("segment"))
            if key in done:
                continue
            frames, times, fps = sample_frames(item["path"], *(item.get("segment") or (None, None)))
            if not frames:
                continue
            while len(frames) < FRAMES:
                frames.append(frames[-1])
                times.append(times[-1])
            starts = list(range(0, len(frames) - FRAMES + 1))
            scores = []
            for begin in range(0, len(starts), args.batch):
                batch = torch.stack([preprocess(frames[s:s + FRAMES]) for s in starts[begin:begin + args.batch]])
                logits = probe(backbone.encode(batch).cuda())
                scores += torch.sigmoid(logits).float().cpu().tolist()
            ends = [times[s + FRAMES - 1] for s in starts]  # runtime decides at the newest frame
            labels = [item["binary"]] * len(ends)
            if "intervals_frames" in item:
                labels = [int(any(a <= round(t * fps) <= b for a, b in item["intervals_frames"])) for t in ends]
            stream.write(json.dumps({"path": item["path"], "segment": item.get("segment"), "source": item["source"],
                                     "label": item["label"], "binary": item["binary"], "t": [round(t, 3) for t in ends],
                                     "score": [round(s, 4) for s in scores], "window_label": labels}) + "\n")
            stream.flush()
            if number % 50 == 0:
                print(f"{number}/{len(items)}  {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    print("done", args.output)


if __name__ == "__main__":
    main()
