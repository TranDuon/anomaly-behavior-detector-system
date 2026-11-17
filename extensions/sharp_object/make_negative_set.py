"""H4: turn videos of people holding NON-weapons (sticks, brooms, pipes, umbrellas...) into a negative set.

    python -m extensions.sharp_object.make_negative_set VIDEO_OR_FOLDER [...] --out vn_negatives_001.zip

Every sampled frame gets an empty YOLO label ("no sharp object"). build_dataset.py keeps empty
labels from inputs that carry the NEGATIVE_SET marker, so every hand crop of these frames becomes
a negative. Upload the zip to MyDrive/SENTINEL/datasets/sharp_object_vn/ (the hand_03 notebook
reads every vn_negatives_*.zip there).

Groups: one per --segment-seconds of each video, so build_dataset's group split sends some
segments to val/test (false alarms on these objects are then measured) without one long video
landing entirely in one split. The source videos are copied into source_videos/ for the clip check.

Only use videos in which nobody holds a real blade at any time.
"""

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def videos_from(paths):
    found = []
    for path in paths:
        if path.is_dir():
            found += sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            found.append(path)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+", help="video files or folders")
    parser.add_argument("--out", type=Path, required=True, help="zip to write (vn_negatives_NNN.zip)")
    parser.add_argument("--fps", type=float, default=2.0, help="frames sampled per second")
    parser.add_argument("--segment-seconds", type=float, default=10.0, help="one split group per segment")
    parser.add_argument("--max-side", type=int, default=1280, help="frames larger than this are downscaled")
    parser.add_argument("--note", default="user-filmed, person holding non-weapon objects")
    parser.add_argument("--no-videos", action="store_true", help="do not copy the source videos into the zip")
    args = parser.parse_args()

    import cv2

    videos = videos_from(args.inputs)
    if not videos:
        raise SystemExit("no videos found")
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("NEGATIVE_SET", "every frame here is a negative: no sharp object is held\n")
        bundle.writestr("data.yaml", "train: images/train\nval: images/train\nnames:\n  0: dao_phong_lon\n  1: ma_tau\n"
                                     "  2: dao_ngan\n  3: kiem\n  4: liem\n  5: riu\n  6: vat_sac_khac\n")
        for index, video_path in enumerate(videos):
            tag = f"neg{index:02d}_" + re.sub(r"[^A-Za-z0-9]", "", video_path.stem)[:24]
            video = cv2.VideoCapture(str(video_path))
            native = video.get(cv2.CAP_PROP_FPS) or 30.0
            stride, frame_index, kept = max(1, round(native / args.fps)), 0, 0
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    seconds = frame_index / native
                    scale = args.max_side / max(frame.shape[:2])
                    if scale < 1:
                        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    stem = f"{tag}_t{seconds:07.2f}".replace(".", "p")
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    bundle.writestr(f"images/train/{stem}.jpg", encoded.tobytes())
                    bundle.writestr(f"labels/train/{stem}.txt", "")
                    rows.append({"image": f"images/train/{stem}.jpg", "source_group":
                                 f"user_negative_{tag}_seg{int(seconds // args.segment_seconds):02d}",
                                 "video": video_path.name, "seconds": round(seconds, 2), "note": args.note})
                    kept += 1
                frame_index += 1
            video.release()
            if not args.no_videos:
                bundle.write(video_path, f"source_videos/{tag}{video_path.suffix.lower()}")
            print(f"{video_path.name}: {kept} frames", flush=True)
        text = io.StringIO()
        writer = csv.DictWriter(text, ["image", "source_group", "video", "seconds", "note"])
        writer.writeheader()
        writer.writerows(rows)
        bundle.writestr("manifest.csv", text.getvalue())
    groups = len({r["source_group"] for r in rows})
    print(f"wrote {args.out}: {len(rows)} negative frames, {len(videos)} videos, {groups} groups")


if __name__ == "__main__":
    main()
