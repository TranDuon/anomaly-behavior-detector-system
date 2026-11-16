"""Fetch vandalism data from sources usable without registration.

Shared by the local V1 baseline and the Colab V2/V3 notebooks. Standard
library + OpenCV only, so it runs in any environment (no extra installs):

* Kinetics-700-2020: one tar per class on the CVDF S3 bucket.
* ASHEN: individual mp4 files on Hugging Face (no login).
* Charades: members of a 16 GB zip fetched with HTTP range requests, so only
  the needed videos are downloaded.

Sources and class choices live in ``datasets/vandalism/sources.json``.
"""

import csv
import hashlib
import io
import json
import random
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "datasets" / "vandalism" / "sources.json"
HF_RAW = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
USER_AGENT = "sentinel-dataset-fetch/1.0"


def sources():
    return json.loads(SOURCES.read_text(encoding="utf-8"))


def _open(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=120)


def download(url, target, chunk=1 << 20, attempts=6):
    """Download to ``target`` with retries and resume (atomic rename). Returns SHA-256."""
    target = Path(target)
    if target.is_file():
        return sha256(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with _open(url, headers) as response:
                if offset and response.status != 206:  # server ignored the range: restart
                    offset = 0
                with partial.open("ab" if offset else "wb") as stream:
                    for block in iter(lambda: response.read(chunk), b""):
                        stream.write(block)
            break
        except (OSError, TimeoutError) as error:
            if attempt == attempts:
                raise
            print(f"  retry {attempt}/{attempts - 1} after {type(error).__name__}: {url}", flush=True)
            time.sleep(min(30, 2 ** attempt))
    partial.replace(target)
    return sha256(target)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ Kinetics
def kinetics_annotations(split, cache_dir):
    meta = sources()["kinetics700_2020"]
    path = Path(cache_dir) / f"k700_2020_{split}.csv"
    download(meta["annotations"].format(split=split), path)
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fetch_kinetics_class(split, label, index, dest, limit=None, seed=0):
    """Stream one class tar and keep at most ``limit`` clips (random, seeded).

    Returns a list of records {path, label, youtube_id, source}.
    """
    meta = sources()["kinetics700_2020"]
    url = meta["url"].format(split=split, index=index)
    dest = Path(dest) / "kinetics" / split / label.replace(" ", "_").replace("(", "").replace(")", "")
    dest.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest.glob("*.mp4"))
    archive_sha = None
    if existing and (limit is None or len(existing) >= limit):
        chosen = existing[:limit] if limit else existing
    else:
        tar_path = Path(dest).parents[2] / "_tars" / Path(url).name
        archive_sha = download(url, tar_path)
        with tarfile.open(tar_path, mode="r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".mp4")]
            wrong = [m.name for m in members if not m.name.startswith(label + "/")]
            if wrong:
                raise ValueError(f"{url} holds {wrong[0]}, expected class '{label}'")
            members.sort(key=lambda m: m.name)
            random.Random(seed).shuffle(members)
            if limit:
                members = members[:limit]
            for member in members:
                (dest / Path(member.name).name).write_bytes(archive.extractfile(member).read())
        tar_path.unlink()
        chosen = sorted(dest / Path(m.name).name for m in members)
    return [{"path": str(path), "label": label, "source": "kinetics700_2020", "kinetics_split": split,
             "youtube_id": path.name[:11], "group": "yt:" + path.name[:11], "archive": Path(url).name,
             "archive_sha256": archive_sha} for path in chosen]


# ------------------------------------------------------------------ ASHEN
def ashen_split(name):
    meta = sources()["ashen_vandalism"]
    url = HF_RAW.format(repo=meta["repo"], path=f"splits_v2_native/{name}")
    with _open(url) as response:
        lines = response.read().decode("utf-8").split()
    return [line.replace(".npy", ".mp4") for line in lines if line.strip()]


def ashen_temporal_annotations():
    meta = sources()["ashen_vandalism"]
    url = HF_RAW.format(repo=meta["repo"], path="splits_v2_native/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    with _open(url) as response:
        rows = response.read().decode("utf-8").splitlines()
    result = {}
    for row in rows:
        parts = row.split()
        if len(parts) < 6:
            continue
        name, label, values = parts[0], parts[1], list(map(int, parts[2:6]))
        result[name] = {"label": label, "intervals": [(values[i], values[i + 1]) for i in (0, 2) if values[i] >= 0]}
    return result


def fetch_ashen(paths, dest):
    meta = sources()["ashen_vandalism"]
    records = []
    for path in paths:
        target = Path(dest) / "ashen" / path
        download(HF_RAW.format(repo=meta["repo"], path=path), target)
        label = "vandalism" if "/Vandalism/" in path else "normal" if path.startswith("normal/") else Path(path).parent.name.lower()
        records.append({"path": str(target), "label": label, "source": "ashen", "group": "ashen:" + Path(path).stem,
                        "ashen_path": path})
    return records


# ------------------------------------------------------------------ Charades
class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url = url
        with _open(url, {"Range": "bytes=0-0"}) as response:
            self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def readinto(self, buffer):
        if self.position >= self.size:
            return 0
        end = min(self.size, self.position + len(buffer)) - 1
        for attempt in range(1, 7):
            try:
                with _open(self.url, {"Range": f"bytes={self.position}-{end}"}) as response:
                    data = response.read()
                break
            except (OSError, TimeoutError):
                if attempt == 6:
                    raise
                time.sleep(min(30, 2 ** attempt))
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def charades_annotations(cache_dir):
    meta = sources()["charades"]
    archive = Path(cache_dir) / "Charades.zip"
    download(meta["annotations"], archive)
    rows = []
    with zipfile.ZipFile(archive) as bundle:
        for split in ("train", "test"):
            name = next(n for n in bundle.namelist() if n.endswith(f"Charades_v1_{split}.csv"))
            for row in csv.DictReader(io.TextIOWrapper(bundle.open(name), encoding="utf-8")):
                row["split"] = split
                row["segments"] = [
                    {"class": part.split()[0], "start": float(part.split()[1]), "end": float(part.split()[2])}
                    for part in (row.get("actions") or "").split(";") if part
                ]
                rows.append(row)
    return rows


def _fetch_member(url, info, target):
    """One ranged GET per zip member: local header + data, then decompress if needed."""
    import struct
    import zlib

    span_end = info.header_offset + 30 + len(info.filename.encode()) + 1024 + info.compress_size
    for attempt in range(1, 7):
        try:
            with _open(url, {"Range": f"bytes={info.header_offset}-{span_end}"}) as response:
                blob = response.read()
            break
        except (OSError, TimeoutError):
            if attempt == 6:
                raise
            time.sleep(min(30, 2 ** attempt))
    signature, *_, name_length, extra_length = struct.unpack("<IHHHHHIIIHH", blob[:30])
    if signature != 0x04034B50:
        raise ValueError(f"bad local header for {info.filename}")
    start = 30 + name_length + extra_length
    payload = blob[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(payload, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression {info.compress_type}")
    partial = target.with_suffix(".partial")
    partial.write_bytes(payload)
    partial.replace(target)


def fetch_charades_videos(video_ids, dest, workers=16):
    """Extract only the given videos from the remote 480p zip (parallel ranged GETs)."""
    from concurrent.futures import ThreadPoolExecutor

    meta = sources()["charades"]
    dest = Path(dest) / "charades"
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {vid for vid in video_ids if not (dest / f"{vid}.mp4").is_file()}
    if wanted:
        remote = io.BufferedReader(HttpRangeFile(meta["videos"]), buffer_size=8 << 20)
        with zipfile.ZipFile(remote) as bundle:  # reads only the central directory
            infos = [i for i in bundle.infolist() if i.filename.endswith(".mp4") and Path(i.filename).stem in wanted]
        with ThreadPoolExecutor(workers) as pool:
            for future in [pool.submit(_fetch_member, meta["videos"], info, dest / f"{Path(info.filename).stem}.mp4")
                           for info in infos]:
                future.result()
    return {vid: str(dest / f"{vid}.mp4") for vid in video_ids if (dest / f"{vid}.mp4").is_file()}
