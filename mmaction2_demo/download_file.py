"""Resume official HTTPS downloads with the verified portable aria2 utility."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
ARIA_URL = "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
ARIA_SHA256 = "67d015301eef0b612191212d564c5bb0a14b5b9c4796b76454276a4d28d9b288"


def download(url, target):
    if not url.startswith(
        ("https://download.pytorch.org/", "https://download.openmmlab.com/")
    ):
        raise ValueError("Only the official model/library download hosts are allowed")
    target = Path(target).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Download must stay inside the isolated demo folder")
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "downloads" / "aria2.zip"
    if not archive.exists():
        subprocess.run(
            [
                "curl.exe",
                "-fL",
                "--retry",
                "2",
                "--max-time",
                "180",
                ARIA_URL,
                "-o",
                str(archive),
            ],
            check=True,
        )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARIA_SHA256:
        raise ValueError("Portable downloader archive checksum mismatch")
    executable = ROOT / "downloads/aria2/aria2-1.37.0-win-64bit-build1/aria2c.exe"
    if not executable.exists():
        with zipfile.ZipFile(archive) as package:
            folder = (ROOT / "downloads/aria2").resolve()
            if not all(
                (folder / item.filename).resolve().is_relative_to(folder)
                for item in package.infolist()
            ):
                raise ValueError("Unsafe downloader archive")
            package.extractall(folder)
    subprocess.run(
        [
            str(executable),
            "--continue=true",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=8M",
            "--file-allocation=none",
            "--max-tries=8",
            "--retry-wait=3",
            "--connect-timeout=15",
            "--timeout=30",
            "--summary-interval=30",
            "--console-log-level=warn",
            "--download-result=full",
            "--auto-file-renaming=false",
            "--allow-overwrite=false",
            "--dir=" + str(target.parent),
            "--out=" + target.name,
            url,
        ],
        check=True,
        timeout=7200,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    download(args.url, args.target)
