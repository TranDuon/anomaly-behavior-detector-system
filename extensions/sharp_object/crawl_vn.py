"""Collect Vietnamese news images/videos of typical weapons (dao phóng lợn, mã tấu, dao/kiếm tự chế).

Starts from datasets/sharp_object/vn_seeds.txt, follows same-site article links whose
URL slug names a weapon or knife violence, and records every article image (with
caption) and video stream. Media are then downloaded with their article URL kept as
provenance. Polite: one request per site at a time with a delay.

    python -m extensions.sharp_object.crawl_vn crawl    --out datasets/sharp_object/vn/raw --max-articles 600
    python -m extensions.sharp_object.crawl_vn download --out datasets/sharp_object/vn/raw

News photos/footage stay copyrighted by the publishers: research use only, keep the
pool private, never redistribute.
"""

import argparse
import hashlib
import html
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "datasets" / "sharp_object" / "vn_seeds.txt"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
SITES = ("vnexpress.net", "tuoitre.vn", "dantri.com.vn", "thanhnien.vn", "vietnamnet.vn", "znews.vn",
         "cand.com.vn", "nld.com.vn", "plo.vn", "vtv.vn", "antv.gov.vn", "vtc.vn")
STRONG = ("dao-phong-lon", "phong-lon", "ma-tau", "dao-tu-che", "kiem-tu-che", "hung-khi", "vac-dao", "cam-dao",
          "mang-dao", "rut-dao", "dao-kiem", "dao-rua", "-rua-", "chem-nguoi", "chem-nhau", "hon-chien", "truy-sat",
          "vu-khi-tho-so", "dao-nhon", "kiem-", "-dao-")
WEAPON_WORDS = ("phóng lợn", "mã tấu", "tự chế", "hung khí", "dao", "kiếm", "rựa", "vũ khí", "tang vật", "chém", "cầm")
SKIP_IMAGE = re.compile(r"logo|icon|avatar|banner|sprite|placeholder|blank|default|\.gif|\.svg|/ads?/|qr", re.I)


def get(url, binary=False, timeout=40):
    request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "vi,en;q=0.8"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data if binary else data.decode("utf-8", "replace")


def site_of(url):
    host = urllib.parse.urlparse(url).netloc.lower()
    return next((s for s in SITES if host == s or host.endswith("." + s)), None)


def is_article(url):
    path = urllib.parse.urlparse(url).path.lower()
    return bool(re.search(r"(\d{5,}|post\d+|-i\d{5,}/?)(\.html?|/)?$", path))


def relevant(url):
    slug = urllib.parse.unquote(urllib.parse.urlparse(url).path.lower())
    return any(word in slug for word in STRONG)


def extract(page, url):
    title = html.unescape(re.search(r"<title>(.*?)</title>", page, re.S | re.I).group(1)).strip() if "<title>" in page.lower() else ""
    links = set()
    for href in re.findall(r'href="([^"#]+)"', page):
        absolute = urllib.parse.urljoin(url, html.unescape(href))
        if site_of(absolute) and is_article(absolute) and relevant(absolute):
            links.add(absolute.split("?")[0])
    images = []
    for match in re.finditer(r'<(?:img|source)[^>]+>', page, re.I):
        tag = match.group(0)
        src = (re.search(r'(?:data-src|data-original|src|data-srcset|srcset)="([^"\s]+)', tag) or [None, None])[1]
        if not src:
            continue
        src = urllib.parse.urljoin(url, html.unescape(src))
        if not re.search(r"\.(jpe?g|png|webp)", src, re.I) or SKIP_IMAGE.search(src):
            continue
        alt = html.unescape((re.search(r'alt="([^"]*)"', tag) or [None, ""])[1])
        window = page[match.end(): match.end() + 1500]
        caption = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", window, re.S | re.I)
        caption = re.sub(r"<[^>]+>", " ", html.unescape(caption.group(1))).strip() if caption else ""
        images.append({"url": src, "alt": alt, "caption": re.sub(r"\s+", " ", caption)[:400]})
    og = re.search(r'<meta[^>]+property="og:image"[^>]+content="([^"]+)"', page) or \
        re.search(r'<meta[^>]+content="([^"]+)"[^>]+property="og:image"', page)
    if og:
        images.insert(0, {"url": html.unescape(og.group(1)), "alt": "og:image", "caption": title})
    unescaped = page.replace("\\/", "/")
    videos = sorted({html.unescape(v) for v in re.findall(r'https?://[^"\'\s<>]+?\.(?:mp4|m3u8)(?:\?[^"\'\s<>]*)?', unescaped)})
    seen, unique = set(), []
    for image in images:
        key = re.sub(r"\?.*$", "", image["url"])
        if key not in seen:
            seen.add(key)
            unique.append(image)
    return title, links, unique, videos


def crawl(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "articles.jsonl"
    done = set()
    if record_path.is_file():
        done = {json.loads(line)["url"] for line in record_path.read_text(encoding="utf-8").splitlines() if line}
    seeds = [l.strip() for l in SEEDS.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    queue, queued = deque(seeds), set(seeds)
    last_hit, fetched = {}, 0
    with record_path.open("a", encoding="utf-8") as stream:
        while queue and fetched < args.max_articles:
            url = queue.popleft()
            site = site_of(url)
            if not site:
                continue
            wait = args.delay - (time.time() - last_hit.get(site, 0))
            if wait > 0:
                time.sleep(wait)
            last_hit[site] = time.time()
            try:
                page = get(url)
            except Exception as error:
                print(f"  ! {url}: {type(error).__name__}", flush=True)
                continue
            title, links, images, videos = extract(page, url)
            for link in sorted(links):
                if link not in queued:
                    queued.add(link)
                    queue.append(link)
            if url in done or not is_article(url):
                continue  # tag/listing pages only feed the queue
            text = (title + " " + " ".join(i["caption"] + " " + i["alt"] for i in images)).lower()
            if not any(word in text for word in WEAPON_WORDS):
                continue
            stream.write(json.dumps({"url": url, "site": site, "title": title, "images": images, "videos": videos},
                                    ensure_ascii=False) + "\n")
            stream.flush()
            fetched += 1
            if fetched % 25 == 0:
                print(f"  {fetched} articles, queue {len(queue)}", flush=True)
    print(f"crawl done: {fetched} new articles -> {record_path}")


def image_size(data):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
    if image is None:
        return None
    return image.shape[1] * 2, image.shape[0] * 2


def download(args):
    out = Path(args.out)
    articles = [json.loads(l) for l in (out / "articles.jsonl").read_text(encoding="utf-8").splitlines() if l]
    media = out / "media.jsonl"
    have = set()
    if media.is_file():
        have = {json.loads(l)["source_url"] for l in media.read_text(encoding="utf-8").splitlines() if l}
    (out / "images").mkdir(exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    with media.open("a", encoding="utf-8") as stream:
        for number, article in enumerate(articles, 1):
            case = hashlib.sha1(article["url"].encode()).hexdigest()[:10]
            for index, image in enumerate(article["images"][: args.max_images_per_article]):
                if image["url"] in have:
                    continue
                try:
                    data = get(image["url"], binary=True)
                except Exception:
                    continue
                if len(data) < args.min_bytes:
                    continue
                size = image_size(data)
                if size is None or min(size) < args.min_side:
                    continue  # thumbnails of related articles, icons
                digest = hashlib.sha256(data).hexdigest()
                suffix = ".png" if data[:4] == b"\x89PNG" else ".webp" if data[8:12] == b"WEBP" else ".jpg"
                target = out / "images" / f"{article['site'].split('.')[0]}_{case}_{index:02d}{suffix}"
                target.write_bytes(data)
                stream.write(json.dumps({"kind": "image", "path": str(target.relative_to(out)), "sha256": digest,
                                         "source_url": image["url"], "article": article["url"], "case_id": case,
                                         "caption": image["caption"], "alt": image["alt"], "title": article["title"]},
                                        ensure_ascii=False) + "\n")
                have.add(image["url"])
            for index, video in enumerate(article["videos"][: args.max_videos_per_article]):
                if video in have:
                    continue
                target = out / "videos" / f"{article['site'].split('.')[0]}_{case}_{index:02d}.mp4"
                command = [args.ffmpeg, "-y", "-loglevel", "error", "-user_agent", UA, "-i", video,
                           "-t", str(args.max_video_seconds), "-c", "copy", str(target)]
                try:
                    subprocess.run(command, check=True, timeout=300)
                except Exception:
                    target.unlink(missing_ok=True)
                    continue
                if target.is_file() and target.stat().st_size > 50_000:
                    stream.write(json.dumps({"kind": "video", "path": str(target.relative_to(out)),
                                             "source_url": video, "article": article["url"], "case_id": case,
                                             "title": article["title"]}, ensure_ascii=False) + "\n")
                    have.add(video)
            stream.flush()
            if number % 25 == 0:
                print(f"  {number}/{len(articles)} articles downloaded", flush=True)
    print("download done ->", media)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["crawl", "download"])
    parser.add_argument("--out", default=str(ROOT / "datasets" / "sharp_object" / "vn" / "raw"))
    parser.add_argument("--max-articles", type=int, default=600)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests to the same site")
    parser.add_argument("--max-images-per-article", type=int, default=12)
    parser.add_argument("--max-videos-per-article", type=int, default=2)
    parser.add_argument("--max-video-seconds", type=int, default=180)
    parser.add_argument("--min-bytes", type=int, default=15_000)
    parser.add_argument("--min-side", type=int, default=300, help="drop images whose short side is smaller")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    crawl(args) if args.step == "crawl" else download(args)


if __name__ == "__main__":
    main()
