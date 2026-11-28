"""
Download of the Canary Song Archive (https://canary.audio, CC BY-NC 4.0): recordings of domestic canaries made at home
by volunteers (Raspberry Pi, phone or handheld recorder), without annotation. The archive publishes monthly snapshots,
each with a manifest (https://dataset.canary.audio/datasets/<snapshot>/manifest.json: one item per recording, with its
bird, its owner, its device, its date, its duration, its location and the URL of its audio).

Snapshot 2026-09 (the last one published on 2026-11-28): 5 554 recordings, 24.5 h, WAV 44.1 kHz mono 16 bits, about
7.8 GB. 5 469 of them (24.0 h) are of a single bird, Xavier (Belgium); the 85 others are of 3 birds.

Written in canopy/datasets/canary_song_archive/ (not tracked): manifest.json (the manifest of the snapshot),
recordings.csv (one row per recording: "wave", "bird", "owner", "device", "date", "duration" in seconds, "country",
"city", "score", the score given by the archive to the upload) and audio/<wave> (<song_id>.wav). A recording already
there is not downloaded again: the command can be run again after an interruption. The server limits the rate of the
requests (HTTP 429 with the time to wait, a few seconds), which the download follows: count a few hours.

Usage: python -m canopy.datasets.canary_audio [snapshot, 2026-09 by default]
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import soundfile as sf

from canopy.datasets.songs import DATASETS_DIR

ARCHIVE_DIR = DATASETS_DIR / "canary_song_archive"
BASE_URL = "https://dataset.canary.audio/"
SNAPSHOT = "2026-09"
USER_AGENT = "canopy-dataset-downloader"
DOWNLOADS_IN_PARALLEL = 2  # (the server limits the rate of the requests: more of them only wait)
ATTEMPTS = 20


def fetch(url, path):
    """Downloads url to path, through a temporary file: path exists only when the download is complete. When the
    server limits the rate of the requests (HTTP 429), waits for the time it asks for and tries again, ATTEMPTS times
    at most."""
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as file:
                expected = response.headers.get("Content-Length")
                size = 0
                while chunk := response.read(1 << 20):
                    size += file.write(chunk)
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == ATTEMPTS - 1:
                raise
            time.sleep(float(error.headers.get("Retry-After") or 10) + 1)
    if expected is not None and size != int(expected):
        part.unlink()
        raise OSError(f"{url}: {size} bytes received, {expected} expected")
    part.rename(path)


def recordings_of(manifest):
    """The recordings of a manifest: a DataFrame, one row per recording, and the URL of the audio of each ("url")."""
    rows = []
    for item in manifest["items"]:
        url = item["urls"]["wav"]
        if not re.fullmatch(r"[0-9a-f-]{36}", item["song_id"]) or not url.startswith(BASE_URL):
            raise ValueError(f"unexpected item of the manifest: {item['song_id']!r}, {url!r}")
        location = item.get("location") or {}
        rows.append(dict(wave=f"{item['song_id']}.wav", bird=item.get("bird_name"), owner=item.get("owner_username"),
                         device=item.get("device_name"), date=item.get("date"), duration=item.get("duration_s"),
                         country=location.get("country"), city=location.get("city"), score=item.get("upload_score"),
                         url=url))
    return pd.DataFrame(rows)


def download(snapshot=SNAPSHOT):
    """Downloads the recordings of snapshot that are not in ARCHIVE_DIR yet. Returns (the recordings, the waves that
    could not be downloaded)."""
    audio_dir = ARCHIVE_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ARCHIVE_DIR / "manifest.json"
    if not manifest_path.exists() or json.loads(manifest_path.read_text())["snapshot"] != snapshot:
        manifest_path.unlink(missing_ok=True)
        fetch(f"{BASE_URL}datasets/{snapshot}/manifest.json", manifest_path)
    recordings = recordings_of(json.loads(manifest_path.read_text()))
    recordings.drop(columns="url").to_csv(ARCHIVE_DIR / "recordings.csv", index=False)
    missing = recordings[[not (audio_dir / wave).exists() for wave in recordings["wave"]]]
    print(f"{len(recordings)} recordings in the snapshot {snapshot}, {len(missing)} to download", flush=True)

    def fetch_or_report(row):
        try:
            fetch(row.url, audio_dir / row.wave)
        except OSError as error:
            return row.wave, str(error)

    failed = []
    with ThreadPoolExecutor(DOWNLOADS_IN_PARALLEL) as pool:
        for done, failure in enumerate(pool.map(fetch_or_report, missing.itertuples()), 1):
            if failure:
                failed.append(failure)
            if done % 500 == 0:
                print(f"{done} / {len(missing)} ({len(failed)} failed)", flush=True)
    return recordings, failed


if __name__ == "__main__":
    recordings, failed = download(*sys.argv[1:2])
    for wave, error in failed[:10]:
        print(f"failed: {wave}: {error}")
    present = [ARCHIVE_DIR / "audio" / wave for wave in recordings["wave"] if (ARCHIVE_DIR / "audio" / wave).exists()]
    hours = sum(sf.info(path).duration for path in present) / 3600
    print(f"{len(present)} of {len(recordings)} recordings in {ARCHIVE_DIR} ({hours:.1f} h of audio, "
          f"{sum(path.stat().st_size for path in present) / 1e9:.1f} GB), {len(failed)} failed")
