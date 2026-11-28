"""
Download of the recordings of the Atlantic canary (Serinus canaria: the wild canaries of the Canary Islands, Madeira
and the Azores) of xeno-canto (https://xeno-canto.org/species/Serinus-canaria): field recordings, without annotation of
the phrases, each under its own Creative Commons licence (column "licence": most of them non-commercial, some without
derivatives).

The list of the recordings comes from the API v3 of xeno-canto (https://xeno-canto.org/explore/api), which needs a
personal key (given on https://xeno-canto.org/account): the variable CANOPY_XENO_CANTO_KEY. The audio files need no
key.

Written in canopy/datasets/xeno_canto_canaries/ (not tracked): recordings.json (the recordings as given by the API, all
their fields), recordings.csv (one row per recording: "wave", "recordist", "country", "locality", "date", "type",
"quality" from A to E, "length", "licence", "also", the other species heard) and audio/<wave> (XC<id>.<extension>, the
file as uploaded, MP3 for most). A recording already there is not downloaded again.

Usage: CANOPY_XENO_CANTO_KEY=<key> python -m canopy.datasets.xeno_canto
"""
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from canopy.datasets.songs import DATASETS_DIR
from canopy.settings import XENO_CANTO_KEY

XENO_CANTO_DIR = DATASETS_DIR / "xeno_canto_canaries"
API_URL = "https://xeno-canto.org/api/3/recordings"
GENUS, SPECIES = "Serinus", "canaria"
QUERIES = (f'sp:"{GENUS} {SPECIES}"', f"gen:{GENUS} sp:{SPECIES}")  # (the species, as the API v3 then the v2 wrote it)
USER_AGENT = "canopy-dataset-downloader"
SECONDS_BETWEEN_DOWNLOADS = 1.0


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def recordings_of_the_species(key):
    """The recordings of the species, as given by the API (a list of dicts): all the pages of the first query of
    QUERIES that has an answer, the recordings of other species removed."""
    for query in QUERIES:
        recordings, page, pages = [], 1, 1
        while page <= pages:
            answer = json.loads(get(f"{API_URL}?{urllib.parse.urlencode(dict(query=query, key=key, page=page))}"))
            recordings += answer.get("recordings", [])
            pages, page = int(answer.get("numPages", 1)), page + 1
        recordings = [r for r in recordings if (r.get("gen"), r.get("sp")) == (GENUS, SPECIES)]
        if recordings:
            return recordings
    raise ValueError(f"no recording of {GENUS} {SPECIES} for the queries {QUERIES}")


def wave_of(recording):
    """Name of the audio file of a recording: XC<id> and the extension of the uploaded file."""
    if not re.fullmatch(r"\d+", str(recording["id"])):
        raise ValueError(f"unexpected id: {recording['id']!r}")
    extension = re.search(r"\.(mp3|wav|flac|ogg|m4a)$", str(recording.get("file-name", "")).lower())
    return f"XC{recording['id']}{extension.group(0) if extension else '.mp3'}"


def download():
    """Downloads the recordings of the species that are not in XENO_CANTO_DIR yet. Returns (the recordings, the waves
    that could not be downloaded)."""
    if not XENO_CANTO_KEY:
        raise SystemExit("CANOPY_XENO_CANTO_KEY is not set: the key of the API is on https://xeno-canto.org/account")
    audio_dir = XENO_CANTO_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    recordings = recordings_of_the_species(XENO_CANTO_KEY)
    (XENO_CANTO_DIR / "recordings.json").write_text(json.dumps(recordings, ensure_ascii=False, indent=1))
    pd.DataFrame([dict(wave=wave_of(r), recordist=r.get("rec"), country=r.get("cnt"), locality=r.get("loc"),
                       date=r.get("date"), type=r.get("type"), quality=r.get("q"), length=r.get("length"),
                       licence=r.get("lic"), also="; ".join(r.get("also") or [])) for r in recordings]
                 ).to_csv(XENO_CANTO_DIR / "recordings.csv", index=False)
    failed = []
    for recording in recordings:
        path = audio_dir / wave_of(recording)
        if path.exists():
            continue
        try:
            content = get(f"https://xeno-canto.org/{recording['id']}/download")
            path.with_name(path.name + ".part").write_bytes(content)
            path.with_name(path.name + ".part").rename(path)
        except OSError as error:
            failed.append((path.name, str(error)))
        time.sleep(SECONDS_BETWEEN_DOWNLOADS)
    return recordings, failed


if __name__ == "__main__":
    recordings, failed = download()
    for wave, error in failed[:10]:
        print(f"failed: {wave}: {error}")
    present = [r for r in recordings if (XENO_CANTO_DIR / "audio" / wave_of(r)).exists()]
    print(f"{len(present)} of {len(recordings)} recordings in {XENO_CANTO_DIR}, {len(failed)} failed")
