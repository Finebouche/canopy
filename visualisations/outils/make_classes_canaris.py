"""
Writes visualisations/résultats/classes_des_canaris.html: the phrase classes of the four canaries (marron1, Giraudon et
al. 2021; llb3, llb11, llb16, Cohen 2022), each with EXAMPLES phrases to see (spectrogram) and to hear (audio), its
number of phrases, its median duration, its syllable period and, for the canaries annotated syllable by syllable
(llb), its median number of syllables per phrase.

The classes of the four canaries are ranged in families (the classes that sound alike from one canary to another), by
hand and, for what is not ranged by hand, automatically from three descriptions of a class, each over up to EMBEDDED
of its phrases:
  - its syllable period: for the canaries annotated syllable by syllable (llb), the median over its phrases of the
    time between the starts of two consecutive annotated syllables (the duration of a phrase of a single syllable);
    for marron1, annotated by phrase, an estimate (syllable_period): the first clear peak (after the first trough, at
    least 25 ms) of the autocorrelation of the loudness of a phrase, the whole phrase without peak. On the classes of
    llb, this estimate is within 25 % of the annotated period for 74 classes of 77 (the others: llb11 3, 23 ms for
    124; llb11 8, 29 ms for 60; llb3 15, 313 ms for 246);
  - its mean syllable (syllable_template): the mel spectrogram of a phrase (40 bands, 500 Hz to 10 kHz) cut every
    period from its quietest phase, each piece resampled to BINS columns, their median; two classes are compared by the
    best correlation of their mean syllables over the shifts in time and over shifts of up to FREQUENCY_SHIFT bands in
    frequency, each band of shift costing SHIFT_COST (the same syllable sung higher or lower);
  - six measures of its mean syllable (syllable_features): the slope of its frequency over the syllable (rising or
    falling), its mean frequency, its width in frequency, the share of the period that is loud, its number of
    harmonics and the excursion of its frequency; with the log-period, z-scored over the classes.
Distance between two classes (class_distances): 1 - the correlation + FEATURE_WEIGHT x the Euclidean distance of the
measures (divided by the square root of their number) + RHYTHM_WEIGHT x the difference of the z-scored log-periods.
Automatic families (tight_families): average linkage on this distance, cut at its quantile FAMILY_QUANTILE over the
pairs of classes to range, so that a family holds only classes close to one another; the classes left alone go to the
last family, ALONE. Check (2026-11): the nearest class is of the same family for 89 % of the 46 classes ranged by hand
in three families, as with the rhythm alone, and for 85 % against 77 % with the first distance (correlation without
shift in frequency + 0.8 x rhythm) when five groups read on the spectrograms (19 classes: rising, falling, plateau,
stacks of harmonics, K) are added. The families suggest classes to compare; they prove nothing.

By hand: GROUPS_FILE (classes_des_canaris/groupes.json) holds the families, a list of {"name", "classes": {canary:
[classes]}} in the order of the page; the classes it does not list go to the family ALONE. Without the file, the
families of MANUAL_FAMILIES come first, the others automatic, and the file is written. With "regroup", the named
families of the file are kept and the classes of its other families ranged again automatically. In the page, the
classes are moved by hand (dragged onto a family, or with the menu of a class) and the families created, named and
removed; the changes are kept in the browser, and the button "Télécharger groupes.json" saves them: put the file in
place of GROUPS_FILE and run this script again to write them into the page. When the file changes while the browser
holds families ranged by hand, the page keeps those of the browser, with the names of the file for the families that
share most of their classes with a named family of the file ("Revenir au fichier" shows those of the file).

Phrases: the annotated segments, silences (SIL) and noises (TRASH) ignored, the consecutive segments of the same label
merged (canopy.datasets.songs.song_phrases); the calls of marron1 are a class. Examples: phrases of distinct songs whose
durations are the closest to the median of their class, drawn with the seed SEED, cut to MAX_SECONDS and played with
CONTEXT seconds before and after, resampled to RATE Hz and normalized to the same peak.

Written in visualisations/résultats/classes_des_canaris/: <canary>/<class>_<k>.ogg (Ogg Vorbis) and <class>_<k>.webp
(spectrogram, 0 to MAX_HZ Hz, dB relative to the maximum of the extract, from -DB_RANGE to 0), and classes.pkl (not
tracked): the classes, their examples and their descriptions, read instead of the recordings unless "rebuild" is
given. The recordings must be in canopy/datasets (marron1: downloaded from Zenodo if needed; llb:
canopy.datasets.cohen_canaries).

Run from the root of canopy: PYTHONPATH=$PWD python visualisations/outils/make_classes_canaris.py [rebuild] [regroup]
"""
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import resample_poly
from scipy.spatial.distance import squareform

from canopy.datasets.songs import DATASETS_DIR, annotations_of_songs, download_canary_if_needed, song_phrases

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "visualisations" / "résultats" / "classes_des_canaris.html"
ASSETS = PAGE.with_suffix("")  # (visualisations/résultats/classes_des_canaris/)
GROUPS_FILE, CACHE = ASSETS / "groupes.json", ASSETS / "classes.pkl"
BIRDS = ["marron1", "llb3", "llb11", "llb16"]
SOURCES = {"marron1": "Giraudon et al. 2021, annoté par phrase", "llb3": "Cohen 2022, annoté par syllabe",
           "llb11": "Cohen 2022, annoté par syllabe", "llb16": "Cohen 2022, annoté par syllabe"}
EXAMPLES, EMBEDDED, SEED = 3, 40, 0
FREQUENCY_SHIFT, SHIFT_COST = 3, 0.02  # (bands of the mel spectrogram, and cost in correlation of each band of shift)
FEATURE_WEIGHT, RHYTHM_WEIGHT, FAMILY_QUANTILE = 0.5, 0.4, 0.05
ALONE = "sans équivalent trouvé"  # (name of the family of the classes left alone)
# families made by ear (2026-11), first in the page: (name, {canary: [classes]})
MANUAL_FAMILIES = [("notes plates", {"marron1": ["Q", "R", "T"], "llb3": ["5", "10"], "llb11": ["19", "22", "23", "25"],
                                     "llb16": ["6", "18", "19"]}),
                   ("trilles rapides à large bande", {"llb3": ["11", "18", "19"], "llb11": ["1", "2", "6", "13", "27"],
                                                      "llb16": ["17", "24", "27", "30"]}),
                   ("trilles", {"marron1": ["A"], "llb3": ["1", "2", "9", "14"], "llb11": ["7", "11", "18", "20"],
                                "llb16": ["1", "2", "3", "5", "11", "12", "13", "14", "16", "21", "26", "28", "29"]})]
RATE, CONTEXT, MAX_SECONDS = 22050, 0.12, 2.0
N_FFT, HOP, MAX_HZ, DB_RANGE = 512, 128, 10000, 70
FINE_HOP, BINS = 64, 16  # (hop of the mel spectrogram of the mean syllable, and its number of columns)


def annotations_and_audio(bird):
    """(annotations "wave", "start", "end", "syll", {wave: path of its recording}, ignored labels) of a canary."""
    if bird == "marron1":
        _, audio_dir = download_canary_if_needed()
        return annotations_of_songs(), {p.name: p for p in audio_dir.rglob("*.wav")}, ("SIL", "TRASH")
    folder = DATASETS_DIR / bird
    annotations = pd.concat([pd.read_csv(f, usecols=["wave", "start", "end", "syll"]) for f in sorted(folder.glob("*.csv"))],
                            ignore_index=True)
    annotations["syll"] = annotations["syll"].astype(str)
    return annotations, {p.name: p for p in folder.glob("*.wav")}, ("SIL",)


def classes_of(bird):
    """{label: [(wave, start, end, syllables, interval)]} of the phrases of a canary, and the paths of its recordings;
    syllables, the number of annotated segments of the phrase, and interval, the median time between the starts of two
    consecutive ones (the duration of a phrase of one segment)."""
    annotations, paths, ignored = annotations_and_audio(bird)
    classes = defaultdict(list)
    for wave, song in annotations.groupby("wave"):
        if wave not in paths:
            continue
        segments = song_phrases(song, wave, ignored, merge_repeats=False)
        for start, end, label in song_phrases(song, wave, ignored, merge_repeats=True):
            starts = sorted(s for s, e, l in segments if l == label and s >= start - 1e-6 and e <= end + 1e-6)
            interval = float(np.median(np.diff(starts))) if len(starts) > 1 else end - start
            classes[label].append((wave, start, end, len(starts), interval))
    return dict(classes), paths


def natural(label):
    """Sort key of a label: its numbers as numbers ("B2" before "B10", "2" before "10")."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(label))]


def safe(label):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(label))


class Recordings:
    """Recordings read once each, mono, resampled to RATE."""
    def __init__(self, paths):
        self.paths, self.cache = paths, {}

    def __call__(self, wave):
        if wave not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            audio, rate = sf.read(self.paths[wave], dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            self.cache[wave] = resample_poly(audio, RATE, rate).astype(np.float32) if rate != RATE else audio
        return self.cache[wave]


def extract(audio, start, end):
    """(extract with CONTEXT before and after, the phrase cut to MAX_SECONDS, peak-normalized with fades; its start and
    end in it, in seconds)."""
    stop = min(end, start + MAX_SECONDS)
    first, last = max(0, int((start - CONTEXT) * RATE)), min(len(audio), int((stop + CONTEXT) * RATE))
    clip = audio[first:last].copy()
    fade = min(len(clip) // 4, int(0.01 * RATE))
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        clip[:fade] *= ramp
        clip[-fade:] *= ramp[::-1]
    clip *= 0.9 / max(np.abs(clip).max(), 1e-9)
    return clip, start - first / RATE, stop - first / RATE


def spectrogram(clip, path):
    """The spectrogram of an extract saved as an image (low frequencies at the bottom)."""
    power = np.abs(librosa.stft(clip, n_fft=N_FFT, hop_length=HOP)) ** 2
    bins = int(MAX_HZ / (RATE / N_FFT)) + 1
    db = 10 * np.log10(power[:bins] / max(power[:bins].max(), 1e-12) + 1e-12)
    db = db[: bins // 2 * 2].reshape(bins // 2, 2, -1).mean(axis=1)  # (2 frequency bins per row: half the height)
    plt.imsave(path, np.flipud(db), cmap="magma", vmin=-DB_RANGE, vmax=0, format="webp", pil_kwargs={"quality": 70})


def syllable_period(clip):
    """Syllable period (s) of a phrase without context: see the docstring of the module."""
    mel = librosa.feature.melspectrogram(y=clip, sr=RATE, n_fft=N_FFT, hop_length=HOP, n_mels=40, fmin=500, fmax=MAX_HZ)
    db = librosa.power_to_db(mel, ref=np.max)
    envelope = db.max(axis=0) - db.max(axis=0).mean()  # (loudness: loudest band of each frame)
    ac = np.correlate(envelope, envelope, "full")[len(envelope) - 1:]
    ac = ac / max(ac[0], 1e-9)
    hop_s, period = HOP / RATE, len(clip) / RATE  # (no repetition found: one syllable, the whole phrase)
    trough = next((k for k in range(1, len(ac) - 1) if ac[k] <= ac[k - 1] and ac[k] <= ac[k + 1]), None)
    if trough is not None:
        peaks = [k for k in range(max(trough, int(0.025 / hop_s)), min(len(ac) - 1, int(0.8 / hop_s)))
                 if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.2]
        period = peaks[0] * hop_s if peaks else period
    return period


def syllable_template(clip, period):
    """(40, BINS) mean syllable of a phrase: its mel spectrogram (dB, -50 to 0 mapped to 0..1) cut every period from
    the quietest phase, each piece resampled to BINS columns, their median; the whole phrase when it has one syllable."""
    mel = librosa.feature.melspectrogram(y=clip, sr=RATE, n_fft=N_FFT, hop_length=FINE_HOP, n_mels=40, fmin=500, fmax=MAX_HZ)
    # (dB below the 99.9th percentile of the phrase, not its maximum: a click would hide all the rest)
    db = np.clip(librosa.power_to_db(mel, ref=np.percentile(mel, 99.9)), -50, 0) / 50 + 1
    frames, p = db.shape[1], period * RATE / FINE_HOP
    columns = (np.arange(BINS) + 0.5) / BINS

    def piece(start, length):
        t = np.clip(start + columns * length, 0, frames - 1.001)
        lo = t.astype(int)
        return db[:, lo] * (1 - (t - lo)) + db[:, lo + 1] * (t - lo)
    if frames < 2.2 * p or p < 3:
        return piece(0, frames - 1)
    envelope, count = db.max(axis=0), int((frames - p) // p)
    level = [np.mean(envelope[np.round(phase + np.arange(count + 1) * p).astype(int).clip(0, frames - 1)])
             for phase in range(int(p))]
    phase = int(np.argmin(level))
    pieces = [piece(phase + k * p, p) for k in range(count) if phase + (k + 1) * p < frames]
    return np.median(pieces, axis=0) if pieces else piece(0, frames - 1)


def aligned_median(templates):
    """Median of the mean syllables, each rolled in time to match the first one best."""
    reference = templates[0]
    return np.median([max((np.roll(t, shift, axis=1) for shift in range(BINS)), key=lambda r: float((r * reference).sum()))
                      for t in templates], axis=0)


def describe():
    """(entries, mean syllables, periods) of the classes of the four canaries, their examples written in ASSETS."""
    rng = np.random.default_rng(SEED)
    entries, templates, periods = [], [], []
    for bird in BIRDS:
        classes, paths = classes_of(bird)
        recordings = Recordings(paths)
        (ASSETS / bird).mkdir(parents=True, exist_ok=True)
        print(f"{bird}: {len(classes)} classes, {sum(map(len, classes.values()))} phrases", flush=True)
        for label in sorted(classes, key=natural):
            phrases = classes[label]
            durations = np.array([end - start for _, start, end, *_ in phrases])
            median = float(np.median(durations))
            order = rng.permutation(len(phrases))
            order = order[np.argsort(np.abs(durations[order] - median), kind="stable")]
            examples, songs = [], set()
            for i in order:
                wave, start, end, *_ = phrases[i]
                if wave in songs:
                    continue
                songs.add(wave)
                clip, phrase_start, phrase_end = extract(recordings(wave), start, end)
                name = f"{safe(label)}_{len(examples)}"
                sf.write(ASSETS / bird / f"{name}.ogg", clip, RATE, format="OGG", subtype="VORBIS")
                spectrogram(clip, ASSETS / bird / f"{name}.webp")
                total = len(clip) / RATE
                examples.append(dict(file=f"{bird}/{name}", seconds=round(total, 2), duration=round(end - start, 2),
                                     start=round(phrase_start / total, 4), end=round(phrase_end / total, 4), song=wave))
                if len(examples) == EXAMPLES:
                    break
            clips = []
            for i in rng.choice(len(phrases), min(EMBEDDED, len(phrases)), replace=False):
                wave, start, end, *_ = phrases[i]
                clip = recordings(wave)[int(start * RATE):int(min(end, start + MAX_SECONDS) * RATE)]
                if len(clip) >= 2 * N_FFT:
                    clips.append(clip)
            # (period: annotated for the canaries annotated syllable by syllable, estimated for marron1)
            intervals = [interval for *_, interval in phrases] if bird != "marron1" else []
            period = float(np.median(intervals if intervals else [syllable_period(clip) for clip in clips]))
            templates.append(aligned_median([syllable_template(clip, period) for clip in clips]))
            periods.append(period)
            syllables = [s for *_, s, _ in phrases] if bird != "marron1" else []
            entries.append(dict(bird=bird, label=str(label), count=len(phrases), median=round(median, 2),
                                period=round(1000 * period), estimated=not intervals, syllables=float(np.median(syllables)) if syllables else None,
                                examples=examples))
    return entries, np.array(templates), np.array(periods)


def shape_distances(templates):
    """1 - the best correlation of the mean syllables of two classes over the shifts in time and the shifts of up to
    FREQUENCY_SHIFT bands in frequency (the correlation over the bands they share, each band of shift costing
    SHIFT_COST): a matrix (classes, classes)."""
    def normalized(parts):
        parts = parts - parts.mean(axis=(1, 2), keepdims=True)
        return parts / np.maximum(np.linalg.norm(parts, axis=(1, 2), keepdims=True), 1e-9)  # (a flat one: far from all)

    bands = templates.shape[1]
    best = np.ones((len(templates), len(templates)))
    for shift in range(FREQUENCY_SHIFT + 1):
        low, high = normalized(templates[:, :bands - shift]), normalized(templates[:, shift:])
        for turn in range(BINS):
            best = np.minimum(best, SHIFT_COST * shift + 1 - np.einsum("ift,jft->ij", low, np.roll(high, turn, axis=2)))
    return np.minimum(best, best.T)


def syllable_features(template):
    """Six measures of a mean syllable (bands, BINS), in bands of the mel spectrogram: the slope of its frequency (the
    centroid of its power in each column) over the syllable from its quietest column, weighted by the loudness; its
    mean frequency; its width in frequency around the centroid; the share of the columns louder than half the loudest;
    the number of peaks of a loud column (harmonics); the excursion of its frequency (standard deviation)."""
    loudness = template.sum(axis=0)
    weight = loudness / loudness.sum()
    power = template ** 2 / np.maximum((template ** 2).sum(axis=0, keepdims=True), 1e-9)
    bands = np.arange(len(template))[:, None]
    centroid = (bands * power).sum(axis=0)
    width = np.sqrt(((bands - centroid) ** 2 * power).sum(axis=0))
    order = (np.arange(BINS) + np.argmin(loudness)) % BINS
    frequency, share = centroid[order] - (centroid * weight).sum(), weight[order]
    time = np.arange(BINS) - (np.arange(BINS) * share).sum()
    loud = loudness > 0.5 * loudness.max()
    peaks = [np.sum((column[1:-1] > column[:-2]) & (column[1:-1] >= column[2:]) & (column[1:-1] > 0.5 * column.max()))
             for column in template.T[loud]]
    return [(share * time * frequency).sum() / max((share * time ** 2).sum(), 1e-9), (centroid * weight).sum(),
            (width * weight).sum(), loud.mean(), np.mean(peaks), np.sqrt((share * frequency ** 2).sum())]


def class_distances(templates, periods):
    """The distance between the classes (see the docstring of the module): a matrix (classes, classes)."""
    features = np.column_stack([[syllable_features(template) for template in templates], np.log(periods)])
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    measures = np.sqrt(((features[:, None] - features[None]) ** 2).sum(axis=-1) / features.shape[1])
    return shape_distances(templates) + FEATURE_WEIGHT * measures + RHYTHM_WEIGHT * np.abs(features[:, None, -1] - features[None, :, -1])


def tight_families(distance, members, periods):
    """The classes members ranged by average linkage on distance, cut at its quantile FAMILY_QUANTILE over the pairs of
    members: (the families of several classes, from the fastest rhythm to the slowest, the classes left alone)."""
    members = list(members)
    if len(members) < 3:
        return [], members
    pairs = squareform(distance[np.ix_(members, members)], checks=False)
    labels = fcluster(linkage(pairs, "average"), np.quantile(pairs, FAMILY_QUANTILE), "distance")
    groups = [[i for i, g in zip(members, labels) if g == label] for label in sorted(set(labels))]
    families = sorted((group for group in groups if len(group) > 1), key=lambda group: np.mean(np.log(periods[group])))
    return families, [group[0] for group in groups if len(group) == 1]


def families_of(entries, templates, periods, regroup=False):
    """The families [{"name", "members": [index of the class]}]: those of GROUPS_FILE or, without the file or with
    regroup, the families by hand then the automatic ones, written to GROUPS_FILE (see the docstring of the module)."""
    index = {(e["bird"], e["label"]): i for i, e in enumerate(entries)}

    def members_of(classes):
        missing = [(bird, label) for bird, labels in classes.items() for label in labels if (bird, str(label)) not in index]
        if missing:
            raise ValueError(f"classes of the families by hand not found: {missing}")
        return [index[(bird, str(label))] for bird, labels in classes.items() for label in labels]

    saved = json.loads(GROUPS_FILE.read_text()) if GROUPS_FILE.exists() else None
    by_hand = MANUAL_FAMILIES if saved is None else [(family["name"], family["classes"]) for family in saved
                                                    if not regroup or (family["name"] and family["name"] != ALONE)]
    families = [dict(name=name, members=members_of(classes)) for name, classes in by_hand]
    ranged = [i for family in families for i in family["members"]]
    if len(ranged) != len(set(ranged)):
        raise ValueError(f"{GROUPS_FILE}: a class is in two families")
    rest = [i for i in range(len(entries)) if i not in set(ranged)]
    if saved is not None and not regroup:
        return families + ([dict(name=ALONE, members=rest)] if rest else [])
    automatic, alone = tight_families(class_distances(templates, periods), rest, periods)
    families += [dict(name="", members=members) for members in automatic] + ([dict(name=ALONE, members=alone)] if alone else [])
    GROUPS_FILE.write_text(json.dumps([dict(name=family["name"], classes={
        bird: [entries[i]["label"] for i in family["members"] if entries[i]["bird"] == bird]
        for bird in BIRDS if any(entries[i]["bird"] == bird for i in family["members"])}) for family in families],
        ensure_ascii=False, indent=1))
    return families


def build(rebuild=False, regroup=False):
    """(entries, families) of the page; the classes and their descriptions from CACHE unless rebuild."""
    if CACHE.exists() and not rebuild:
        entries, templates, periods = pickle.loads(CACHE.read_bytes())
    else:
        entries, templates, periods = describe()
        CACHE.write_bytes(pickle.dumps((entries, templates, periods)))
    return entries, families_of(entries, templates, periods, regroup)


def write(entries, families):
    counts = {bird: sum(e["bird"] == bird for e in entries) for bird in BIRDS}
    keys = [f"{e['bird']}|{e['label']}" for e in entries]
    structure = [dict(name=family["name"], members=[keys[i] for i in family["members"]]) for family in families]
    PAGE.write_text(TEMPLATE.replace("__DATA__", json.dumps(entries, ensure_ascii=False))
                    .replace("__FAMILIES__", json.dumps(structure, ensure_ascii=False))
                    .replace("__SOURCES__", json.dumps(SOURCES, ensure_ascii=False))
                    .replace("__TOTAL__", str(len(entries))).replace("__COUNTS__", escape(", ".join(f"{b} {n}" for b, n in counts.items()))))
    print(f"{PAGE} ({len(entries)} classes, {len(families)} families)")


TEMPLATE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Classes des quatre canaris</title>
<style>
/* written by visualisations/outils/make_classes_canaris.py */
:root{--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#898781;--bg:#fcfcfb;--bg-neutral:#eceae3;--border:#e1e0d9;--border-strong:#c3c2b7;
--marron1:#2a78d6;--llb3:#eb6834;--llb11:#1baf7a;--llb16:#6250d6}
@media (prefers-color-scheme: dark){:root{--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#898781;--bg:#151515;--bg-neutral:#232321;--border:#2c2c2a;--border-strong:#444441;
--marron1:#3987e5;--llb3:#d95926;--llb11:#199e70;--llb16:#9085e9}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text-primary);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:24px 16px 56px}
main{max-width:min(1500px,94%);margin:0 auto}
h1{font-size:22px;font-weight:600;margin:0 0 6px}
p.note{font-size:13px;color:var(--text-secondary);margin:0 0 10px;max-width:960px}
small{font-size:12px;font-weight:400;color:var(--text-secondary)}
.controls{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin:12px 0 4px;font-size:13px;color:var(--text-secondary);position:sticky;top:0;background:var(--bg);padding:6px 0;z-index:5;border-bottom:1px solid var(--border)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chips button{font:inherit;font-size:13px;padding:3px 10px;border-radius:999px;border:1px solid var(--border-strong);background:var(--bg);color:var(--text-primary);cursor:pointer;display:inline-flex;align-items:center;gap:6px}
.chips button[aria-pressed="true"]{background:var(--text-primary);border-color:var(--text-primary);color:var(--bg)}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}
#status{font-size:12px}#status.changed{color:var(--text-primary);font-weight:600}

/* a family */
.family{border:1px solid var(--border);border-radius:12px;margin:14px 0 0;background:var(--bg)}
.family>header{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 10px;padding:8px 12px;cursor:pointer;border-radius:12px}
.family>header h2{font-size:16px;font-weight:600;margin:0}
.family>header .caret{display:inline-block;width:12px;color:var(--text-muted)}
.family>header .birds{display:inline-flex;gap:4px;margin-left:auto;align-self:center}
.family.over{background:var(--bg-neutral);outline:2px dashed var(--border-strong)}
.family .body{padding:0 12px 12px}
.tool{font:inherit;font-size:11px;border:1px solid var(--border-strong);border-radius:6px;background:var(--bg);color:var(--text-secondary);padding:0 6px;cursor:pointer;max-width:170px}
.tool:hover{color:var(--text-primary)}
.newfamily{border:2px dashed var(--border-strong);border-radius:12px;padding:14px;text-align:center;color:var(--text-secondary);font-size:13px;margin:16px 0 0}
.newfamily.over{background:var(--bg-neutral)}
.hint{grid-column:1 / -1;font-size:12px;color:var(--text-muted);padding:6px 0}

/* the classes: cards (three examples) or tiles (one example) */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}
.card,.tile{border:1px solid var(--border);border-left:3px solid var(--c);border-radius:8px;background:var(--bg);transition:box-shadow .3s}
.card{padding:8px 10px 10px}.tile{padding:4px 6px 6px}
.card[draggable="true"],.tile[draggable="true"]{cursor:grab}
.dragging{opacity:.4}
.flash{box-shadow:0 0 0 3px var(--c, var(--border-strong))}
.card h4,.tile h4{font-size:13px;font-weight:600;margin:0;display:flex;flex-wrap:wrap;align-items:baseline;gap:5px}
.card h4 .dot,.tile h4 .dot{align-self:center}
.ex{position:relative;display:block;width:100%;margin:5px 0 0;padding:0;border:0;background:#000;border-radius:4px;overflow:hidden;cursor:pointer}
.ex img{display:block;width:100%;height:70px;object-fit:fill}
.tile .ex{margin-top:3px}.tile .ex img{height:46px}
.ex .span{position:absolute;bottom:0;height:3px;background:rgba(255,255,255,.75)}
.ex .head{position:absolute;top:0;bottom:0;width:2px;background:#fff;opacity:0}
.ex.playing .head{opacity:.9}
.ex .play{position:absolute;left:5px;top:4px;width:18px;height:18px;border-radius:50%;background:rgba(0,0,0,.55);color:#fff;font-size:9px;line-height:18px;text-align:center}
.ex .len{position:absolute;right:5px;top:4px;font-size:11px;color:#fff;background:rgba(0,0,0,.55);padding:0 5px;border-radius:4px;font-variant-numeric:tabular-nums}
.tile .ex .len{display:none}
.move{font:inherit;font-size:12px;border:1px solid var(--border-strong);border-radius:6px;background:var(--bg);color:var(--text-primary);padding:1px 4px;max-width:100%;margin-top:6px}
h2.bird{font-size:17px;font-weight:600;margin:24px 0 8px;display:flex;align-items:baseline;gap:10px}
@media (max-width:820px){main{max-width:none}.controls{position:static}}
</style>
</head>
<body>
<main>
<h1><a class="home-link" href="../index.html" title="Index des visualisations" aria-label="Index des visualisations"><svg viewBox="0 0 24 24" width="0.92em" height="0.92em" aria-hidden="true"><path d="M12 2.8 2 11.4h2.8V21h5.4v-6.2h3.6V21h5.4v-9.6H22z" fill="currentColor"/></svg></a>Classes de phrases des quatre canaris</h1>
<p class="note">__TOTAL__ classes (__COUNTS__), rangées en familles : à la main ou, sinon, automatiquement par le rythme et la forme de la syllabe moyenne. Dans une famille, les classes vont de la syllabe la plus rapide à la plus lente. Cliquer sur un spectrogramme (0 à 10 kHz) le fait écouter. Pour ranger une classe, la glisser sur une famille ; « Télécharger groupes.json » enregistre le rangement, à placer dans visualisations/résultats/classes_des_canaris/ avant de régénérer la page.</p>
<div class="controls">
<div class="chips" id="view"><button type="button" data-view="groups" aria-pressed="true">par famille</button><button type="button" data-view="birds" aria-pressed="false">par canari</button></div>
<div class="chips" id="density"><button type="button" data-density="compact" aria-pressed="true">compact</button><button type="button" data-density="detail" aria-pressed="false">détaillé</button></div>
<div class="chips" id="birds"></div>
<div class="chips" id="fold"><button type="button" id="fold-all">Tout replier</button><button type="button" id="unfold-all">Tout déplier</button></div>
<div class="chips" id="edit"><button type="button" id="add-family">Ajouter une famille</button><button type="button" id="download">Télécharger groupes.json</button><button type="button" id="reset">Revenir au fichier</button></div>
<span id="status"></span>
</div>
<div id="sections"></div>
</main>
<style id="home-link-style">.home-link{display:inline-flex;vertical-align:-0.08em;margin-right:.4em;color:inherit;text-decoration:none}@media print{.home-link{display:none}}</style>
<script>
const DATA = __DATA__;
const FILE_FAMILIES = __FAMILIES__;  // (the families of groupes.json: [{name, members: ["canary|class"]}])
const SOURCES = __SOURCES__;
const BIRDS = ['marron1', 'llb3', 'llb11', 'llb16'];
const shown = new Set(BIRDS);
let view = 'groups', density = 'compact';
const fmt = (v, d) => v.toLocaleString('fr-FR', {minimumFractionDigits: d, maximumFractionDigits: d});
const natural = s => s.split(/(\d+)/).map(p => /^\d+$/.test(p) ? p.padStart(6, '0') : p).join('');
const key = e => `${e.bird}|${e.label}`;
const BY_KEY = Object.fromEntries(DATA.map(e => [key(e), e]));
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const plural = (n, word) => `${n} ${word}${n > 1 ? 's' : ''}`;

// ---- the families: from the file of the page or, when they were changed in this browser since, from its storage
const STORE = 'canopy-classes-des-canaris-familles', OLD_STORE = 'canopy-classes-des-canaris-groupes';
const FINGERPRINT = JSON.stringify(FILE_FAMILIES), ALONE = 'sans équivalent trouvé';
const canonical = list => JSON.stringify(list.filter(f => f.members.length).map(f => ({name: f.name, members: [...f.members].sort()})));
let families = JSON.parse(FINGERPRINT);
const locate = k => families.findIndex(f => f.members.includes(k));
function prune() { families = families.filter(f => f.members.length || f.hand); }  // (the emptied families go, except those made by hand to be filled)
function complete() {  // (each class of the page in one family: the unknown ones dropped, the missing ones in the family of the classes left alone)
  const seen = new Set();
  for (const family of families) family.members = family.members.filter(k => BY_KEY[k] && !seen.has(k) && seen.add(k));
  const rest = DATA.map(key).filter(k => !seen.has(k));
  if (rest.length) families.push({name: ALONE, members: rest});
  prune();
}
function adoptNames() {  // (a family takes the name of the family of the file that shares most of its classes, unless it was named in this browser)
  for (const family of families) {
    if (family.renamed) continue;
    for (const other of FILE_FAMILIES) {
      const shared = family.members.filter(k => other.members.includes(k)).length;
      if (other.name && other.name !== ALONE && 2 * shared > family.members.length && 2 * shared > other.members.length) family.name = other.name;
    }
  }
}
function migrate(old) {  // (the groups made by hand in the first version of the page, kept in this browser: now families)
  for (const group of old.filter(g => g.name || g.hand)) {
    let family = group.name && families.find(f => f.name === group.name);
    if (!family) { family = {name: group.name, hand: true, members: []}; families.splice(families.filter(f => f.name).length, 0, family); }
    const wanted = new Set(group.members.filter(k => BY_KEY[k])), leaving = family.members.filter(k => !wanted.has(k));
    for (const other of families) other.members = other.members.filter(k => !wanted.has(k));
    family.members = [...wanted];
    if (leaving.length) {
      let rest = families.find(f => f.name === 'à reclasser');
      if (!rest) { rest = {name: 'à reclasser', hand: true, members: []}; families.push(rest); }
      rest.members.push(...leaving);
    }
  }
  prune();
}
try {
  const saved = JSON.parse(localStorage.getItem(STORE) || 'null');
  if (saved && saved.families.some(f => f.groups)) {  // (the version with sub-groups: its sub-groups merged)
    families = saved.families.map(f => ({name: f.name, hand: f.hand, open: f.open, members: f.groups.flatMap(g => g.members)}));
    complete();
  } else if (saved && saved.fingerprint === FINGERPRINT) families = saved.families;
  else if (saved && saved.changed !== false) { families = saved.families; complete(); adoptNames(); }  // (the file changed since, and the families were ranged by hand in this browser: they are kept)
  else if (!saved) { const old = JSON.parse(localStorage.getItem(OLD_STORE) || 'null'); if (old && Array.isArray(old.groups)) migrate(old.groups); }
} catch (error) { /* (no storage: the families of the file) */ }
function save() {
  const changed = canonical(families) !== canonical(FILE_FAMILIES), status = document.getElementById('status');
  try { localStorage.setItem(STORE, JSON.stringify({fingerprint: FINGERPRINT, changed, families})); } catch (error) { /* (not kept) */ }
  status.textContent = changed ? 'Rangement modifié dans ce navigateur, pas encore dans le fichier' : 'Rangement du fichier';
  status.className = changed ? 'changed' : '';
}
function changed() { prune(); save(); render(); }
function flash(id) { const node = document.getElementById(id); if (node) { node.classList.add('flash'); setTimeout(() => node.classList.remove('flash'), 1200); } }
function moveClass(k, f) {  // (to the family f; 'new': a new family)
  const from = locate(k);
  if (from === f) return;
  const family = f === 'new' ? {name: '', hand: true, members: []} : families[f];
  if (f === 'new') families.push(family);
  if (from >= 0) families[from].members = families[from].members.filter(m => m !== k);
  family.members.push(k);
  family.open = true;
  changed();
  flash(`c-${k}`);
}

// ---- an example: its spectrogram, played on a click
let current = null;  // (the example being played: {audio, button})
function stop() { if (current) { current.audio.pause(); current.button.classList.remove('playing'); current.button.querySelector('.play').textContent = '▶'; current = null; } }
function example(x) {
  const button = el('button', 'ex'); button.type = 'button'; button.title = `${x.song}, phrase de ${fmt(x.duration, 2)} s`;
  const img = el('img'); img.src = `classes_des_canaris/${x.file}.webp`; img.alt = 'spectrogramme'; img.loading = 'lazy'; img.draggable = false;
  const span = el('span', 'span'); span.style.left = `${100 * x.start}%`; span.style.width = `${100 * (x.end - x.start)}%`;
  const head = el('span', 'head'), play = el('span', 'play', '▶'), len = el('span', 'len', `${fmt(x.duration, 2)} s`);
  button.append(img, span, head, play, len);
  button.addEventListener('click', () => {
    if (current && current.button === button) { stop(); return; }
    stop();
    const audio = new Audio(`classes_des_canaris/${x.file}.ogg`);
    current = {audio, button};
    button.classList.add('playing'); play.textContent = '■';
    const tick = () => { if (current && current.audio === audio) { head.style.left = `${100 * audio.currentTime / x.seconds}%`; requestAnimationFrame(tick); } };
    audio.addEventListener('ended', stop);
    audio.play().then(tick).catch(stop);
  });
  return button;
}

// ---- a class: a tile (one example) or a card (three examples and the menu of its family)
function classBox(e) {
  const compact = density === 'compact', box = el('div', compact ? 'tile' : 'card'), k = key(e);
  box.id = `c-${k}`;
  box.style.setProperty('--c', `var(--${e.bird})`);
  const details = `${e.count.toLocaleString('fr-FR')} phrases · ${fmt(e.median, 2)} s · syllabe toutes les ${e.period} ms${e.estimated ? ' (estimée)' : ''}` + (e.syllables ? ` · ${fmt(e.syllables, 0)} syllabes` : '');
  const title = el('h4'), dot = el('span', 'dot'); dot.style.background = `var(--${e.bird})`;
  const position = locate(k);
  title.append(dot, `${e.bird} ${e.label}`, el('small', '', compact ? `${e.period} ms` + (view === 'birds' && position >= 0 ? ` · famille ${position + 1}` : '') : details));
  box.title = `${e.bird} ${e.label} : ${details}`;
  box.append(title, ...e.examples.slice(0, compact ? 1 : e.examples.length).map(example));
  if (!compact) {
    const menu = el('select', 'move'); menu.setAttribute('aria-label', `Famille de ${e.bird} ${e.label}`);
    families.forEach((family, f) => { const o = el('option', '', `Famille ${f + 1}${family.name ? ' : ' + family.name : ''} (${family.members.length})`); o.value = f; menu.append(o); });
    const fresh = el('option', '', 'nouvelle famille'); fresh.value = 'new'; menu.append(fresh);
    if (position >= 0) menu.value = position;
    menu.addEventListener('change', () => moveClass(k, menu.value === 'new' ? 'new' : Number(menu.value)));
    box.append(menu);
  }
  if (view === 'groups') {
    box.draggable = true;
    box.addEventListener('dragstart', event => { event.dataTransfer.setData('text/plain', k); event.dataTransfer.effectAllowed = 'move'; box.classList.add('dragging'); });
    box.addEventListener('dragend', () => box.classList.remove('dragging'));
  }
  return box;
}
function dropZone(zone, drop) {  // (a family or a zone receiving a dragged class)
  zone.addEventListener('dragover', event => { event.preventDefault(); event.stopPropagation(); event.dataTransfer.dropEffect = 'move'; zone.classList.add('over'); });
  zone.addEventListener('dragleave', event => { if (!zone.contains(event.relatedTarget)) zone.classList.remove('over'); });
  zone.addEventListener('drop', event => { event.preventDefault(); event.stopPropagation(); zone.classList.remove('over'); const k = event.dataTransfer.getData('text/plain'); if (BY_KEY[k]) drop(k); });
}
const sorted = entries => entries.sort((a, b) => BIRDS.indexOf(a.bird) - BIRDS.indexOf(b.bird) || natural(a.label).localeCompare(natural(b.label)));
function tool(text, action) { const b = el('button', 'tool', text); b.type = 'button'; b.addEventListener('click', event => { event.stopPropagation(); action(); }); return b; }
function stats(all) {
  if (!all.length) return 'vide';
  const birds = BIRDS.filter(b => all.some(e => e.bird === b)), periods = all.map(e => e.period).sort((a, b) => a - b);
  return `${plural(all.length, 'classe')} de ${plural(birds.length, 'canari')} · syllabe toutes les ${periods[0] === periods[periods.length - 1] ? periods[0] : periods[0] + ' à ' + periods[periods.length - 1]} ms`;
}
function birdDots(all) { const dots = el('span', 'birds'); for (const b of BIRDS.filter(b => all.some(e => e.bird === b))) { const d = el('span', 'dot'); d.style.background = `var(--${b})`; d.title = b; dots.append(d); } return dots; }

function render() {
  const sections = document.getElementById('sections');
  sections.replaceChildren();
  const grid = () => el('div', density === 'compact' ? 'tiles' : 'grid');
  if (view === 'birds') {
    for (const bird of BIRDS.filter(b => shown.has(b))) {
      const entries = sorted(DATA.filter(e => e.bird === bird));
      const h = el('h2', 'bird'), dot = el('span', 'dot'); dot.style.background = `var(--${bird})`;
      h.append(dot, bird, el('small', '', `${entries.length} classes, ${entries.reduce((s, e) => s + e.count, 0).toLocaleString('fr-FR')} phrases · ${SOURCES[bird]}`));
      const box = grid(); box.append(...entries.map(classBox));
      sections.append(h, box);
    }
    return;
  }
  families.forEach((family, f) => {
    const all = family.members.map(k => BY_KEY[k]).sort((a, b) => a.period - b.period || BIRDS.indexOf(a.bird) - BIRDS.indexOf(b.bird) || natural(a.label).localeCompare(natural(b.label)));
    const open = family.open !== false;
    const section = el('section', 'family'); section.id = `f-${f + 1}`;
    const header = el('header'), h = el('h2');
    h.append(el('span', 'caret', open ? '▾' : '▸'), ` Famille ${f + 1}` + (family.name ? ` : ${family.name}` : ''));
    header.append(h, el('small', '', stats(all)),
      tool(family.name ? 'renommer' : 'nommer', () => { const name = prompt(`Nom de la famille ${f + 1}`, family.name); if (name !== null) { family.name = name.trim(); family.renamed = true; changed(); } }));
    if (!all.length) header.append(tool('supprimer', () => { families.splice(f, 1); changed(); }));
    header.append(birdDots(all));
    header.addEventListener('click', () => { family.open = !open; save(); render(); });
    section.append(header);
    dropZone(section, k => moveClass(k, f));
    if (open) {
      const body = el('div', 'body'), box = grid();
      box.append(...all.filter(e => shown.has(e.bird)).map(classBox));
      if (!all.length) box.append(el('div', 'hint', 'Glisser une classe ici'));
      body.append(box);
      section.append(body);
    }
    sections.append(section);
  });
  const fresh = el('div', 'newfamily', 'Déposer une classe ici pour créer une nouvelle famille');
  dropZone(fresh, k => moveClass(k, 'new'));
  sections.append(fresh);
}
function renderControls() {
  document.getElementById('birds').replaceChildren(...BIRDS.map(bird => {
    const b = el('button'); b.type = 'button'; b.setAttribute('aria-pressed', shown.has(bird));
    const d = el('span', 'dot'); d.style.background = `var(--${bird})`; b.append(d, bird);
    b.addEventListener('click', () => { shown.has(bird) && shown.size > 1 ? shown.delete(bird) : shown.add(bird); stop(); renderControls(); render(); });
    return b;
  }));
}
function choice(id, attribute, apply) {
  document.getElementById(id).addEventListener('click', event => {
    const value = event.target.dataset[attribute]; if (!value) return;
    apply(value); stop();
    document.querySelectorAll(`#${id} button`).forEach(b => b.setAttribute('aria-pressed', b.dataset[attribute] === value));
    render();
  });
}
choice('view', 'view', value => { view = value; });
choice('density', 'density', value => { density = value; });
document.getElementById('fold-all').addEventListener('click', () => { families.forEach(f => { f.open = false; }); save(); render(); });
document.getElementById('unfold-all').addEventListener('click', () => { families.forEach(f => { f.open = true; }); save(); render(); });
document.getElementById('add-family').addEventListener('click', () => {
  families.splice(families.filter(f => f.name || f.hand).length, 0, {name: '', hand: true, open: true, members: []});
  changed();
});
document.getElementById('download').addEventListener('click', () => {
  const classes = members => Object.fromEntries(BIRDS.map(b => [b, members.filter(k => BY_KEY[k].bird === b).map(k => BY_KEY[k].label)]).filter(([, labels]) => labels.length));
  const file = families.filter(f => f.members.length).map(f => ({name: f.name, classes: classes(f.members)}));
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([JSON.stringify(file, null, 1)], {type: 'application/json'}));
  link.download = 'groupes.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
document.getElementById('reset').addEventListener('click', () => {
  if (!confirm('Revenir au rangement du fichier ? Les changements faits dans ce navigateur seront perdus.')) return;
  families = JSON.parse(FINGERPRINT); save(); render();
});
renderControls(); save(); render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    write(*build(rebuild="rebuild" in sys.argv[1:], regroup="regroup" in sys.argv[1:]))
