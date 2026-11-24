"""
Number of syllables of each phrase of marron1, for the reservoirs that read one syllable per step
(canopy.studies.next_syllable.task). The annotations of marron1 are those of the phrases (repetitions of a syllable):
the syllables of a phrase are found as for the GAN of the syllables (canopy.studies.syllable_generation.syllables:
threshold crossings of the smoothed power of the song, the threshold relative to the level of the phrase), and a phrase
where none is found counts one syllable. Phrases: those of the next phrase with the calls (CANOPY_WITH_CALLS=1,
canopy.datasets.songs.song_phrases without SIL and TRASH, repeated labels merged), in their order in the song.

Result: outputs/next_syllable/analysis/syllable_counts.csv, one row per phrase: wave, phrase (index in the song),
label, start and end (s), n_syllables, detected (syllables found before the minimum of one).

Usage: python -m canopy.studies.next_syllable.segmentation
"""
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import resample_poly

from canopy.datasets.songs import SONGS_SAMPLING_RATE, annotations_of_songs, download_canary_if_needed, song_phrases
from canopy.settings import outputs_dir
from canopy.studies.syllable_generation.syllables import SAMPLING_RATE, smoothed_power, syllable_bounds

COUNTS = outputs_dir("next_syllable", "analysis") / "syllable_counts.csv"
IGNORED = ("SIL", "TRASH")  # (the phrases of the next phrase with the calls)


def song_counts(path, phrases):
    """[(phrase index, label, start, end, syllables found)] of the phrases [(start, end, label)] of the song at path."""
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != SONGS_SAMPLING_RATE:
        raise ValueError(f"{path.name}: {rate} Hz, not {SONGS_SAMPLING_RATE} Hz")
    power = smoothed_power(resample_poly(audio.mean(axis=1), SAMPLING_RATE // 100, SONGS_SAMPLING_RATE // 100))
    rows = []
    for k, (start, end, label) in enumerate(phrases):
        first, last = int(start * SAMPLING_RATE), min(int(end * SAMPLING_RATE), len(power))
        found = len(syllable_bounds(power[first:last])[0]) if last - first > 1 else 0
        rows.append((k, label, start, end, found))
    return rows


if __name__ == "__main__":
    _, audio_dir = download_canary_if_needed()
    annotations = annotations_of_songs()
    paths = {path.name: path for path in audio_dir.rglob("*.wav")}
    waves = sorted(set(annotations["wave"]) & set(paths))
    by_song = Parallel(n_jobs=-1)(delayed(song_counts)(paths[w], song_phrases(annotations, w, IGNORED)) for w in waves)
    table = pd.DataFrame([(w, *row) for w, rows in zip(waves, by_song) for row in rows],
                         columns=["wave", "phrase", "label", "start", "end", "detected"])
    table["n_syllables"] = table["detected"].clip(lower=1)
    COUNTS.parent.mkdir(parents=True, exist_ok=True)
    table[["wave", "phrase", "label", "start", "end", "n_syllables", "detected"]].to_csv(COUNTS, index=False)
    rates = table["detected"] / (table["end"] - table["start"])
    print(f"{len(waves)} songs, {len(table)} phrases, {table['n_syllables'].sum()} syllables "
          f"({table['n_syllables'].median():.0f} per phrase in median); no syllable found in {(table['detected'] == 0).sum()} "
          f"phrases; median rate {rates.median():.1f} syllables/s")
    print(table.groupby("label")["n_syllables"].median().sort_values().to_string())
    print("Saved to", COUNTS)
