"""
Single syllables of marron1 for the GAN, as in Pagliarini et al. (2021): the 16 syllable types of their dataset
(LABELS, all of marron1), each syllable at SAMPLING_RATE, filtered by a high-pass filter (HIGH_PASS Hz, order 5) and
followed by silence up to SLICE_LENGTH samples (~1 s, the length of the WaveGAN), PER_LABEL syllables of each type.

The annotations of marron1 are those of the phrases (repetitions of a syllable): the syllables are cut in each phrase
by threshold crossings of the smoothed power of the song (the segmentation of their datasets.py: power of the song
filtered above DETECTION_HIGH_PASS Hz, averaged over SMOOTHING s; gaps shorter than MIN_SILENCE s closed, segments
shorter than MIN_SYLLABLE s removed), with a threshold relative to the level of the phrase (THRESHOLD times the 99th
percentile of its smoothed power; theirs is absolute, 1e5 on the int16 samples for most types, and misses the faint
types H, A and R of the recordings). Each syllable is cut from BEFORE samples before its onset to AFTER samples after
its offset (their margins), within its phrase, at most up to the onset of the next one (without its first BEFORE
samples) and from the offset of the previous one: a single syllable per sample.

Syllables of the train songs (the split of canopy.datasets.songs.split_canary_songs) for the GAN, those of the test
songs to evaluate it (decoders of the syllable type), cached in CANARY_DIR (syllables_<songs>.npz, not tracked).

Example:
    >>> data = syllable_dataset("train")  # x (16000, 16384) float32, label (16000,), labels, wave, onset
"""
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import butter, resample_poly, sosfiltfilt

from canopy.datasets.songs import (CANARY_DIR, SONGS_SAMPLING_RATE, annotations_of_songs, build_phrase_sequences,
                                   download_canary_if_needed, song_phrases, split_canary_songs)

LABELS = ("A", "B1", "B2", "C", "D", "E", "H", "J1", "J2", "L", "M", "N", "O", "Q", "R", "V")
SAMPLING_RATE = 16000
SLICE_LENGTH = 16384  # (samples of a GAN sample: ~1 s)
PER_LABEL = 1000  # (syllables of each type in the training set of the GAN, theirs)
HIGH_PASS = 700  # Hz, filter of the syllables (theirs, order 5)
DETECTION_HIGH_PASS = 500  # Hz, filter of the song for the segmentation (theirs: band 500-7999 Hz)
SMOOTHING = 0.01  # s
THRESHOLD = 0.02  # (of the 99th percentile of the smoothed power of the phrase)
MIN_SILENCE = 0.003  # s
MIN_SYLLABLE = 0.01  # s
BEFORE, AFTER = 200, 2000  # samples, margins of a syllable (theirs)


def high_pass(audio, cutoff, rate=SAMPLING_RATE):
    """audio filtered above cutoff Hz (Butterworth of order 5, zero phase)."""
    return sosfiltfilt(butter(5, cutoff, "highpass", fs=rate, output="sos"), audio)


def smoothed_power(audio, rate=SAMPLING_RATE):
    """Power of audio above DETECTION_HIGH_PASS Hz, averaged over SMOOTHING s (centered)."""
    width = int(round(rate * SMOOTHING))
    return np.convolve(high_pass(audio, DETECTION_HIGH_PASS, rate) ** 2, np.ones(width) / width, mode="same")


def syllable_bounds(power, rate=SAMPLING_RATE):
    """(onsets, offsets) in samples of the segments of power above THRESHOLD times its 99th percentile (gaps shorter
    than MIN_SILENCE closed, segments shorter than MIN_SYLLABLE removed)."""
    above = (power > THRESHOLD * np.percentile(power, 99)).astype(int)
    change = np.diff(np.concatenate([[0], above, [0]]))
    onsets, offsets = np.flatnonzero(change == 1), np.flatnonzero(change == -1)
    if len(onsets) > 1:
        kept = (onsets[1:] - offsets[:-1]) > MIN_SILENCE * rate
        onsets, offsets = onsets[np.concatenate([[True], kept])], offsets[np.concatenate([kept, [True]])]
    long_enough = (offsets - onsets) > MIN_SYLLABLE * rate
    return onsets[long_enough], offsets[long_enough]


def song_syllables(path, phrases, labels=LABELS):
    """[(label, onset (s), duration (s), audio)] of the syllables of the phrases [(start, end, label)] of labels of the
    song at path (audio at SAMPLING_RATE, filtered above HIGH_PASS Hz, cut with its margins)."""
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != SONGS_SAMPLING_RATE:
        raise ValueError(f"{path.name}: {rate} Hz, not {SONGS_SAMPLING_RATE} Hz")
    audio = resample_poly(audio.mean(axis=1), SAMPLING_RATE // 100, SONGS_SAMPLING_RATE // 100)
    power, filtered = smoothed_power(audio), high_pass(audio, HIGH_PASS)
    syllables = []
    for start, end, label in phrases:
        if label not in labels:
            continue
        first, last = int(start * SAMPLING_RATE), min(int(end * SAMPLING_RATE), len(audio))
        onsets, offsets = syllable_bounds(power[first:last])
        onsets, offsets = onsets + first, offsets + first
        for k, (onset, offset) in enumerate(zip(onsets, offsets)):
            begin = max(onset - BEFORE, offsets[k - 1] if k else first)
            stop = min(offset + AFTER, onsets[k + 1] - BEFORE if k + 1 < len(onsets) else last)
            syllables.append((label, onset / SAMPLING_RATE, (offset - onset) / SAMPLING_RATE,
                              filtered[begin:stop].astype(np.float32)))
    return syllables


def canary_split():
    """(train songs, test songs) of marron1 (those of the next-phrase prediction)."""
    return split_canary_songs(build_phrase_sequences(annotations_of_songs())["wave"])


def cut_syllables(waves, labels=LABELS, n_jobs=-1):
    """(DataFrame of the syllables of labels of the songs waves: columns wave, label, onset (s), duration (s); list of
    their audio)."""
    _, audio_dir = download_canary_if_needed()
    annotations, paths = annotations_of_songs(), {path.name: path for path in audio_dir.rglob("*.wav")}
    by_song = Parallel(n_jobs=n_jobs)(delayed(song_syllables)(paths[w], song_phrases(annotations, w), labels)
                                      for w in waves)
    rows = [(w, label, onset, duration) for w, syllables in zip(waves, by_song)
            for label, onset, duration, _ in syllables]
    return (pd.DataFrame(rows, columns=["wave", "label", "onset", "duration"]),
            [audio for syllables in by_song for *_, audio in syllables])


def padded(audios, length=SLICE_LENGTH):
    """(len(audios), length) array of audios followed by silence (cut at length), clipped to [-1, 1] (as WAV files)."""
    x = np.zeros((len(audios), length), np.float32)
    for i, audio in enumerate(audios):
        x[i, :min(len(audio), length)] = audio[:length]
    return np.clip(x, -1, 1)


def syllable_dataset(songs="train", per_label=PER_LABEL, seed=0):
    """Syllables of the train or test songs (songs): dict(x (n, SLICE_LENGTH) float32, label (n,) index in labels,
    labels (LABELS), wave, onset (s), duration (s)), at most per_label of each type drawn with seed (all of them for
    per_label None), cached."""
    cache = CANARY_DIR / f"syllables_{songs}_{per_label}_{seed}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    train_songs, test_songs = canary_split()
    table, audios = cut_syllables(train_songs if songs == "train" else test_songs)
    rng = np.random.default_rng(seed)
    kept = np.concatenate([rng.permutation(rows)[:per_label] if per_label else rows
                           for rows in (np.flatnonzero(table["label"] == label) for label in LABELS)])
    table = table.iloc[kept]
    data = dict(x=padded([audios[i] for i in kept]), label=table["label"].map(LABELS.index).to_numpy(),
                labels=np.array(LABELS), wave=table["wave"].to_numpy(str), onset=table["onset"].to_numpy(),
                duration=table["duration"].to_numpy())
    np.savez(cache, **data)
    return data
