"""
Songs of the canary tasks (canary.next_phrase_prediction, canary.syllables_classification): MFCC features of the songs
of canary.datasets.songs.load_songs (cached in FEATURES_CACHE, with their train/test split and their phrases) and HAG
pretraining phrases.
"""
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from canary.datasets.songs import (CANARY_DIR, SONGS_SAMPLING_RATE, annotations_of_songs, build_phrase_sequences,
                                   canary_mfcc_config, load_songs, phrase_boundaries, song_phrases, songs_mfcc)
from hag.datasets.preprocessing import scale_data

RIDGES = 10.0 ** np.arange(-14, 3)
NB_PRETRAIN_PHRASES = 500
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs" / "canary"
FEATURES_CACHE = CANARY_DIR / f"boundary_features_{SONGS_SAMPLING_RATE}.pkl"
# MFCC without the floor at 80 dB below the maximum of the whole song (load_dataset(top_db=None))
NO_FLOOR_FEATURES_CACHE = FEATURES_CACHE.with_name(f"boundary_features_{SONGS_SAMPLING_RATE}_no_floor.pkl")


def phrase_colors():
    """Color of each phrase label of the songs (calls included) and of END (black), the same in all the figures: a
    distinct color for each label, those of tab20, Dark2 and three dark ones of tab20b, without their greys (kept for
    the silences and the unknown labels)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    pairs = build_phrase_sequences(annotations_of_songs())
    labels = sorted(set(pairs["current"]) | set(pairs["next"]))
    greys = {"#7f7f7f", "#c7c7c7", "#666666"}
    palette = [c for c in map(to_hex, list(plt.cm.tab20.colors) + list(plt.cm.Dark2.colors)) if c not in greys]
    palette += [to_hex(plt.cm.tab20b.colors[k]) for k in (0, 4, 12)]  # (dark blue, olive, dark red)
    colors = {label: palette[k % len(palette)] for k, label in enumerate(labels)}
    colors["END"] = "#000000"
    return colors


def pretrain_phrases(songs, train_songs, frame):
    """MFCC of NB_PRETRAIN_PHRASES phrases of the train songs (HAG's pretraining instances), drawn with numpy's RNG."""
    phrases = [songs[w]["mfcc"][int(start / frame):int(end / frame) + 1]
               for w in train_songs for start, end, _ in songs[w]["phrases"]]
    phrases = [p for p in phrases if len(p) > 1]
    indices = np.random.choice(len(phrases), min(NB_PRETRAIN_PHRASES, len(phrases)), replace=False)
    return np.array(phrases, dtype=object)[indices]


def load_dataset(top_db=80.0):
    """
    Songs of the tasks: songs[wave] = dict(mfcc (MFCC + deltas, MinMax-scaled (0, 1) on the train songs), phrases,
    boundaries), train and test songs, frame duration. The MFCC of all the songs are cached in FEATURES_CACHE (no
    audio kept). top_db: floor of the dB of the mel spectrogram below the maximum of the whole song (librosa's
    default, 80); None: no floor (cached in NO_FLOOR_FEATURES_CACHE), for the next-phrase prediction, since this
    maximum depends on the future of each frame. The phrases and boundaries are those of the current annotations
    (annotations_of_songs, the corrected files included), not those of the cache.
    """
    import pickle
    if top_db not in (80.0, None):
        raise ValueError(f"top_db={top_db}: only 80 (FEATURES_CACHE) and None (NO_FLOOR_FEATURES_CACHE) are cached")
    cache = FEATURES_CACHE if top_db is not None else NO_FLOOR_FEATURES_CACHE
    if cache.exists():
        with open(cache, "rb") as f:
            songs, train_songs, test_songs, frame = pickle.load(f)
    else:
        songs, train_songs, test_songs = load_songs()
        config = dict(canary_mfcc_config(SONGS_SAMPLING_RATE), top_db=top_db)
        mfccs = songs_mfcc([song["audio"] for song in songs.values()], config)
        for song, features in zip(songs.values(), mfccs):
            song["mfcc"] = features
            del song["audio"]
        frame = config["hop_length"] / SONGS_SAMPLING_RATE
        with open(cache, "wb") as f:
            pickle.dump((songs, train_songs, test_songs, frame), f)
    annotations = annotations_of_songs()
    for wave, song in songs.items():
        song["phrases"] = song_phrases(annotations, wave)
        song["boundaries"] = phrase_boundaries(song["phrases"])
    other_songs = [w for w in songs if w not in set(train_songs)]
    train, _, other = scale_data([songs[w]["mfcc"] for w in train_songs], None,
                                 [songs[w]["mfcc"] for w in other_songs], MinMaxScaler(feature_range=(0, 1)), True)
    for wave, mfcc in zip(train_songs + other_songs, train + other):
        songs[wave]["mfcc"] = mfcc
    return songs, train_songs, test_songs, frame
