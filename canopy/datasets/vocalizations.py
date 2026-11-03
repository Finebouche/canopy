"""
Annotated vocalizations of several species, for the analyses of their segments (the outliers of the notebook
"2. Outliers study"): the canary songs of marron1 (canopy.datasets.songs) and, each in its own directory of
canopy/datasets (not tracked), the recordings (WAV) and their annotations (a CSV for each recording, columns "wave",
"start", "end", "syll", one row per annotated segment) of:
  - bengalese_gy6or6: songs of the Bengalese finch gy6or6 (32 kHz, 113 songs, 11 syllables);
  - kerguelen2014: blue and fin whale calls at Kerguelen in 2014 (250 Hz, 167 recordings of an hour, 3 calls);
  - llb3: songs of the canary llb3 (44.1 kHz, 2655 songs, 20 syllables, silences SIL);
  - mouse_usv: ultrasonic vocalizations of mice (300 kHz, 51 recordings, a single label: no label to check).

The segments of a recording (load_vocalizations: "phrases") are its annotated segments, SIL ignored, and for the canary
songs the consecutive segments of the same label merged (phrases). Their features: 13 MFCC and their deltas and
delta-deltas (canopy.datasets.songs.songs_mfcc) without floor of the dB, with a window and a band of frequencies suited
to the species (MFCC below), cached in the directory of the dataset (mfcc_no_floor.pkl; the canary songs: the cache of
canopy.common.load_dataset).

Listening (denoised): the audio of a recording with its background noise removed by spectral subtraction (Wiener
gain): the noise of each frequency is measured in the parts of the recording without annotated segment
(noise_profile), and each bin of the spectrogram attenuated by the share of this noise in its power (the faint whale
calls of kerguelen2014, near the level of the noise: their band, 15-80 Hz, has the chorus of distant whales).
"""
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter

from canopy.datasets.songs import DATASETS_DIR, SONGS_SAMPLING_RATE, canary_mfcc_config, song_phrases, songs_mfcc

CANARY = "canary_marron1"


@dataclass(frozen=True)
class Vocalizations:
    """Settings of a dataset: mfcc, the librosa arguments of its MFCC (canopy.datasets.songs.songs_mfcc, at its
    sampling rate sr); merge_repeats, whether the consecutive segments of the same label are merged (canary phrases);
    spectrogram, (n_fft, hop_length, highest frequency in Hz) of the spectrograms shown; context, seconds of silence
    played before and after a segment, at most; speed, playback speed of the audio (sped up or slowed down to be
    audible and playable by a browser: infrasounds, ultrasounds); denoise, whether the audio listened to is denoised by
    default (denoised)."""
    mfcc: dict
    merge_repeats: bool = False
    spectrogram: tuple = (1024, 256, 10000)
    context: float = 0.2
    speed: float = 1.0
    ignored_labels: tuple = field(default=("SIL",))
    denoise: bool = False


DATASETS = {
    CANARY: Vocalizations(dict(canary_mfcc_config(SONGS_SAMPLING_RATE), top_db=None), merge_repeats=True),
    "bengalese_gy6or6": Vocalizations(dict(sr=32000, n_mfcc=13, n_fft=512, win_length=512, hop_length=128, fmin=500,
                                           fmax=10000, n_mels=40, lifter=40, deltas=True, top_db=None),
                                      spectrogram=(512, 64, 10000), context=0.1),
    "kerguelen2014": Vocalizations(dict(sr=250, n_mfcc=13, n_fft=256, win_length=256, hop_length=64, fmin=10, fmax=125,
                                        n_mels=32, lifter=40, deltas=True, top_db=None),
                                   spectrogram=(256, 16, 125), context=5.0, speed=16.0, denoise=True),
    "llb3": Vocalizations(dict(canary_mfcc_config(44100), top_db=None), merge_repeats=True),
    "mouse_usv": Vocalizations(dict(sr=300000, n_mfcc=13, n_fft=1024, win_length=1024, hop_length=256, fmin=20000,
                                    fmax=120000, n_mels=64, lifter=40, deltas=True, top_db=None),
                               spectrogram=(1024, 128, 125000), context=0.05, speed=1 / 16),
}


def recording_mfcc(path, config):
    """MFCC (frames, 39) of the recording at path, at the sampling rate of config."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != config["sr"]:
        raise ValueError(f"{path.name}: {rate} Hz, not the {config['sr']} Hz of the MFCC of its dataset")
    return songs_mfcc([audio.mean(axis=1)], config, nb_jobs=1)[0].astype(np.float32)


def mfcc_cache(name):
    """Path of the cache of the MFCC of the dataset name (canary_marron1: that of canopy.common.load_dataset)."""
    if name == CANARY:
        from canopy.common import NO_FLOOR_FEATURES_CACHE
        return NO_FLOOR_FEATURES_CACHE
    return DATASETS_DIR / name / "mfcc_no_floor.pkl"


def mfcc_cached(name):
    """Whether the MFCC of the dataset name are cached."""
    return mfcc_cache(name).exists()


def dataset_mfcc(name):
    """({wave: MFCC (frames, 39)}, frame duration (s)) of the recordings of the dataset name (not canary_marron1), from
    its cache, computed and cached if missing."""
    settings, audio_dir, cache = DATASETS[name], DATASETS_DIR / name, mfcc_cache(name)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    paths = sorted(audio_dir.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No recording in {audio_dir}: extract the archive of {name} there")
    print(f"MFCC of the {len(paths)} recordings of {name} (cached in {cache}) ...", flush=True)
    features = Parallel(n_jobs=-1)(delayed(recording_mfcc)(path, settings.mfcc) for path in paths)
    mfccs, frame = {path.name: x for path, x in zip(paths, features)}, settings.mfcc["hop_length"] / settings.mfcc["sr"]
    with open(cache, "wb") as f:
        pickle.dump((mfccs, frame), f)
    return mfccs, frame


def prepare_mfcc(name):
    """The MFCC of the dataset name, computed and cached if they are not (canary_marron1: canopy.common.load_dataset,
    the audio downloaded from Zenodo if needed)."""
    if name == CANARY:
        from canopy.common import load_dataset
        load_dataset(top_db=None)
    else:
        dataset_mfcc(name)


def load_vocalizations(name, max_songs=None, seed=0):
    """Recordings of the dataset name (DATASETS): dict(songs={wave: {"mfcc": (frames, 39)}}, train_songs, test_songs
    (the split of the canary songs; the other datasets: all their recordings, no test), frame (s), annotations
    (DataFrame "wave", "start", "end", "syll"), wav_paths {wave: path}, phrases {wave: [(start, end, label)]}, settings
    (Vocalizations)), their MFCC computed if they are not cached. max_songs: at most this number of recordings, drawn
    with seed (the large datasets)."""
    settings = DATASETS[name]
    if name == CANARY:
        from canopy.common import load_dataset
        from canopy.datasets.songs import annotations_of_songs, download_canary_if_needed
        songs, train_songs, test_songs, frame = load_dataset(top_db=None)
        annotations = annotations_of_songs()
        _, audio_dir = download_canary_if_needed()
    else:
        audio_dir = DATASETS_DIR / name
        mfccs, frame = dataset_mfcc(name)
        annotations = pd.concat([pd.read_csv(f, usecols=["wave", "start", "end", "syll"])
                                 for f in sorted(audio_dir.glob("*.csv"))], ignore_index=True)
        annotations["syll"] = annotations["syll"].astype(str)
        annotated = set(annotations["wave"])
        songs = {w: {"mfcc": x} for w, x in mfccs.items() if w in annotated}
        train_songs, test_songs = sorted(songs), []
    if max_songs is not None and len(songs) > max_songs:
        kept = set(np.random.default_rng(seed).choice(sorted(songs), max_songs, replace=False))
        train_songs, test_songs = [w for w in train_songs if w in kept], [w for w in test_songs if w in kept]
        songs = {w: songs[w] for w in train_songs + test_songs}
        annotations = annotations[annotations["wave"].isin(kept)]
    by_song = dict(tuple(annotations.groupby("wave")))
    phrases = {w: song_phrases(by_song[w], w, settings.ignored_labels, settings.merge_repeats) if w in by_song else []
               for w in train_songs + test_songs}
    wav_paths = {path.name: path for path in audio_dir.rglob("*.wav")}
    return dict(songs=songs, train_songs=train_songs, test_songs=test_songs, frame=frame, annotations=annotations,
                wav_paths=wav_paths, phrases=phrases, settings=settings)



MIN_NOISE_FRAMES = 20  # frames of the STFT without segment, at least, for a noise profile
GAIN_FLOOR = 0.05  # lowest gain of the denoising (-26 dB: some background left, fewer artifacts)
POWER_SMOOTHING = (3, 5)  # (frequency bins, frames) of the smoothing of the power of the extract (denoising)


def noise_profile(audio, rate, segments, n_fft):
    """Mean power of each frequency of the STFT of audio (a whole recording, at rate; window n_fft, hop n_fft / 4) over
    its frames outside the segments [(start, end, ...)] (their windows included): its background noise; None with fewer
    than MIN_NOISE_FRAMES such frames."""
    import librosa
    power = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=n_fft // 4)) ** 2
    times = np.arange(power.shape[1]) * (n_fft // 4) / rate  # (centers of the frames)
    half = n_fft / 2 / rate
    noise = np.ones(len(times), bool)
    for start, end, *_ in segments:
        noise &= (times < start - half) | (times > end + half)
    return power[:, noise].mean(axis=1) if noise.sum() >= MIN_NOISE_FRAMES else None


def denoised(audio, profile, n_fft, floor=GAIN_FLOOR, smoothing=POWER_SMOOTHING):
    """audio (an extract of a recording) denoised by spectral subtraction (Wiener gain): each bin of its STFT multiplied
    by 1 - noise power / its power (profile of noise_profile; the power smoothed over smoothing (frequency bins,
    frames), against the musical noise), at least floor."""
    import librosa
    S = librosa.stft(audio, n_fft=n_fft, hop_length=n_fft // 4)
    power = uniform_filter(np.abs(S) ** 2, size=smoothing, mode="nearest")
    gain = np.clip(1 - profile[:, None] / np.maximum(power, 1e-30), floor, 1)
    return librosa.istft(S * gain, hop_length=n_fft // 4, length=len(audio)).astype(np.float32)
