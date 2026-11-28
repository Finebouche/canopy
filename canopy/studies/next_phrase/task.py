"""
Prediction of the next phrase of canary songs from their audio, in the configuration of Maxime Canonne's next-phrase
prediction (ESN reading the phrase labels of the song, with START and END), but with the MFCC of the song as input: the
reservoir runs over the whole song (no reset between the phrases) and, at the end of each phrase, the readout predicts
the label of the next one (END after the last phrase); before the first phrase (Maxime's START), it predicts the first
one. Phrases as in canopy.datasets.songs.load_songs: SIL, TRASH and call ignored, repeated labels merged; with
CANOPY_WITH_CALLS=1 (canopy.settings.CALLS), the calls are phrases too (SIL and TRASH ignored). With
CANOPY_RESET_EACH_PHRASE=1 (RESET), the reservoirs hear the current phrase only, from a zero state (as the Canary task
of hag), instead of the whole song: the state of a point is the last one of a run on the frames of the phrase that ends
there (the zero state before the first phrase, or for a phrase shorter than the delay of the points). Input: the 13
MFCC, without their deltas (their effect, with and without leak: outputs/next_phrase/ablation_deltas_leak).

No input after the prediction point: the MFCC are computed without the floor of the dB at 80 dB below the maximum of
the whole song (librosa's default, which depends on the future of each frame: canopy.common.load_dataset(top_db=None)),
and the MFCC frames are centered (half a window, 11.6 ms, of audio after their time), so the state used to predict
what follows the time t is the one of the last frame whose window ends before t (DELAY earlier).

Readout: ridge regression on the one-hot next label at the prediction points, on the state at each point
(reservoirpy.nodes.Ridge, solver CANOPY_RIDGE_SOLVER, default "svd"), its coefficient chosen by the 3-fold
cross-validation by song on the train songs of the studies (canopy.studies.next_phrase.study.choose_ridge), then trained
on all the train songs and scored on the test songs (evaluate). Metrics as Maxime's (metrics): accuracy and macro F1 of
all the points, without the last one of each song ("skip_last": its target is END), and of the phrases only ("no_end":
points whose target is a phrase, argmax over the phrases), and the share of the ends of the songs found (end_recall).
(Readouts on other windows of states, trained online or of order 2: bachue.)
"""
import numpy as np
from reservoirpy.nodes import Ridge
from sklearn.metrics import f1_score

from canopy.common import load_dataset, pretrain_phrases
from canopy.datasets.songs import SONGS_SAMPLING_RATE, annotations_of_songs, canary_mfcc_config, song_phrases
from canopy.models.reservoirs import CHUNK, build, state_chunks, states
from canopy.settings import CALLS, DATASET_NAME, RESET, RIDGE_SOLVER

START, END = "START", "END"
N_STATIC = 13  # MFCC, before their deltas and delta-deltas in the cached features
DELAY = canary_mfcc_config(SONGS_SAMPLING_RATE)["win_length"] / 2 / SONGS_SAMPLING_RATE


def prepare(songs, train_songs, frame):
    """
    The 13 MFCC only (without the deltas of the cached features) and, for each song, its phrases
    songs[w]["next_phrases"] (with the calls if CALLS) and the prediction points: songs[w]["points"] (frame of each
    point, -1 for the initial zero state), songs[w]["starts"] (first frame of the phrase ending at each point, -1 for
    the point before the first phrase) and songs[w]["next"] (label of the next phrase, END after the last one).
    Returns the classes (the phrase labels and END).
    """
    annotations = annotations_of_songs() if CALLS else None
    for song in songs.values():
        song["mfcc"] = np.ascontiguousarray(song["mfcc"][:, :N_STATIC])
    for wave, song in songs.items():
        phrases = song_phrases(annotations, wave, ("SIL", "TRASH")) if CALLS else [tuple(p) for p in song["phrases"]]
        song["next_phrases"] = phrases
        labels = [label for _, _, label in phrases]
        times = [phrases[0][0]] + [end for _, end, _ in phrases]  # before the 1st, after each phrase
        song["points"] = np.array([min(int(np.floor((t - DELAY) / frame)), len(song["mfcc"]) - 1) for t in times])
        song["points"] = np.maximum(song["points"], -1)
        song["starts"] = np.array([-1] + [int(np.floor(start / frame)) for start, _, _ in phrases])
        song["next"] = np.array(labels + [END], dtype=object)
    return np.array(sorted({label for song in songs.values() for label in song["next"]} - {END}) + [END], dtype=object)


def load_task_dataset():
    """(songs, train songs, test songs, frame, classes) of the canary CANOPY_DATASET (canopy.settings.DATASET_NAME): the
    MFCC without the floor of the dB (canopy.common.load_dataset with top_db=None) and the prediction points (prepare)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None, dataset=DATASET_NAME)
    return songs, train_songs, test_songs, frame, prepare(songs, train_songs, frame)


def point_states(reservoirs, songs, waves):
    """{wave: (points, units)} states of the reservoirs (the inputs without reservoir) at the prediction points of the
    songs: the state at the frame of each point (the zero state at the point -1), or, with RESET, the last state of a
    run from a zero state on the frames of the phrase that ends there (the zero state without frame)."""
    result = {}
    if not RESET:
        for part, part_states in state_chunks(reservoirs, songs, waves):
            for x, w in zip(part_states, part):
                x = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float64), np.asarray(x, dtype=np.float64)])
                result[w] = x[songs[w]["points"] + 1]  # (row 0: the initial zero state, point -1)
        return result
    for i in range(0, len(waves), CHUNK):
        part = waves[i:i + CHUNK]
        runs = states(reservoirs, [songs[w]["mfcc"][start:point + 1] for w in part
                                   for start, point in zip(songs[w]["starts"], songs[w]["points"])
                                   if 0 <= start <= point])
        zero = np.zeros(runs[0].shape[1])
        runs = iter(runs)
        for w in part:
            result[w] = np.array([np.asarray(next(runs)[-1], dtype=np.float64) if 0 <= start <= point else zero
                                  for start, point in zip(songs[w]["starts"], songs[w]["points"])])
    return result


def one_hot(songs, waves, classes):
    """(points, classes) one-hot next labels of the points of the songs, in the order of the waves."""
    labels = np.concatenate([songs[w]["next"] for w in waves])
    return (labels[:, None] == classes[None, :]).astype(np.float64)


def fit_readout(states_of, songs, waves, classes, ridge):
    """The ridge readout of the next label trained on the points of the songs waves (states_of: point_states; the
    rows in the order of states_of, that of the runs of the reservoirs)."""
    wanted = set(waves)
    order = [w for w in states_of if w in wanted]
    return Ridge(ridge=ridge, solver=RIDGE_SOLVER).fit(np.vstack([states_of[w] for w in order]),
                                                         one_hot(songs, order, classes))


def metrics(scores, true, classes):
    """Maxime's metrics of the scores (points, classes) of the next labels true (points,)."""
    predicted = classes[scores.argmax(axis=1)]
    phrase = true != END
    no_end = classes[:-1][scores[:, :-1].argmax(axis=1)]  # (END is the last class)

    def f1(t, p):
        return f1_score(t.astype(str), p.astype(str), average="macro", zero_division=0)
    return dict(accuracy=np.mean(predicted == true), f1_macro=f1(true, predicted),
                accuracy_skip_last=np.mean(predicted[phrase] == true[phrase]),
                f1_macro_skip_last=f1(true[phrase], predicted[phrase]),
                accuracy_no_end=np.mean(no_end[phrase] == true[phrase]),
                f1_macro_no_end=f1(true[phrase], no_end[phrase]),
                end_recall=np.mean(predicted[~phrase] == END))


def evaluate(model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed):
    """Metrics on the test songs: the reservoirs drawn with np.random.seed(seed), HAG learning on phrases of the train
    songs, the readout trained on all the train songs."""
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    states_of = point_states(reservoirs, songs, train_songs + test_songs)
    readout = fit_readout(states_of, songs, train_songs, classes, ridge)
    true = np.concatenate([songs[w]["next"] for w in test_songs])
    return metrics(readout.run(np.vstack([states_of[w] for w in test_songs])), true, classes)
