"""
Prediction of the next phrase of canary songs from their audio, in the configuration of Maxime Canonne's next-phrase
prediction (ESN reading the phrase labels of the song, with START and END), but with the MFCC of the song as input:
the reservoir runs over the whole song (no reset between the phrases) and, at the end of each phrase, the readout predicts
the label of the next one (END after the last phrase); before the first phrase (Maxime's START), it predicts the first
one. Phrases as in canary.datasets.songs.load_songs: SIL, TRASH and call ignored, repeated labels merged; with
HAG_CANARY_CALLS=1, the calls are phrases too (SIL and TRASH ignored). With HAG_CANARY_RESET=1, the reservoirs hear
the current phrase only, from a zero state (as the Canary task of hag), instead of the whole song: the
state of a point is the last one of a run on the frames of the phrase that ends there (the zero state before the first
phrase); the HAG learning is the same (on phrases). Input: the 13 MFCC, without their deltas (their effect, with and
without leak: outputs/canary/ablation_deltas_leak).

No input after the prediction point: the MFCC are computed without the floor of the dB at 80 dB below the maximum of
the whole song (librosa's default, which depends on the future of each frame: canary.common.load_dataset(top_db=None)),
and the MFCC frames are centered (half a window, 11.6 ms, of audio after their time), so the state used to predict
what follows the time t is the one of the last frame whose window ends before t (DELAY earlier).

Readout: ridge regression on the one-hot next label at the prediction points (reservoirpy.nodes.Ridge, solver
HAG_CANARY_RIDGE_SOLVER, default "svd": the singular value decomposition of the centered states, stable with the
smallest coefficients), its coefficient chosen by the 3-fold cross-validation by song on the train songs (out-of-fold
accuracy), then trained on all the train songs and scored on the test songs for each seed. With HAG_CANARY_READOUT_ORDER=n (n >= 2), the generalized
readout of Ohkubo & Inubushi (2024) instead (canary.generalized_readout.GeneralizedRidge: ridge regression on all the
monomials of the state up to degree n, the quadratic form for n = 2), its coefficient always chosen by the same
cross-validation (the coefficients of the studies are those of the linear readout).
Training rows of the readout (HAG_CANARY_POINTS), from the states of the window of each point: those of the frames of
the phrase ending there, up to the point (from the first frame of the song for the point before the first phrase; the
state at the point alone when there is no such frame): "last" (default) its last state only, "tail" those of its
last HAG_CANARY_TAIL frames (default 10, 116 ms), "tail_mean" their mean; all the rows of a point have its next label.
Decision (HAG_CANARY_DECISION, the scores of the validation and test points): the outputs on the last state of the
window ("last") or on the mean of the states of its last HAG_CANARY_TAIL frames ("tail_mean", the mean of the outputs
on these states with the linear readout); by default, on the training row of the point, or on the mean of its rows
("tail": "tail_mean"). (The means of all the states of the window, "mean", and all its states as rows, "all", were
removed: always worse, outputs/canary/next_phrase_calls{,_leak}_hpo_readouts.csv.)
Comma-separated lists of these three variables: all their combinations (readouts (points, decision, tail), the tail
None when neither uses it), on the same reservoirs, their coefficients chosen by the same cross-validation (that of
the study for "last" -> "last"), and their paired comparisons with "last" -> "last".
With the studies of the readout (HAG_CANARY_TAIL_HPO=1, canary.next_phrase_prediction.hpo), the readouts of each
condition are instead: "last" -> "last", that of its study (column hpo; for the conditions without study, mfcc and
hybrids, the best of TAIL_READOUTS in the cross-validation) and, for each kind of tail readout (training rows "tail"
or "tail_mean", decision "last" or "tail_mean"), the length of tail of the best cross-validation accuracy (column
best_tail), the coefficients of all of them chosen by the same cross-validation. With HAG_CANARY_TAIL_HPO=2 (TAIL_SEPARATE),
the tails of the training rows (tail) and of the decision (decision_tail) are two parameters of the studies, and
best_tail is the best pair of tails of each kind of readout; HAG_CANARY_DECISION_TAIL does the same for lists of
readouts (default: the tail of the training rows).
Metrics as Maxime's: accuracy and macro F1 of all the points,
without the last one of each song ("skip_last": its target is END), and of the phrases only ("no_end": points whose
target is a phrase, argmax over the phrases).
References: n-grams of the true labels of the previous phrases (canary.next_phrase_prediction.prediction baselines): fixed order n
(the n previous labels, START before the first phrase; unseen context: the most frequent next label) and adaptive (the
longest context seen in the train songs, up to MAX_ORDER).

Results: outputs/canary/<HAG_CANARY_TASK=next_phrase prefix>hpo.csv (e.g. next_phrase_leak_hpo.csv) and
<...>_comparisons.csv (<...>hpo_order<n>.csv with the generalized readout, <...>hpo_points_<points>[<tail>].csv
with other training rows),
references in next_phrase_ngrams.csv (next_phrase_calls_ngrams.csv with the calls).
Usage (HAG_CANARY_TASK=next_phrase, with the variables of the studies, e.g. HAG_CANARY_LEAK=1):
  python -m canary.next_phrase_prediction.prediction [sizes] [nb_seeds]   all the conditions (default 500,1000, 10 seeds)
  python -m canary.next_phrase_prediction.prediction run <condition> <size> [nb_seeds]   one condition (slurm/submit_next_phrase.sh),
                                                                          results in outputs/canary/<name>_parts
  python -m canary.next_phrase_prediction.prediction merge                               results of the parts and the comparisons
  python -m canary.next_phrase_prediction.prediction list <sizes>                        "<condition> <size>" of all the conditions
  python -m canary.next_phrase_prediction.prediction baselines                           the n-grams
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from canary.common import OUTPUTS, RIDGES, load_dataset, pretrain_phrases
from canary.evaluate import compare
from canary.generalized_readout import GeneralizedRidge
from reservoirpy.nodes import Ridge
from canary.hpo import CV_SEED, FOLD_SEED, N_FOLDS
from canary.models import CHUNK, build, state_chunks, states
from canary.next_phrase_prediction.evaluate import conditions
from canary.next_phrase_prediction.hpo import CALLS, MAX_TAIL, RESET, STUDY_PREFIX, TAIL_HPO, TAIL_SEPARATE, TASK
from canary.datasets.songs import SONGS_SAMPLING_RATE, annotations_of_songs, canary_mfcc_config, song_phrases

SIZES = [500, 1000]
START, END = "START", "END"
N_STATIC = 13  # MFCC, before their deltas and delta-deltas in the cached features
MAX_ORDER = 10
READOUT_ORDER = int(os.environ.get("HAG_CANARY_READOUT_ORDER", "1"))  # 1: linear ridge readout, n: generalized readout
RIDGE_SOLVER = os.environ.get("HAG_CANARY_RIDGE_SOLVER", "svd")  # (of reservoirpy.nodes.Ridge, linear readout)
TAIL_MODES = ("tail", "tail_mean")  # (those of the last HAG_CANARY_TAIL frames of the windows)
BASE_READOUT = ("last", "last", None, None)  # (training rows, decision, tail, decision tail): the last state only


def readout_of(points, decision, tail, decision_tail=None):
    """The readout (points, decision, tail, decision tail), the decision by default that of the training rows, the tail
    None when neither uses it, the tail of the decision by default that of the training rows (None for the decision on
    the last state)."""
    decision = decision or {"tail": "tail_mean"}.get(points, points)
    if points not in ("last",) + TAIL_MODES or decision not in ("last", "tail_mean"):
        raise ValueError(f"Unknown readout {points} -> {decision}: HAG_CANARY_POINTS last, tail or tail_mean, "
                         f"HAG_CANARY_DECISION last or tail_mean")
    if READOUT_ORDER > 1 and points == "tail":
        raise ValueError("The generalized readout needs a single row per point (HAG_CANARY_POINTS last or tail_mean)")
    return (points, decision, tail if points in TAIL_MODES or decision == "tail_mean" else None,
            (decision_tail or tail) if decision == "tail_mean" else None)


def readout_of_params(params):
    """Readout of the parameters "readout:*" of a study of the readout (canary.next_phrase_prediction.hpo,
    TAIL_HPO), None without them: "last" -> "last" for a tail of 1 frame. With "readout:decision_tail" (TAIL_SEPARATE),
    the tail of the training rows ("readout:tail", 1: the last state) and that of the decision are separate."""
    tail = params.get("readout:tail")
    if tail is None:
        return None
    decision_tail = params.get("readout:decision_tail")
    if decision_tail is not None:
        points = "last" if tail == 1 else "tail_mean" if params["readout:train_mean"] else "tail"
        return readout_of(points, "last" if decision_tail == 1 else "tail_mean", int(tail) if tail > 1 else None,
                          int(decision_tail) if decision_tail > 1 else None)
    if tail == 1:
        return BASE_READOUT
    return readout_of("tail_mean" if params["readout:train_mean"] else "tail",
                      "tail_mean" if params["readout:decision_mean"] else "last", int(tail))


# all the readouts of the studies of the readout (TAIL_HPO), "last" -> "last" first
# (TAIL_SEPARATE: all the pairs of a training tail, "last" or 2 to MAX_TAIL frames, and a decision tail)
TAIL_READOUTS = list(dict.fromkeys(
    [BASE_READOUT] + ([readout_of(points, decision, tail, decision_tail)
                       for points, tail in [("last", None)] + [(m, t) for m in TAIL_MODES for t in range(2, MAX_TAIL + 1)]
                       for decision, decision_tail in [("last", None)] + [("tail_mean", t) for t in range(2, MAX_TAIL + 1)]]
                      if TAIL_SEPARATE else [readout_of(points, decision, tail) for points in TAIL_MODES
                                             for decision in ("last", "tail_mean") for tail in range(2, MAX_TAIL + 1)])))
# (HAG_CANARY_DECISION_TAIL: tails of the decision, by default that of the training rows)
READOUTS = TAIL_READOUTS if TAIL_HPO and READOUT_ORDER == 1 else list(dict.fromkeys(
    readout_of(points, decision, int(tail), int(decision_tail) if decision_tail else None)
    for points in os.environ.get("HAG_CANARY_POINTS", "last").split(",")
    for decision in os.environ.get("HAG_CANARY_DECISION", "").split(",")
    for tail in os.environ.get("HAG_CANARY_TAIL", "10").split(",")
    for decision_tail in os.environ.get("HAG_CANARY_DECISION_TAIL", "").split(",")))
# audio after the time of a frame in its features: half a window
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


def point_windows(reservoirs, songs, waves):
    """
    Yields (wave, windows): for each prediction point of the song, the states (frames, units) of the reservoirs
    (inputs without reservoir) on the frames of the phrase ending there, up to the point (from the first frame of the
    song for the point before the first phrase); the state at the point alone when there is no such frame (the zero
    state for the initial point, the state at the point for a phrase shorter than the delay of the points). With the
    reservoirs reset at each phrase (RESET): the states of a run from a zero state on these frames (the zero state
    without frame, as for the point before the first phrase).
    """
    if not RESET:
        for part, part_states in state_chunks(reservoirs, songs, waves):
            for x, w in zip(part_states, part):
                x = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float64), np.asarray(x, dtype=np.float64)])
                # (row 0: the initial zero state, point -1)
                yield w, [x[max(start, 0) + 1:point + 2] if max(start, 0) <= point else x[point + 1:point + 2]
                          for start, point in zip(songs[w]["starts"], songs[w]["points"])]
        return
    for i in range(0, len(waves), CHUNK):
        part = waves[i:i + CHUNK]
        runs = states(reservoirs, [songs[w]["mfcc"][start:point + 1] for w in part
                                   for start, point in zip(songs[w]["starts"], songs[w]["points"])
                                   if 0 <= start <= point])
        zero = np.zeros((1, runs[0].shape[1]))
        runs = iter(runs)
        for w in part:
            yield w, [np.asarray(next(runs), dtype=np.float64) if 0 <= start <= point else zero
                      for start, point in zip(songs[w]["starts"], songs[w]["points"])]


def decision_of(readout):
    """(decision, tail) of a readout: the states of its decision."""
    points, decision, tail, decision_tail = readout
    return decision, decision_tail if decision == "tail_mean" else None


def training_of(readout):
    """(training rows, tail) of a readout: its training data."""
    points, decision, tail, decision_tail = readout
    return points, tail if points in TAIL_MODES else None


def point_data(reservoirs, songs, waves, readouts, fit=(), classes=None):
    """
    States of the prediction points of the songs for the decisions of the readouts, {decision_of(readout): {wave:
    (points, units)}}: in the window of each point (point_windows), its last state or the mean of its last tail ones.
    With the songs fit (and the classes), also the training data of the readouts on them,
    {training_of(readout): (rows, labels)}: the rows of the windows with their one-hot next labels.
    """
    fit = set(fit)
    states = {decision: {} for decision in {decision_of(readout) for readout in readouts}}
    accumulated = {training: ([], []) for training in {training_of(readout) for readout in readouts}} if fit else {}
    for w, windows in point_windows(reservoirs, songs, waves):
        def selection(mode, tail):  # (the states of each window used)
            return [x[-1:] for x in windows] if mode == "last" else [x[-tail:] for x in windows]
        for (decision, tail), result in states.items():
            result[w] = np.array([x.mean(axis=0) for x in selection(decision, tail)])
        if w not in fit:
            continue
        next_labels = (songs[w]["next"][:, None] == classes[None, :]).astype(np.float64)
        for (points, tail), (inputs, labels) in accumulated.items():
            rows = selection(points, tail)
            if points == "tail":
                inputs.append(np.vstack(rows))
                labels.append(np.repeat(next_labels, [len(x) for x in rows], axis=0))
            else:
                inputs.append(np.array([x.mean(axis=0) for x in rows]))
                labels.append(next_labels)
    if not fit:
        return states
    return states, {training: (np.vstack(inputs), np.vstack(labels))
                    for training, (inputs, labels) in accumulated.items()}


def point_states(reservoirs, songs, waves, fit=(), classes=None):
    """{wave: states of the prediction points of the songs, (points, units)} for the decision of the first readout
    (READOUTS) and, with the songs fit (and the classes), its training data on them (point_data)."""
    result = point_data(reservoirs, songs, waves, READOUTS[:1], fit, classes)
    if not fit:
        return result[decision_of(READOUTS[0])]
    return result[0][decision_of(READOUTS[0])], result[1][training_of(READOUTS[0])]


def fitted_readouts(data, ridges):
    """Yields (ridge, the readout trained on the data (rows, labels) with this coefficient) for each coefficient of
    ridges: reservoirpy's Ridge (solver RIDGE_SOLVER), or the generalized readout (a single fit, solved again for each
    coefficient)."""
    if READOUT_ORDER == 1:
        for ridge in ridges:
            yield ridge, Ridge(ridge=ridge, solver=RIDGE_SOLVER).fit(*data)
        return
    readout = GeneralizedRidge(order=READOUT_ORDER).fit(*data)
    for ridge in ridges:
        yield ridge, readout.set_ridge(ridge)


def fit_readout(data, ridge):
    """The readout trained on the data (rows, labels) with the coefficient ridge (fitted_readouts)."""
    return next(fitted_readouts(data, [ridge]))[1]


def choose_ridges(model, params, size, songs, train_songs, frame, classes, readouts):
    """{readout: (ridge coefficient of the best out-of-fold accuracy, this accuracy)}, in a 3-fold cross-validation by
    song (the folds of canary.next_phrase_prediction.hpo), the readouts on the same reservoirs."""
    correct = {readout: np.zeros(len(RIDGES)) for readout in readouts}
    total = 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        states, data = point_data(reservoirs, songs, fit + val, readouts, fit, classes)
        true = np.concatenate([songs[w]["next"] for w in val])
        X_val = {decision: np.vstack([result[w] for w in val]) for decision, result in states.items()}
        for training, training_data in data.items():  # (each readout trained once, for all its decisions)
            for r, (_, fitted) in enumerate(fitted_readouts(training_data, RIDGES)):
                for readout in readouts:
                    if training_of(readout) == training:
                        scores = fitted.run(X_val[decision_of(readout)])
                        correct[readout][r] += (classes[scores.argmax(axis=1)] == true).sum()
        total += len(true)
    return {readout: (float(RIDGES[int(np.argmax(c))]), c.max() / total) for readout, c in correct.items()}


def choose_ridge(model, params, size, songs, train_songs, frame, classes):
    """(ridge coefficient, out-of-fold accuracy) of the first readout (choose_ridges)."""
    return choose_ridges(model, params, size, songs, train_songs, frame, classes, READOUTS[:1])[READOUTS[0]]


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
                accuracy_no_end=np.mean(no_end[phrase] == true[phrase]), f1_macro_no_end=f1(true[phrase], no_end[phrase]),
                end_recall=np.mean(predicted[~phrase] == END))


def evaluate(model, params, size, ridges, songs, train_songs, test_songs, frame, classes, seed):
    """{readout: metrics on the test songs}, the readouts (with their coefficients ridges) on the same reservoirs, each
    readout trained once for all its decisions."""
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    states, data = point_data(reservoirs, songs, train_songs + test_songs, list(ridges), train_songs, classes)
    true = np.concatenate([songs[w]["next"] for w in test_songs])
    fitted = {}
    for readout, ridge in ridges.items():
        if (training_of(readout), ridge) not in fitted:
            fitted[training_of(readout), ridge] = fit_readout(data[training_of(readout)], ridge)
    return {readout: metrics(fitted[training_of(readout), ridge].run(
        np.vstack([states[decision_of(readout)][w] for w in test_songs])), true, classes)
            for readout, ridge in ridges.items()}


# ------------------------------------------------- n-grams --------------------------------------------------------
def baselines():
    """For each order (None: adaptive), metrics of the scores (frequencies of the next labels) of the n-gram at the
    test points, and the share of the points whose context was not seen in the train songs (marginal frequencies
    then)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    true = np.concatenate([songs[w]["next"] for w in test_songs])
    index = {label: i for i, label in enumerate(classes)}

    def contexts(song, n):
        history = [START] * n + [label for _, _, label in song["next_phrases"]]
        return [tuple(history[j:j + n]) for j in range(len(song["next"]))]

    rows = []
    for order in [*range(0, MAX_ORDER + 1), None]:
        orders = list(range(MAX_ORDER, -1, -1)) if order is None else sorted({order, 0}, reverse=True)
        counts = {n: {} for n in orders}
        for w in train_songs:
            for n in orders:
                for context, label in zip(contexts(songs[w], n), songs[w]["next"]):
                    counts[n].setdefault(context, Counter())[label] += 1
        scores, unseen = [], 0
        for w in test_songs:
            by_order = {n: contexts(songs[w], n) for n in orders}
            for j in range(len(songs[w]["next"])):
                n = next(n for n in orders if by_order[n][j] in counts[n])
                unseen += n == 0 and (order is None or order > 0)
                row = np.zeros(len(classes))
                for label, count in counts[n][by_order[n][j]].items():
                    row[index[label]] = count
                scores.append(row / row.sum())
        rows.append(dict(condition="ngram_adaptive" if order is None else f"ngram_{order}", order=order,
                         unseen_context=unseen / len(scores), **metrics(np.array(scores), true, classes)))
    results = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    file_name = OUTPUTS / f"next_phrase_{'calls_' * CALLS}ngrams.csv"
    results.to_csv(file_name, index=False)
    print("Saved to", file_name)


# ------------------------------------------------ reservoirs ------------------------------------------------------
def readout_name(readout):
    """Suffix of the results of a single readout ("" for "last" -> "last")."""
    points, decision, tail, decision_tail = readout
    default = readout_of(points, None, tail)[1]
    return ((f"_points_{points}{tail if points in TAIL_MODES else ''}" if points != "last" else "")
            + (f"_decision_{decision}{decision_tail if decision == 'tail_mean' and points not in TAIL_MODES else ''}"
               if decision != default else "")
            + (f"_decision_tail{decision_tail}" if decision == default == "tail_mean" and decision_tail != tail else ""))


NAME = (f"{STUDY_PREFIX}hpo" + (f"_order{READOUT_ORDER}" if READOUT_ORDER > 1 else "")
        + (readout_name(READOUTS[0]) if len(READOUTS) == 1 else "_readouts"))
PARTS = OUTPUTS / f"{NAME}_parts"


def all_conditions(sizes):
    """[(condition, model, params, ridge, size)], mfcc first (size 0); ridge None when chosen by cross-validation."""
    todo = [("mfcc", "mfcc", {}, None, 0)]
    for size in sizes:
        todo += [(condition, model, params, ridge, size) for condition, model, params, ridge, _ in conditions(size)]
    return todo


def run(todo, nb_seeds):
    """Results of the readouts (READOUTS, columns points, decision and tail when they are not "last" -> "last" alone)
    for each condition and seed."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, {len(classes)} classes, "
          f"{sum(len(songs[w]['next']) for w in test_songs)} test points", flush=True)
    rows = []
    for condition, model, params, ridge, size in todo:
        # (the coefficient of the study for "last" -> "last": that of the linear readout)
        study = ridge is not None and READOUT_ORDER == 1 and BASE_READOUT in READOUTS and not TAIL_HPO
        chosen = {} if study and READOUTS == [BASE_READOUT] else \
            choose_ridges(model, params, size, songs, train_songs, frame, classes, READOUTS)
        chosen.update({BASE_READOUT: (ridge, np.nan)} if study else {})
        readouts, columns_of = READOUTS, {}
        if TAIL_HPO:  # "last" -> "last", the readout of the study and the best tail of each kind of readout
            hpo = readout_of_params(params) or max(READOUTS, key=lambda readout: chosen[readout][1])
            best_tail = {max((r for r in READOUTS[1:] if r[:2] == kind), key=lambda r: chosen[r][1])
                         for kind in dict.fromkeys(r[:2] for r in READOUTS[1:])}
            readouts = list(dict.fromkeys([BASE_READOUT, hpo, *sorted(best_tail)]))
            columns_of = {r: dict(hpo=r == hpo, best_tail=r in best_tail) for r in readouts}
        ridges = {readout: chosen[readout][0] for readout in readouts}
        accuracies = {readout: [] for readout in readouts}
        for seed in range(nb_seeds):
            results = evaluate(model, params, size, ridges, songs, train_songs, test_songs, frame, classes, seed)
            for readout, result in results.items():
                points, decision, tail, decision_tail = readout
                columns = {} if READOUTS == [BASE_READOUT] else dict(points=points, decision=decision, tail=tail,
                                                                     decision_tail=decision_tail,
                                                                     **columns_of.get(readout, {}))
                rows.append(dict(condition=condition, network_size=size, seed=seed, **columns,
                                 ridge=ridges[readout], cv_accuracy=chosen[readout][1], **result))
                accuracies[readout].append(result["accuracy"])
        for readout, accuracy in accuracies.items():
            tails = [t for t in dict.fromkeys(readout[2:]) if t is not None]
            print(f"{condition} ({size}) {readout[0]} -> {readout[1]}{f' ({'/'.join(map(str, tails))})' if tails else ''}:"
                  f" accuracy {100 * np.mean(accuracy):.2f} ± {100 * np.std(accuracy):.2f} % "
                  f"(ridge {ridges[readout]:.0e})", flush=True)
    return pd.DataFrame(rows)


def filled_tails(results):
    """The results with the tails of the training and of the decision filled (-1 for none; the decision tail of the
    results without this column, before TAIL_SEPARATE, is the tail of the readout)."""
    decision_tail = results["decision_tail"] if "decision_tail" in results else results["tail"]
    return results.assign(tail=results["tail"].fillna(-1), decision_tail=decision_tail.fillna(-1))


def compare_readouts(results, metric="accuracy"):
    """Paired comparisons by seed of the metric of each readout with that of "last" -> "last" (readout - last), for
    each condition and size."""
    keys = ["condition", "network_size", "seed"]
    readouts = filled_tails(results)
    base = readouts[(readouts.points == "last") & (readouts.decision == "last")].set_index(keys)[metric]
    rows = []
    for (condition, size, points, decision, tail, decision_tail), group in readouts.groupby(
            ["condition", "network_size", "points", "decision", "tail", "decision_tail"]):
        if points == decision == "last":
            continue
        diff = (group.set_index(keys)[metric] - base).dropna().to_numpy()
        if len(diff) == 0:
            continue
        rows.append(dict(condition=condition, network_size=size, points=points, decision=decision,
                         tail=None if tail < 0 else int(tail), decision_tail=None if decision_tail < 0 else int(decision_tail),
                         nb_seeds=len(diff), mean=group[metric].mean(),
                         mean_diff=diff.mean(), wilcoxon_p=wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0,
                         a_better=int((diff > 0).sum())))
    return pd.DataFrame(rows)


def save(results):
    """The results, the paired comparisons of the conditions (for each readout) and, with several readouts, those of
    the readouts with "last" -> "last"."""
    pd.set_option("display.width", 200)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS / f"{NAME}.csv", index=False)
    if "points" not in results:
        comparisons = compare(results, metric="accuracy")
    elif "hpo" in results:  # (the readouts of the studies, TAIL_HPO; the tail differs between the conditions)
        comparisons = compare(results[results.hpo.astype(bool)], metric="accuracy")
        readout_comparisons = compare_readouts(results[results.hpo.astype(bool) | ((results.points == "last")
                                                                                   & (results.decision == "last"))])
        print(readout_comparisons.round(4).to_string(index=False))
        readout_comparisons.to_csv(OUTPUTS / f"{NAME}_readout_comparisons.csv", index=False)
        print("Saved to", OUTPUTS / f"{NAME}_readout_comparisons.csv")
    else:
        comparisons = pd.concat([compare(group, metric="accuracy").assign(
            points=points, decision=decision, tail=None if tail < 0 else int(tail),
            decision_tail=None if decision_tail < 0 else int(decision_tail))
            for (points, decision, tail, decision_tail), group in filled_tails(results).groupby(
                ["points", "decision", "tail", "decision_tail"])], ignore_index=True)
        readout_comparisons = compare_readouts(results)
        print(readout_comparisons.round(4).to_string(index=False))
        readout_comparisons.to_csv(OUTPUTS / f"{NAME}_readout_comparisons.csv", index=False)
        print("Saved to", OUTPUTS / f"{NAME}_readout_comparisons.csv")
    print(comparisons.round(4).to_string(index=False))
    comparisons.to_csv(OUTPUTS / f"{NAME}_comparisons.csv", index=False)
    print("Saved to", OUTPUTS / f"{NAME}.csv", "and", OUTPUTS / f"{NAME}_comparisons.csv")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "baselines":
        baselines()
    elif TASK != "next_phrase":
        raise RuntimeError("HAG_CANARY_TASK=next_phrase is needed (names of the studies)")
    elif command == "list":
        for condition, _, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size] if size else []) if c[0] == condition and c[4] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        PARTS.mkdir(parents=True, exist_ok=True)
        run(todo, nb_seeds).to_csv(PARTS / f"{condition}_{size}.csv", index=False)
    elif command == "merge":
        parts = sorted(PARTS.glob("*.csv"))
        print(f"{len(parts)} conditions")
        save(pd.concat([pd.read_csv(part) for part in parts], ignore_index=True))
    else:
        sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SIZES
        nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        save(run(all_conditions(sizes), nb_seeds))

