"""
Study of the reservoirs that read one syllable per step and predict the next syllable
(canopy.studies.next_syllable.task), on the train songs only (the test songs are used once, by
canopy.studies.next_syllable.evaluation).

Objective: out-of-fold accuracy of the next phrase at its prediction points (START and the last syllable of each
phrase), in the 3-fold cross-validation by song on the train songs of the other studies of the next phrase (same folds
and seeds): the raw accuracy, or with CANOPY_SYLLABLE_SCORE=change_known the accuracy knowing that the phrase changes
(canopy.studies.next_syllable.task.changed). In each fold, the reservoirs are built (HAG learning on the inputs of the
fold's training songs) and the ridge readout (task.readout_rows: the next syllable on all the steps, or with
CANOPY_SYLLABLE_READOUT=phrases the next phrase at its prediction points) is solved for every coefficient of RIDGES from
one singular value decomposition of the centered states of the fold's training songs; the coefficient of the best score
is stored in the trial (user attribute "ridge"), with the three out-of-fold accuracies with this coefficient
("accuracy", "accuracy_change_known" and, for the readout of the syllables, "syllable_accuracy"). Three variants,
each with its own studies: the readout of the syllables scored raw (the default) or knowing the changes, and the readout
of the phrases. Search space:
canopy.studies.common.reservoir_params, with the leak rate of each reservoir with CANOPY_LEAKY_RESERVOIRS=1 and an input
scaling up to INPUT_SCALING_MAX (one-hot inputs).

Studies: "next_syllable_calls_[phrases_][change_][leak_]<model>_<size>" (CANOPY_WITH_CALLS=1,
CANOPY_SYLLABLE_READOUT=phrases, CANOPY_SYLLABLE_SCORE=change_known, CANOPY_LEAKY_RESERVOIRS=1) of the pure models, all in
outputs/next_syllable/hpo/marron1_db.sqlite3.

Usage: CANOPY_WITH_CALLS=1 [CANOPY_SYLLABLE_READOUT=phrases | CANOPY_SYLLABLE_SCORE=change_known] \
  python -m canopy.studies.next_syllable.study <model> <size>   (CANOPY_N_TRIALS trials, CANOPY_WORKERS processes
  optimizing the same study)
"""
import sys

import numpy as np
from sklearn.model_selection import KFold

from canopy.common import RIDGES
from canopy.models.reservoirs import build
from canopy.settings import CV_SEED, FOLD_SEED, LEAK, N_FOLDS, SYLLABLE_READOUT, SYLLABLE_SCORE
from canopy.studies.common import reservoir_params, run_study
from canopy.studies.next_syllable.task import (changed, load_task_dataset, pretrain_songs, readout_rows, scores_of,
                                               step_states)

INPUT_SCALING_MAX = 2.0


def ridge_outputs(X, Y, X_val, ridges):
    """Yields the outputs on X_val of the ridge readout of Y on X (with a bias, not penalized) for each coefficient,
    from one singular value decomposition of the centered X (as canopy.studies.next_phrase_labels.study)."""
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    U, s, Vt = np.linalg.svd(X - x_mean, full_matrices=False)
    projected, validation = U.T @ (Y - y_mean), (X_val - x_mean) @ Vt.T
    for ridge in ridges:
        yield validation @ ((s / (s ** 2 + ridge))[:, None] * projected) + y_mean


def choose_ridge(model, params, size, songs, train_songs, classes):
    """(ridge coefficient of the best out-of-fold score (SYLLABLE_SCORE), {"accuracy", "accuracy_change_known",
    "syllable_accuracy" (NaN for the readout of the phrases)} out of fold with this coefficient), in a 3-fold
    cross-validation by song on the train songs."""
    counts = {key: np.zeros(len(RIDGES)) for key in ("accuracy", "accuracy_change_known", "syllable_accuracy")}
    total = total_steps = 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_songs(songs, fit))
        states_of = step_states(reservoirs, songs, fit + val)
        true = np.concatenate([songs[w]["next"] for w in val])
        true_steps = np.concatenate([songs[w]["next_syllable"] for w in val])
        current = np.concatenate([songs[w]["current"] for w in val])
        outputs = ridge_outputs(*readout_rows(states_of, songs, fit, classes),
                                readout_rows(states_of, songs, val, classes)[0], RIDGES)
        for r, scores in enumerate(outputs):
            at_points, at_steps = scores_of(scores, songs, val)
            counts["accuracy"][r] += (classes[at_points.argmax(axis=1)] == true).sum()
            counts["accuracy_change_known"][r] += (changed(at_points, current, classes) == true).sum()
            if at_steps is not None:
                counts["syllable_accuracy"][r] += (classes[at_steps.argmax(axis=1)] == true_steps).sum()
        total, total_steps = total + len(true), total_steps + len(true_steps)
    best = int(np.argmax(counts["accuracy_change_known" if SYLLABLE_SCORE == "change_known" else "accuracy"]))
    scores = {key: count[best] / (total_steps if key == "syllable_accuracy" else total) for key, count in counts.items()}
    if SYLLABLE_READOUT == "phrases":
        scores["syllable_accuracy"] = float("nan")
    return float(RIDGES[best]), scores


def make_objective(model, size):
    songs, train_songs, _, classes = load_task_dataset()
    max_length = max(len(songs[w]["syllables_input"]) for w in train_songs)  # (steps of the longest train song)

    def objective(trial):
        params = reservoir_params(trial, model, max_length, leak=LEAK, input_scaling_max=INPUT_SCALING_MAX)
        ridge, scores = choose_ridge(model, params, size, songs, train_songs, classes)
        trial.set_user_attr("ridge", ridge)
        for key, value in scores.items():
            trial.set_user_attr(key, float(value))
        return scores["accuracy_change_known" if SYLLABLE_SCORE == "change_known" else "accuracy"]
    return objective


if __name__ == "__main__":
    run_study("next_syllable", sys.argv[1], int(sys.argv[2]), make_objective)
