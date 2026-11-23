"""
Study of the reservoirs that read the labels of the phrases (canopy.studies.next_phrase_labels.task), on the train
songs only (the test songs are used once, by canopy.studies.next_phrase_labels.evaluation): the reservoirs of this
control are optimized for their own input, as those of the task on the audio (canopy.studies.next_phrase.study).

Objective: out-of-fold accuracy of the next phrase, in the 3-fold cross-validation by song on the train songs of the
studies on the audio (same folds and seeds; choose_ridge): in each fold, the reservoirs are built (HAG learning on the
inputs of the fold's training songs) and the ridge readout is solved for every coefficient of RIDGES from one singular
value decomposition of the centered states of the fold's training songs (the solution of the "svd" solver of the
readout of the final evaluation); the ridge coefficient is chosen on the out-of-fold points, stored in the trial
(user attribute "ridge"); it is not a hyperparameter of the search. Search space:
canopy.studies.common.reservoir_params, with the leak rate of each reservoir with CANOPY_LEAKY_RESERVOIRS=1 and an
input scaling up to INPUT_SCALING_MAX (the inputs are one-hot vectors, not MFCC).

Studies: "next_phrase_labels_[calls_][durations_][leak_]<model>_<size>" (canopy.settings: CANOPY_WITH_CALLS=1,
CANOPY_LABELS_WITH_DURATIONS=1, CANOPY_LEAKY_RESERVOIRS=1) of the pure models (canopy.models.reservoirs.PURE_MODELS),
all in the SQLite database of the dataset, outputs/next_phrase_labels/hpo/marron1_db.sqlite3.

Usage: python -m canopy.studies.next_phrase_labels.study <model> <size>   (CANOPY_N_TRIALS trials, CANOPY_WORKERS
  processes optimizing the same study)
"""
import sys

import numpy as np
from sklearn.model_selection import KFold

from canopy.common import RIDGES
from canopy.models.reservoirs import build
from canopy.settings import CV_SEED, FOLD_SEED, LEAK, N_FOLDS
from canopy.studies.common import reservoir_params, run_study
from canopy.studies.next_phrase.task import one_hot
from canopy.studies.next_phrase_labels.task import load_task_dataset, point_states, pretrain_songs

INPUT_SCALING_MAX = 2.0


# ---- Ridge coefficient of the readout, chosen by the 3-fold cross-validation by song on the train songs (out-of-fold
# accuracy; the reservoirs of fold k drawn with the seed CV_SEED + k): the objective of the studies, and the
# coefficient of the conditions without study in canopy.studies.next_phrase_labels.evaluation.
def ridge_outputs(X, Y, X_val, ridges):
    """Yields the outputs on X_val of the ridge readout of Y on X (with a bias, not penalized) for each coefficient,
    from one singular value decomposition of the centered X."""
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    U, s, Vt = np.linalg.svd(X - x_mean, full_matrices=False)
    projected, validation = U.T @ (Y - y_mean), (X_val - x_mean) @ Vt.T
    for ridge in ridges:
        yield validation @ ((s / (s ** 2 + ridge))[:, None] * projected) + y_mean


def choose_ridge(model, params, size, songs, train_songs, classes):
    """(ridge coefficient of the best out-of-fold accuracy, this accuracy), in a 3-fold cross-validation by song on the
    train songs."""
    correct = np.zeros(len(RIDGES))
    total = 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_songs(songs, fit))
        states_of = point_states(reservoirs, songs, fit + val)
        true = np.concatenate([songs[w]["next"] for w in val])
        outputs = ridge_outputs(np.vstack([states_of[w] for w in fit]), one_hot(songs, fit, classes),
                                np.vstack([states_of[w] for w in val]), RIDGES)
        for r, scores in enumerate(outputs):
            correct[r] += (classes[scores.argmax(axis=1)] == true).sum()
        total += len(true)
    return float(RIDGES[int(np.argmax(correct))]), correct.max() / total


# ---- study
def make_objective(model, size):
    songs, train_songs, _, classes = load_task_dataset()
    max_length = max(len(songs[w]["labels_input"]) for w in train_songs)  # (steps of the longest train song)

    def objective(trial):
        params = reservoir_params(trial, model, max_length, leak=LEAK, input_scaling_max=INPUT_SCALING_MAX)
        ridge, accuracy = choose_ridge(model, params, size, songs, train_songs, classes)
        trial.set_user_attr("ridge", ridge)
        return accuracy
    return objective


if __name__ == "__main__":
    run_study("next_phrase_labels", sys.argv[1], int(sys.argv[2]), make_objective)
