"""
Study of the reservoirs of the phrase boundaries (canopy.studies.phrase_boundaries.task), on the train songs only (the test songs are
used once, by canopy.studies.phrase_boundaries.evaluation).

Objective: F1 of the boundaries, out of fold, in a 3-fold cross-validation by song on the train songs
(cross_validate): in each fold, the reservoirs are built (HAG learning on phrases of the fold's training songs) and
the ridge readout of canopy.studies.phrase_boundaries.task is trained for every coefficient of RIDGES on the fold's training songs;
the ridge coefficient and the peak threshold are then chosen on the out-of-fold outputs of all the train songs, stored
in the trial (user attributes "ridge" and "threshold") for the final evaluation; they are not hyperparameters of the
search. Search space: canopy.studies.common.reservoir_params, without leak.

Studies: "<model>_<size>" of the pure models (canopy.models.reservoirs.PURE_MODELS), one SQLite database each in
outputs/phrase_boundaries/hpo (canopy.settings.studies_dir).

Usage: python -m canopy.studies.phrase_boundaries.study <model> <size>   (HAG_CANARY_N_TRIALS trials, HAG_CANARY_WORKERS
  processes optimizing the same study)
"""
import sys

import numpy as np
from sklearn.model_selection import KFold

from canopy.common import RIDGES, pretrain_phrases
from canopy.studies.common import longest_phrase, reservoir_params, run_study
from canopy.models.reservoirs import build, normal_equations, readout_outputs, ridge_readout
from canopy.settings import CV_SEED, FOLD_SEED, N_FOLDS
from canopy.studies.phrase_boundaries.task import THRESHOLDS, load_task_dataset, scores


# ---- Ridge coefficient and peak threshold of the readout of the boundaries, chosen by the 3-fold cross-validation by
# song on the train songs (out-of-fold F1; the reservoirs of fold k drawn with the seed CV_SEED + k): the objective of
# the studies, and those of the conditions without study in canopy.studies.phrase_boundaries.evaluation.
def cross_validate(model, params, size, songs, train_songs, frame):
    """(out-of-fold F1, ridge, threshold) of the model on the train songs, see the docstring of the module."""
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)
    outputs = {}  # (ridge index, wave) -> readout output
    for k, (fit_index, val_index) in enumerate(folds):
        fit = [train_songs[i] for i in fit_index]
        val = [train_songs[i] for i in val_index]
        np.random.seed(CV_SEED + k)
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        statistics = normal_equations(reservoirs, songs, fit)
        Wouts = [ridge_readout(statistics, ridge) for ridge in RIDGES]
        for w, output in readout_outputs(reservoirs, songs, val, Wouts).items():
            for r in range(len(RIDGES)):
                outputs[r, w] = output[:, r]

    best = (-1.0, None, None)
    for r, ridge in enumerate(RIDGES):
        song_outputs = [outputs[r, w] for w in train_songs]
        for threshold in THRESHOLDS:
            f1 = scores(song_outputs, songs, train_songs, threshold, frame)["f1"]
            if f1 > best[0]:
                best = (f1, float(ridge), float(threshold))
    return best


# ---- study
def make_objective(model, size):
    songs, train_songs, _, frame = load_task_dataset()
    max_length = longest_phrase(songs, train_songs, frame)

    def objective(trial):
        params = reservoir_params(trial, model, max_length)
        f1, ridge, threshold = cross_validate(model, params, size, songs, train_songs, frame)
        trial.set_user_attr("ridge", ridge)
        trial.set_user_attr("threshold", threshold)
        return f1
    return objective


if __name__ == "__main__":
    run_study("phrase_boundaries", sys.argv[1], int(sys.argv[2]), make_objective)
