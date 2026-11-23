"""
Study of the reservoirs of the syllable classification (canopy.studies.syllables_classification.task), on the train songs only (the test
songs are used once, by canopy.studies.syllables_classification.evaluation).

Objective: out-of-fold frame accuracy, in a 3-fold cross-validation by song on the train songs (choose_ridge): in each
fold, the reservoirs are built (HAG learning on phrases of the fold's training songs) and the ridge readout of
canopy.studies.syllables_classification.task is trained for every coefficient of RIDGES on the fold's training songs; the ridge coefficient
is chosen on the out-of-fold frames, stored in the trial (user attribute "ridge") for the final evaluation; it is not a
hyperparameter of the search. Search space:
canopy.studies.common.reservoir_params, with the gain of each group of input features with HAG_CANARY_INPUT_GAINS=1 and
the leak rate of each reservoir with HAG_CANARY_LEAK=1.

Studies: "syllables_[gains_][leak_]<model>_<size>" of the pure models (canopy.models.reservoirs.PURE_MODELS), one
SQLite database each in outputs/syllables_classification/hpo (canopy.settings.studies_dir).

Usage: python -m canopy.studies.syllables_classification.study <model> <size>   (HAG_CANARY_N_TRIALS trials, HAG_CANARY_WORKERS processes
  optimizing the same study)
"""
import sys

import numpy as np
from sklearn.model_selection import KFold

from canopy.common import RIDGES, pretrain_phrases
from canopy.studies.common import longest_phrase, reservoir_params, run_study
from canopy.models.reservoirs import build, normal_equations, ridge_readout
from canopy.settings import CV_SEED, FOLD_SEED, INPUT_GAINS, LEAK, N_FOLDS
from canopy.studies.syllables_classification.task import correct_frames, load_task_dataset


# ---- Ridge coefficient of the readout of the syllables, chosen by the 3-fold cross-validation by song on the train
# songs (out-of-fold frame accuracy; the reservoirs of fold k drawn with the seed CV_SEED + k): the objective of the
# studies, and the coefficient of the conditions in canopy.studies.syllables_classification.evaluation and
# canopy.studies.syllables_classification.certainty.
def choose_ridge(model, params, size, songs, train_songs, frame, classes):
    """Ridge coefficient of the best out-of-fold frame accuracy, in a 3-fold cross-validation by song."""
    correct = np.zeros(len(RIDGES))
    total = 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit = [train_songs[i] for i in fit_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        statistics = normal_equations(reservoirs, songs, fit, target="Y")
        fold_correct, _, fold_total, _ = correct_frames(reservoirs, songs, [train_songs[i] for i in val_index],
                                                        [ridge_readout(statistics, ridge) for ridge in RIDGES], classes)
        correct += fold_correct
        total += fold_total
    best = int(np.argmax(correct))
    return float(RIDGES[best]), correct[best] / total


# ---- study
def make_objective(model, size):
    songs, train_songs, _, frame, classes = load_task_dataset()
    max_length = longest_phrase(songs, train_songs, frame)

    def objective(trial):
        params = reservoir_params(trial, model, max_length, leak=LEAK, input_gains=INPUT_GAINS)
        ridge, accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
        trial.set_user_attr("ridge", ridge)
        return accuracy
    return objective


if __name__ == "__main__":
    run_study("syllables_classification", sys.argv[1], int(sys.argv[2]), make_objective)
