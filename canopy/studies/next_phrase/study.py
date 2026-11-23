"""
Study of the reservoirs of the prediction of the next phrase (canopy.studies.next_phrase.task), on the train songs only (the
test songs are used once, by canopy.studies.next_phrase.evaluation).

Objective: out-of-fold accuracy of the next phrase, in a 3-fold cross-validation by song on the train songs
(choose_ridge): in each fold, the reservoirs are built (HAG learning on phrases of the fold's training songs), the
ridge readout of canopy.studies.next_phrase.task is trained on the fold's training songs for every coefficient of RIDGES and
predicts the points of the fold's validation songs; the ridge coefficient is chosen on the out-of-fold points, stored
in the trial (user attribute "ridge") for the final evaluation; it is not a hyperparameter of the search. Search space:
canopy.studies.common.reservoir_params, with the leak rate of each reservoir with HAG_CANARY_LEAK=1.

Studies: "next_phrase_[calls_][reset_][leak_]<model>_<size>" (canopy.settings: HAG_CANARY_CALLS=1, the calls are
phrases too; HAG_CANARY_RESET=1, the reservoirs reset at each phrase; HAG_CANARY_LEAK=1) of the pure models
(canopy.models.reservoirs.PURE_MODELS), one SQLite database each in outputs/next_phrase/hpo (canopy.settings.studies_dir). (The
campaigns with the deltas of the MFCC and their results: outputs/next_phrase/ablation_deltas_leak; the studies of other readouts: bachue.)

Usage: python -m canopy.studies.next_phrase.study <model> <size>   (HAG_CANARY_N_TRIALS trials, HAG_CANARY_WORKERS
  processes optimizing the same study)
"""
import sys

import numpy as np
from sklearn.model_selection import KFold

from canopy.common import RIDGES, pretrain_phrases
from canopy.studies.common import longest_phrase, reservoir_params, run_study
from canopy.models.reservoirs import build
from canopy.settings import CV_SEED, FOLD_SEED, LEAK, N_FOLDS
from canopy.studies.next_phrase.task import fit_readout, load_task_dataset, point_states


# ---- Ridge coefficient of the readout of the next phrase, chosen by the 3-fold cross-validation by song on the train
# songs (out-of-fold accuracy; the reservoirs of fold k drawn with the seed CV_SEED + k): the objective of the studies,
# and the coefficient of the conditions without study in canopy.studies.next_phrase.evaluation,
# canopy.studies.next_phrase.certainty and canopy.studies.song_generation.generator.
def choose_ridge(model, params, size, songs, train_songs, frame, classes):
    """(ridge coefficient of the best out-of-fold accuracy, this accuracy), in a 3-fold cross-validation by song on the
    train songs (the reservoirs of fold k drawn with np.random.seed(CV_SEED + k), HAG learning on its training songs)."""
    correct = np.zeros(len(RIDGES))
    total = 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        states_of = point_states(reservoirs, songs, fit + val)
        true = np.concatenate([songs[w]["next"] for w in val])
        X_val = np.vstack([states_of[w] for w in val])
        for r, ridge in enumerate(RIDGES):
            scores = fit_readout(states_of, songs, fit, classes, ridge).run(X_val)
            correct[r] += (classes[scores.argmax(axis=1)] == true).sum()
        total += len(true)
    return float(RIDGES[int(np.argmax(correct))]), correct.max() / total


# ---- study
def make_objective(model, size):
    songs, train_songs, _, frame, classes = load_task_dataset()
    max_length = longest_phrase(songs, train_songs, frame)

    def objective(trial):
        params = reservoir_params(trial, model, max_length, leak=LEAK)
        ridge, accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
        trial.set_user_attr("ridge", ridge)
        return accuracy
    return objective


if __name__ == "__main__":
    run_study("next_phrase", sys.argv[1], int(sys.argv[2]), make_objective)
