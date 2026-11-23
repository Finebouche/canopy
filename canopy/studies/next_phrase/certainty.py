"""
Certainty of the prediction of the next phrase (canopy.studies.next_phrase.task) along the current phrase: its readout
is trained at the prediction points of the train songs, then applied to every frame of each phrase of the test songs up
to its prediction point (no frame hearing the next phrase), the true class being the next phrase (END after the last
one); output of its unit, margin and correctness along the phrase (canopy.studies.common). With
CANOPY_RESET_EACH_PHRASE=1, the frames of a phrase are those of a run of the reservoirs from a zero state on the phrase.

Reservoirs: best parameters of the next-phrase studies (canopy.studies.next_phrase.study, with the same variables, e.g.
CANOPY_LEAKY_RESERVOIRS=1, CANOPY_RESET_EACH_PHRASE=1, CANOPY_WITH_CALLS=1), ridge of the best trial of the study.
Results: outputs/next_phrase/analysis/<prefix of the studies>certainty_<size>.csv, ..._phrases.csv and .png (e.g.
next_phrase_leak_certainty_1000.csv).

Usage: python -m canopy.studies.next_phrase.certainty [size] [conditions] [nb_seeds]
  (default 1000, mean_hag,var_hag,random_ee,random_ei, 3 seeds)
"""
import sys

import numpy as np

from canopy.studies.common import MIN_FRAMES, curve, evaluate_conditions, save_certainty, summarize
from canopy.common import pretrain_phrases
from canopy.models.reservoirs import CHUNK, build, state_chunks, states
from canopy.settings import PREFIXES, RESET
from canopy.studies.next_phrase.study import choose_ridge
from canopy.studies.next_phrase.task import fit_readout, load_task_dataset, point_states


def run(condition, model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed):
    """Curves of the readout of the next phrase on the frames of each phrase of the test songs, from its onset to its
    prediction point (canopy.studies.next_phrase.task), the phrases of fewer than MIN_FRAMES frames
    skipped."""
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    readout = fit_readout(point_states(reservoirs, songs, train_songs), songs, train_songs, classes, ridge)
    Wout = np.vstack([readout.Wout, readout.bias[None]])
    index = {label: k for k, label in enumerate(classes)}
    curves = []
    for i in range(0, len(test_songs), CHUNK):
        part = test_songs[i:i + CHUNK]
        phrases = [(w, j, int(np.floor(start / frame)), songs[w]["points"][j + 1] + 1)
                   for w in part for j, (start, _, _) in enumerate(songs[w]["next_phrases"])]
        phrases = [(w, j, f0, f1) for w, j, f0, f1 in phrases if f1 - f0 >= MIN_FRAMES]
        if RESET:  # runs of the reservoirs on each phrase, from a zero state
            runs = states(reservoirs, [songs[w]["mfcc"][f0:f1] for w, _, f0, f1 in phrases])
            outputs = [run.astype(np.float64) @ Wout[:-1] + Wout[-1] for run in runs]
        else:
            song_outputs = {w: x.astype(np.float64) @ Wout[:-1] + Wout[-1]
                            for waves, xs in state_chunks(reservoirs, songs, part) for w, x in zip(waves, xs)}
            outputs = [song_outputs[w][f0:f1] for w, _, f0, f1 in phrases]
        for (w, j, _, _), out in zip(phrases, outputs):
            label = songs[w]["next"][j + 1]
            curves.append(curve(out, index[label], label, frame))
    return summarize(curves, condition, seed)


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else ["mean_hag", "var_hag", "random_ee", "random_ei"]
    nb_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    songs, train_songs, test_songs, frame, classes = load_task_dataset()
    bins, phrases = evaluate_conditions("next_phrase", size, wanted, nb_seeds, songs, train_songs, test_songs, frame,
                                        classes, choose_ridge, run)
    save_certainty(bins, phrases, "next_phrase", f"{PREFIXES['next_phrase']}certainty", size,
                   "Certainty of the next phrase along the phrases",
                   ("Output of the next-phrase unit", "Accuracy of the next phrase"))
