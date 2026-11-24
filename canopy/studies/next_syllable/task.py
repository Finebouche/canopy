"""
Reservoirs that read the labels of the syllables, one step per syllable, and predict the next syllable: a control of
the prediction of the next phrase (canopy.studies.next_phrase), between the reservoirs that read the audio and those
that read one label per phrase (canopy.studies.next_phrase_labels). Inside a phrase the next syllable is the same
type; at the last syllable of a phrase it is the next phrase. To predict the end of a phrase, the reservoirs must
count its syllables (or learn the usual length of each phrase type).

Input of a song: START, then each syllable of its phrases, as a one-hot vector over START and the phrase labels (a
canary phrase repeats a single syllable type, so the label of a syllable is that of its phrase). The phrases are those
of the next phrase with the calls (CANOPY_WITH_CALLS=1 only) and their numbers of syllables those of
canopy.studies.next_syllable.segmentation (outputs/next_syllable/analysis/syllable_counts.csv, from the audio). Arrays
of a song: songs[w]["syllables_input"] (steps, features), songs[w]["next_syllable"] (label of the next syllable after
each step, END after the last one) and songs[w]["phrase_steps"] (the step of each prediction point of the next phrase:
START, then the last syllable of each phrase; there the next syllable is songs[w]["next"]) and songs[w]["current"]
(the syllable type heard at each prediction point, START at the first).

Readout (CANOPY_SYLLABLE_READOUT, canopy.settings.SYLLABLE_READOUT): ridge regression of the one-hot next syllable on
the states of all the steps of the train songs ("syllables", the default), or of the one-hot next phrase on their states
at the prediction points of the next phrase only ("phrases": the reservoirs still advance one syllable per step, and the
readout is trained and read as that of the reservoirs on the labels of the phrases). Metrics: those of the next phrase
(canopy.studies.next_phrase.task.metrics) on the prediction points of the next phrase, the accuracy there knowing that
the phrase changes (accuracy_change_known: the most likely class other than the current syllable type; the reservoirs on
the labels of the phrases know the changes, one step per phrase), and, for the readout of the syllables, the accuracy of
the next syllable on all the steps (syllable_accuracy).
"""
import numpy as np
import pandas as pd
from reservoirpy.nodes import Ridge

from canopy.models.reservoirs import build, states
from canopy.settings import CALLS, RIDGE_SOLVER, SYLLABLE_READOUT, outputs_dir
from canopy.studies.next_phrase.task import END, START, load_task_dataset as load_next_phrase, metrics, one_hot

COUNTS = outputs_dir("next_syllable", "analysis") / "syllable_counts.csv"


def load_task_dataset():
    """(songs, train songs, test songs, classes) of the next phrase with the calls, with the inputs, the next syllables
    and the steps of the prediction points of each song (see the docstring of the module)."""
    if not CALLS:
        raise ValueError("canopy.studies.next_syllable: the syllables are counted in the phrases with the calls "
                         "(CANOPY_WITH_CALLS=1)")
    songs, train_songs, test_songs, _, classes = load_next_phrase()
    counts = pd.read_csv(COUNTS).sort_values(["wave", "phrase"])
    tokens = np.array([START] + [label for label in classes if label != END], dtype=object)
    for w, rows in counts.groupby("wave"):
        if w not in songs:
            continue
        song = songs[w]
        labels, n = rows["label"].tolist(), rows["n_syllables"].to_numpy()
        if labels != [label for _, _, label in song["next_phrases"]]:
            raise ValueError(f"{w}: the phrases of {COUNTS} are not those of the next phrase")
        steps = np.array([START] + [label for label, k in zip(labels, n) for _ in range(k)], dtype=object)
        song["syllables_input"] = (steps[:, None] == tokens[None, :]).astype(np.float64)
        song["next_syllable"] = np.concatenate([steps[1:], [END]])
        song["phrase_steps"] = np.concatenate([[0], np.cumsum(n)])
        song["current"] = np.array([START] + labels, dtype=object)  # (the syllable heard at each point)
        assert (song["next_syllable"][song["phrase_steps"]] == song["next"]).all()
    missing = [w for w in songs if "syllables_input" not in songs[w]]
    if missing:
        raise ValueError(f"{len(missing)} songs without their syllables in {COUNTS} (e.g. {missing[0]})")
    return songs, train_songs, test_songs, classes


def pretrain_songs(songs, waves):
    """Inputs of the songs (HAG's pretraining instances)."""
    return np.array([songs[w]["syllables_input"] for w in waves], dtype=object)


def step_states(reservoirs, songs, waves):
    """{wave: (steps, units)} states of the reservoirs after each step of the songs, from a zero state (the inputs
    themselves without reservoir)."""
    runs = states(reservoirs, [songs[w]["syllables_input"] for w in waves])
    return {w: np.asarray(x, dtype=np.float64) for w, x in zip(waves, runs)}


def one_hot_syllables(songs, waves, classes):
    """(steps, classes) one-hot next syllables of all the steps of the songs, in the order of the waves."""
    labels = np.concatenate([songs[w]["next_syllable"] for w in waves])
    return (labels[:, None] == classes[None, :]).astype(np.float64)


def readout_rows(states_of, songs, waves, classes):
    """(inputs, one-hot targets) of the readout on the songs waves: all the steps and their next syllables, or with
    SYLLABLE_READOUT "phrases" the prediction points of the next phrase and their next phrases."""
    if SYLLABLE_READOUT == "phrases":
        return np.vstack([states_of[w][songs[w]["phrase_steps"]] for w in waves]), one_hot(songs, waves, classes)
    return np.vstack([states_of[w] for w in waves]), one_hot_syllables(songs, waves, classes)


def scores_of(readout_outputs, songs, waves):
    """(scores at the prediction points of the next phrase, scores at all the steps or None), from the outputs of the
    readout on readout_rows of the songs waves."""
    if SYLLABLE_READOUT == "phrases":
        return readout_outputs, None
    bounds = np.cumsum([0] + [len(songs[w]["next_syllable"]) for w in waves])
    points = np.concatenate([bounds[i] + songs[w]["phrase_steps"] for i, w in enumerate(waves)])
    return readout_outputs[points], readout_outputs


def changed(scores, current, classes):
    """Predictions of the scores at the prediction points of the next phrase knowing that the phrase changes: the most
    likely class other than the current syllable type."""
    masked = np.where(classes[None, :] == current[:, None], -np.inf, scores)
    return classes[masked.argmax(axis=1)]


def evaluate(model, params, size, ridge, songs, train_songs, test_songs, classes, seed):
    """Metrics on the test songs: the reservoirs drawn with np.random.seed(seed), HAG learning on the inputs of the
    train songs, the readout trained on the train songs (readout_rows)."""
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_songs(songs, train_songs))
    states_of = step_states(reservoirs, songs, train_songs + test_songs)
    readout = Ridge(ridge=ridge, solver=RIDGE_SOLVER).fit(*readout_rows(states_of, songs, train_songs, classes))
    at_points, at_steps = scores_of(readout.run(readout_rows(states_of, songs, test_songs, classes)[0]), songs,
                                    test_songs)
    result = metrics(at_points, np.concatenate([songs[w]["next"] for w in test_songs]), classes)
    result["syllable_accuracy"] = np.nan
    if at_steps is not None:
        true_steps = np.concatenate([songs[w]["next_syllable"] for w in test_songs])
        result["syllable_accuracy"] = float(np.mean(classes[at_steps.argmax(axis=1)] == true_steps))
    current = np.concatenate([songs[w]["current"] for w in test_songs])
    result["accuracy_change_known"] = float(np.mean(changed(at_points, current, classes)
                                                    == np.concatenate([songs[w]["next"] for w in test_songs])))
    return result
