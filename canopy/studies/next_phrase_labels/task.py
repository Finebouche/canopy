"""
Control of the prediction of the next phrase (canopy.studies.next_phrase): reservoirs that read the labels of the
phrases instead of the audio. Does the audio of a song say more about its next phrase than the sequence of its labels?
The n-grams of the labels (canopy.studies.next_phrase.ngrams) are a first reference, but they are short of data at
high orders; here the same reservoirs and the same ridge readout as the task on the audio read the sequence of the
labels, in the configuration of Maxime Canonne (ESN reading the phrase labels, with START and END).

Input of a song: one step per prediction point of canopy.studies.next_phrase.task (same songs, phrases, classes and
split): START, then each phrase, as a one-hot vector over START and the phrase labels (songs[w]["labels_input"]). The
state after a step is that of its prediction point: after START the readout predicts the first phrase, after the last
phrase END. With CANOPY_LABELS_WITH_DURATIONS=1, two more inputs at each phrase: the logarithm of its duration and that
of the silence before it (plus SILENCE_FLOOR), standardized on the train songs (zero at START); the reservoirs then
know the labels and their timing, but nothing of the acoustics of the phrases.

Readout and metrics: those of the task on the audio (canopy.studies.next_phrase.task.fit_readout, metrics).
"""
import numpy as np

from canopy.models.reservoirs import build, states
from canopy.settings import LABELS_WITH_DURATIONS
from canopy.studies.next_phrase.task import END, START, fit_readout, load_task_dataset as load_next_phrase, metrics

SILENCE_FLOOR = 0.01  # s, added to the silences before their logarithm (the first phrase has none)


def load_task_dataset():
    """(songs, train songs, test songs, classes) of the next phrase, with the input of each song,
    songs[w]["labels_input"] (points, features): see the docstring of the module."""
    songs, train_songs, test_songs, _, classes = load_next_phrase()
    tokens = np.array([START] + [label for label in classes if label != END], dtype=object)
    timing = {}
    for w, song in songs.items():
        phrases = song["next_phrases"]
        labels = np.array([START] + [label for _, _, label in phrases], dtype=object)
        song["labels_input"] = (labels[:, None] == tokens[None, :]).astype(np.float64)
        ends = [phrases[0][0]] + [end for _, end, _ in phrases[:-1]]  # (the end of the previous phrase)
        timing[w] = np.log([[end - start, max(start - previous, 0) + SILENCE_FLOOR]  # (no silence if they overlap)
                            for (start, end, _), previous in zip(phrases, ends)])
    if LABELS_WITH_DURATIONS:
        train = np.vstack([timing[w] for w in train_songs])
        mean, std = train.mean(axis=0), train.std(axis=0)
        for w, song in songs.items():
            standardized = np.vstack([np.zeros((1, 2)), (timing[w] - mean) / std])  # (zero at START)
            song["labels_input"] = np.hstack([song["labels_input"], standardized])
    return songs, train_songs, test_songs, classes


def pretrain_songs(songs, waves):
    """Inputs of the songs (HAG's pretraining instances)."""
    return np.array([songs[w]["labels_input"] for w in waves], dtype=object)


def point_states(reservoirs, songs, waves):
    """{wave: (points, units)} states of the reservoirs at the prediction points of the songs: after each step of
    their inputs, from a zero state (the inputs themselves without reservoir)."""
    runs = states(reservoirs, [songs[w]["labels_input"] for w in waves])
    return {w: np.asarray(x, dtype=np.float64) for w, x in zip(waves, runs)}


def evaluate(model, params, size, ridge, songs, train_songs, test_songs, classes, seed):
    """Metrics on the test songs: the reservoirs drawn with np.random.seed(seed), HAG learning on the inputs of the
    train songs, the readout trained on all the train songs."""
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_songs(songs, train_songs))
    states_of = point_states(reservoirs, songs, train_songs + test_songs)
    readout = fit_readout(states_of, songs, train_songs, classes, ridge)
    true = np.concatenate([songs[w]["next"] for w in test_songs])
    return metrics(readout.run(np.vstack([states_of[w] for w in test_songs])), true, classes)
