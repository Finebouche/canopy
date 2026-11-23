"""
Syllable classification of canary songs, frame by frame, as canapy (https://github.com/birds-canopy/canapy, Trouvain &
Hinaut): the readout predicts the label of every frame (the 27 syllable types, call, TRASH and SIL), and the score is
canapy's frame accuracy: the share of the frames of the recordings (silences included) whose raw prediction (most
likely class, no post-processing) is the annotated label. A frame k is labeled with the annotated segment that covers
it, frames from floor(onset / hop) to floor(offset / hop) (canapy's as_frame_comparison), SIL elsewhere.

Readout: ridge regression on the one-hot labels (centered statistics, canopy.models.reservoirs.normal_equations), its
coefficient chosen by the 3-fold cross-validation by song on the train songs of the studies
(canopy.studies.syllables_classification.study.choose_ridge), then trained on all the train songs and scored on the test songs (evaluate).
"""
import numpy as np

from canopy.common import load_dataset, pretrain_phrases
from canopy.datasets.songs import annotations_of_songs
from canopy.models.reservoirs import build, normal_equations, ridge_readout, state_chunks

SILENCE = "SIL"


def add_frame_labels(songs, frame):
    """songs[w]["labels"]: label of each frame (all the annotated labels, SIL between segments); returns the classes."""
    annotations = annotations_of_songs()
    for wave, segments in annotations[annotations["wave"].isin(list(songs))].groupby("wave"):
        n_frames = len(songs[wave]["mfcc"])
        labels = np.full(n_frames, SILENCE, dtype=object)
        for start, end, label in segments.sort_values("start")[["start", "end", "syll"]].itertuples(index=False):
            labels[int(np.floor(start / frame)):min(int(np.floor(end / frame)), n_frames)] = label
        songs[wave]["labels"] = labels
    classes = np.array(sorted({label for song in songs.values() for label in song["labels"]}))
    for song in songs.values():
        song["Y"] = (song["labels"][:, None] == classes[None, :]).astype(np.float64)
    return classes


def correct_frames(reservoirs, songs, waves, Wouts, classes):
    """(correct frames per readout of Wouts, correct non-silent frames per readout, frames, non-silent frames)."""
    stacked = np.concatenate(Wouts, axis=1)  # (units + 1, readouts * classes)
    n_classes = len(classes)
    correct, correct_sound, total, total_sound = np.zeros(len(Wouts)), np.zeros(len(Wouts)), 0, 0
    for part, part_states in state_chunks(reservoirs, songs, waves):
        for x, w in zip(part_states, part):
            outputs = x.astype(np.float64) @ stacked[:-1] + stacked[-1]
            predicted = classes[outputs.reshape(len(x), len(Wouts), n_classes).argmax(axis=2)]  # (T, readouts)
            labels = songs[w]["labels"][:, None]
            sound = labels[:, 0] != SILENCE
            correct += (predicted == labels).sum(axis=0)
            correct_sound += (predicted[sound] == labels[sound]).sum(axis=0)
            total += len(x)
            total_sound += int(sound.sum())
    return correct, correct_sound, total, total_sound


def evaluate(model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed):
    """Frame accuracies on the test songs (all the frames, and without the silences): the reservoirs drawn with
    np.random.seed(seed), HAG learning on phrases of the train songs, the readout trained on all the train songs."""
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    Wout = ridge_readout(normal_equations(reservoirs, songs, train_songs, target="Y"), ridge)
    correct, correct_sound, total, total_sound = correct_frames(reservoirs, songs, test_songs, [Wout], classes)
    return dict(frame_accuracy=correct[0] / total, frame_accuracy_without_silence=correct_sound[0] / total_sound)


def load_task_dataset():
    """(songs, train songs, test songs, frame, classes): canopy.common.load_dataset and the labels of the frames
    (add_frame_labels)."""
    songs, train_songs, test_songs, frame = load_dataset()
    return songs, train_songs, test_songs, frame, add_frame_labels(songs, frame)
