import numpy as np
from hag.models.activation_functions import tanh
from hag.performances.esn_model_evaluation import init_reservoir

from canopy.models.reservoirs import states
from canopy.studies.song_generation.generator import (END, START, RecordedVoice, WorkingMemory, advance, choose, jax_nodes,
                                              markov_songs, ngrams, total_variation)


def reservoir(units=20, features=3, leak=0.3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1 / np.sqrt(units), (units, units))
    return init_reservoir(W, rng.uniform(-1, 1, (units, features)), rng.uniform(-0.1, 0.1, units), leak, tanh)


def test_advance_continues_the_states_of_each_series():
    """Running a series in two pieces, each series from its own state, gives the last state of the whole run."""
    rng = np.random.default_rng(1)
    reservoirs = [reservoir(seed=0), reservoir(units=10, seed=1)]
    series = [rng.uniform(0, 1, (n, 3)) for n in (70, 5, 130)]
    whole = [x[-1] for x in states(reservoirs, series)]
    nodes = jax_nodes(reservoirs)
    middle = advance(nodes, np.zeros((3, 30)), [x[:40] for x in series])
    end = advance(nodes, middle, [x[40:] for x in series])  # (the second one empty: its state kept)
    np.testing.assert_allclose(end, np.array(whole), atol=1e-5)


def test_choose():
    rng = np.random.default_rng(0)
    outputs = np.array([[0.2, 0.9, -0.3], [-1.0, -2.0, -0.5]])
    allowed = np.array([True, False, True])
    np.testing.assert_array_equal(choose(outputs, allowed, 0, rng), [0, 2])
    draws = np.array([choose(outputs, allowed, 1, rng) for _ in range(200)])
    assert set(draws[:, 0]) == {0} and set(draws[:, 1]) == {2}  # (only positive allowed outputs; else the largest)
    draws = np.array([choose(np.array([[0.3, 0.0, 0.1]]), np.ones(3, bool), 1, rng)[0] for _ in range(4000)])
    assert abs(np.mean(draws == 0) - 0.75) < 0.03 and not (draws == 1).any()


def test_the_recorded_phrases_of_a_song_in_order_are_the_song():
    rng = np.random.default_rng(0)
    songs = {"a": dict(mfcc=rng.uniform(size=(50, 2)), points=np.array([4, 12, 30, 41]),
                       next=np.array(["B", "C", "B", END], dtype=object)),
             "b": dict(mfcc=rng.uniform(size=(20, 2)), points=np.array([-1, 9, 19]),
                       next=np.array(["C", "C", END], dtype=object))}
    voice = RecordedVoice(songs, ["a", "b"])
    assert voice.phrases["B"] == [("a", 5, 12), ("a", 31, 41)] and voice.beginnings == [("a", 0, 4), ("b", 0, -1)]
    for w, beginning in zip(["a", "b"], voice.beginnings):
        mine = [source for label in voice.phrases for source in voice.phrases[label] if source[0] == w]
        frames = np.vstack([voice.frames(beginning)] + [voice.frames(s) for s in sorted(mine, key=lambda s: s[1])])
        np.testing.assert_array_equal(frames, songs[w]["mfcc"][:songs[w]["points"][-1] + 1])


def test_working_memory():
    memory = WorkingMemory(3)
    memory.write(np.array([0, 2]), np.array(["B", "C"], dtype=object))
    assert list(memory.read(np.arange(3))) == ["B", START, "C"]


def test_ngrams_and_markov_songs():
    songs = [(["A", "B"], True), (["A"], False)]
    assert ngrams(songs, 2) == {(START, "A"): 0.5, ("A", "B"): 0.25, ("B", END): 0.25}
    assert ngrams(songs, 1) == {("A",): 0.5, ("B",): 0.25, (END,): 0.25}
    assert total_variation({"x": 1.0}, {"y": 1.0}) == 1.0
    generated = markov_songs([(["A", "B", "A", "B"], True)], 50, max_phrases=10)
    assert all(labels[:1] == ["A"] and labels[-1:] == ["B"] for labels, ended in generated if ended)
