import numpy as np
import pytest
from reservoirpy.nodes import LMS as NumpyLMS

from canary.eh_readout import EH as NumpyEH
from canary.next_phrase_prediction.readout_rules.__main__ import trained
from canary.next_phrase_prediction.readout_rules.eh import EH
from canary.next_phrase_prediction.readout_rules.lms import LMS


def data(rng, n=300, d=20, k=4):
    X = rng.normal(size=(n, d))
    return X, np.eye(k)[rng.integers(0, k, n)]


def test_lms_matches_reservoirpy():
    rng = np.random.default_rng(0)
    X, Y = data(rng)
    jax_readout, numpy_readout = LMS(learning_rate=1e-2), NumpyLMS(learning_rate=1e-2)
    for _ in range(3):
        order = rng.permutation(len(X))
        jax_readout.partial_fit(X, Y, order)
        numpy_readout.partial_fit(X[order], Y[order])
    np.testing.assert_allclose(jax_readout.run(X), numpy_readout.run(X), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("baseline", ["filtered", "sample"])
@pytest.mark.parametrize("reward", ["per_output", "global"])
@pytest.mark.parametrize("modulation", ["binary", "continuous"])
@pytest.mark.parametrize("decay", [None, 100.0])
def test_eh_matches_numpy_node(baseline, reward, modulation, decay):
    rng = np.random.default_rng(0)
    X, Y = data(rng)
    options = dict(learning_rate=1e-2, noise=0.5, baseline=baseline, reward=reward, modulation=modulation, decay=decay)
    jax_readout, numpy_readout = EH(**options, seed=1), NumpyEH(**options, seed=1)
    for _ in range(3):
        order = rng.permutation(len(X))
        jax_readout.partial_fit(X, Y, order)
        numpy_readout.partial_fit(X[order], Y[order])
    np.testing.assert_allclose(jax_readout.run(X), numpy_readout.run(X), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("rule", ["lms", "eh", "eh_np", "eh_np_local"])
def test_trained_matches_numpy_nodes(rule):
    # the songs in the order of the generator at each epoch, as the loop over the songs of the NumPy nodes
    rng = np.random.default_rng(0)
    rows = {f"song{i}": data(rng, n=int(rng.integers(5, 30))) for i in range(8)}
    fit = list(rows)[:6]
    hyper = dict(rate=0.1, noise=0.5)
    curves = list(trained(rule, hyper, rows, fit, np.random.default_rng(3), [1, 3]))
    mean_norm = np.mean(np.concatenate([np.sum(rows[w][0] ** 2, axis=1) for w in fit])) + 1
    numpy_rng = np.random.default_rng(3)
    if rule == "lms":
        node = NumpyLMS(learning_rate=hyper["rate"] / mean_norm)
    else:
        node = NumpyEH(learning_rate=hyper["rate"] / mean_norm, noise=hyper["noise"],
                       reward="per_output" if rule == "eh_np_local" else "global",
                       baseline="filtered" if rule == "eh" else "sample", seed=numpy_rng)
    for _ in range(3):
        for w in numpy_rng.permutation(fit):
            node.partial_fit(*rows[w])
    assert [epoch for epoch, _ in curves] == [1, 3]
    X = np.vstack([rows[w][0] for w in rows])  # (trained yields the same node, trained further: its last state)
    np.testing.assert_allclose(curves[-1][1].run(X), node.run(X), rtol=1e-10, atol=1e-12)
