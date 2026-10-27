import numpy as np
import pytest

from canary.eh_readout import EH


def clusters(rng, centers, n):
    labels = rng.integers(0, len(centers), n)
    return centers[labels] + rng.normal(size=(n, centers.shape[1])), np.eye(len(centers))[labels]


@pytest.mark.parametrize("reward", ["per_output", "global"])
@pytest.mark.parametrize("modulation", ["binary", "continuous"])
def test_sample_baseline_learns_classes(reward, modulation):
    # independent samples: the output without noise as baseline (node perturbation)
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(5, 50))
    X, Y = clusters(rng, centers, 3000)
    Xt, Yt = clusters(rng, centers, 1000)
    readout = EH(learning_rate=2e-3, noise=0.5, baseline="sample", reward=reward, modulation=modulation, seed=0)
    for _ in range(3):
        readout.partial_fit(X, Y)
    assert np.mean(readout.run(Xt).argmax(axis=1) == Yt.argmax(axis=1)) > 0.95


def test_filtered_baseline_learns_slow_inputs():
    # the paper's filtered baseline, on inputs held 20 steps (a slowly changing time series)
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(5, 50))
    X, Y = clusters(rng, centers, 600)
    Xt, Yt = clusters(rng, centers, 1000)
    X = np.repeat(X, 20, axis=0) + 0.05 * rng.normal(size=(12000, 50))
    readout = EH(learning_rate=2e-5, noise=0.5, baseline="filtered", alpha=0.5, seed=0)
    readout.partial_fit(X, np.repeat(Y, 20, axis=0))
    assert np.mean(readout.run(Xt).argmax(axis=1) == Yt.argmax(axis=1)) > 0.95


def test_run_is_deterministic_and_targets_needed():
    rng = np.random.default_rng(0)
    readout = EH(seed=0)
    readout.partial_fit(rng.normal(size=(10, 4)), rng.normal(size=(10, 2)))
    x = rng.normal(size=(3, 4))
    np.testing.assert_array_equal(readout.run(x), x @ readout.Wout + readout.bias)
    with pytest.raises(ValueError):
        readout.partial_fit(x, None)
    with pytest.raises(ValueError):
        EH(reward="local")
