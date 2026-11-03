import numpy as np
import pytest

from canopy.models.generalized_readout import monomials
from canopy.models.readout_rules.eh import EH
from canopy.models.readout_rules.kernel import KernelEH, KernelLMS
from canopy.models.readout_rules.lms import LMS


def data(rng, n=80, d=6, k=3):
    X = rng.normal(size=(n, d))
    return X, np.eye(k)[rng.integers(0, k, n)]


def centered_monomials(X, X_test, order):
    F, F_test = monomials(X, order), monomials(X_test, order)
    mean = F.mean(axis=0)
    return F - mean, F_test - mean


@pytest.mark.parametrize("order", [1, 2, 3])
def test_kernel_lms_matches_nlms_on_the_monomials(order):
    rng = np.random.default_rng(0)
    (X, Y), (X_test, _) = data(rng), data(rng, n=20)
    F, F_test = centered_monomials(X, X_test, order)
    dual, primal = KernelLMS(X, order, learning_rate=0.3), LMS(learning_rate=0.3, normalized=True)
    for _ in range(3):
        rows = rng.permutation(len(X))
        dual.partial_fit(Y, rows)
        primal.partial_fit(F, Y, rows)
    np.testing.assert_allclose(dual.run(X_test), primal.run(F_test), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("baseline", ["filtered", "sample"])
@pytest.mark.parametrize("reward", ["per_output", "global"])
def test_kernel_eh_matches_normalized_eh_on_the_monomials(baseline, reward):
    rng = np.random.default_rng(0)
    (X, Y), (X_test, _) = data(rng), data(rng, n=20)
    F, F_test = centered_monomials(X, X_test, 2)
    options = dict(learning_rate=0.1, noise=0.5, baseline=baseline, reward=reward, seed=1)
    dual, primal = KernelEH(X, 2, **options), EH(**options, normalized=True)
    for _ in range(3):
        rows = rng.permutation(len(X))
        dual.partial_fit(Y, rows)
        primal.partial_fit(F, Y, rows)
    np.testing.assert_allclose(dual.run(X_test), primal.run(F_test), rtol=1e-8, atol=1e-10)
