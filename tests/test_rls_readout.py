import numpy as np
import pytest
from reservoirpy.nodes import RLS as RLSNode

from canopy.models.readout_rules.rls import RLS, with_constant


def data(rng, n=400, d=12):
    """Correlated inputs (eigenvalues over 6 decades) with a mean, targets with a mean far from 0."""
    X = rng.normal(size=(n, d)) @ (rng.normal(size=(d, d)) * 10.0 ** np.linspace(0, -3, d)) + rng.normal(size=d)
    return X, X @ rng.normal(size=(d, 3)) + 5.0 + 0.1 * rng.normal(size=(n, 3))


@pytest.mark.parametrize("alpha", [1e-6, 1e-3, 1.0])
def test_one_pass_gives_the_ridge_regression(alpha):
    rng = np.random.default_rng(0)
    X, Y = data(rng)
    readout = RLS(alpha=alpha)
    for rows in np.array_split(rng.permutation(len(X)), 20):  # (any order of the rows, in several calls)
        readout.partial_fit(X[rows], Y[rows])
    A = with_constant(X)
    ridge = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ Y)
    X_test = rng.normal(size=(50, X.shape[1]))
    np.testing.assert_allclose(readout.run(X_test), with_constant(X_test) @ ridge, rtol=1e-5, atol=1e-5)
    assert readout.Wout.shape == (12, 3) and readout.bias.shape == (3,)
    np.testing.assert_allclose(readout.run(X_test), X_test @ readout.Wout + readout.bias)


def test_the_bias_of_the_node_is_not_the_ridge_regression():
    # (why the constant input: the bias of reservoirpy's node, a running mean of the errors, is not the least squares)
    rng = np.random.default_rng(0)
    X, Y = data(rng)
    A = with_constant(X)
    ridge = A @ np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ Y)
    ours = RLS(alpha=1e-6).partial_fit(X, Y).run(X)
    node = RLSNode(alpha=1e-6)
    node.partial_fit(X, Y)
    assert np.abs(ours - ridge).max() < 1e-4 < np.abs(np.asarray(node.run(X)) - ridge).max()
