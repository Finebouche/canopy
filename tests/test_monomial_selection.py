import numpy as np
import pytest

from canopy.models.generalized_readout import GeneralizedRidge
from canopy.next_phrase_prediction.monomial_selection import Selection, correct_counts, ridges_of


def data(rng, n=900, d=12, songs=30):
    """Two classes given by the sign of x_0 x_1 (no linear information), plus a linear one on x_2."""
    X = rng.normal(size=(n, d))
    labels = (X[:, 0] * X[:, 1] > 0).astype(int) + 2 * (X[:, 2] > 0)
    return X, np.eye(4)[labels], np.repeat(np.arange(songs), n // songs), labels


def moments(X):
    return len(X), X.sum(axis=0), X.T @ X


@pytest.fixture(scope="module")
def selection():
    rng = np.random.default_rng(0)
    X, Y, groups, _ = data(rng)
    return Selection(X, Y, groups, moments(rng.normal(size=(5000, 12)) * np.arange(1, 13)), rho_linear=1.0)


def test_residual_finds_the_product_the_linear_readout_misses(selection):
    i, j = selection.unit_pairs("residual:1")
    assert (i[0], j[0]) == (0, 1)
    assert selection.weight_norms.argmax() == 2  # (the only unit with a linear effect)


@pytest.mark.parametrize("method, products", [("weights_high:4", 10), ("weights_low:4", 10), ("random:4", 10),
                                              ("residual:7", 7), ("random_pairs:7", 7), ("pca:3", 6),
                                              ("pca_frames:3", 6), ("scores", 10)])
def test_features(selection, method, products):
    F, inputs = selection.features(method)
    assert F.shape == (900, 12 + products)
    if method != "scores":  # (the outputs of the training rows are out-of-fold: not those of inputs)
        np.testing.assert_allclose(inputs(selection.X), F, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(F[:, 12:].mean(axis=0), 0, atol=1e-9)


def test_unit_selections(selection):
    order = np.argsort(selection.weight_norms)
    i, j = selection.unit_pairs("weights_high:4")
    assert set(i) | set(j) == set(order[-4:]) and np.all(i <= j) and len(set(zip(i, j))) == 10
    i, j = selection.unit_pairs("weights_low:4")
    assert set(i) | set(j) == set(order[:4])


def test_all_the_products_give_the_generalized_readout(selection):
    # all the products of the units, not standardized: the monomials of order 2 of z
    rng = np.random.default_rng(1)
    X_test = rng.normal(size=(50, 12))
    i, j = np.triu_indices(12)
    Z, Z_test = selection.Z, selection.standardized(X_test)
    explicit = GeneralizedRidge(order=1, ridge=1e-3).fit(np.hstack([Z, Z[:, i] * Z[:, j]]), selection.Y)
    full, inputs, n_inputs = selection.readout("full_std")
    assert n_inputs == 12 + 78
    np.testing.assert_allclose(full.set_ridge(1e-3).run(inputs(X_test)),
                               explicit.run(np.hstack([Z_test, Z_test[:, i] * Z_test[:, j]])), rtol=1e-6, atol=1e-8)


def test_selected_product_learns_the_classes(selection):
    rng = np.random.default_rng(2)
    X_test, _, _, labels = data(rng)
    classes = np.arange(4)
    counts = correct_counts(selection, ["linear", "residual:1", "random_pairs:1", "full_std"], X_test, labels, classes)
    accuracy = {method: count.max() / len(labels) for method, count in counts.items()}
    assert all(len(count) == len(ridges_of(method)) for method, count in counts.items())
    # (ridge regression on one-hot labels: the product alone gives ~0.78, all the 78 products on 900 rows ~0.64)
    assert accuracy["linear"] < 0.55 and accuracy["random_pairs:1"] < 0.55
    assert accuracy["residual:1"] > 0.7 and accuracy["full_std"] > 0.6


def test_diagnostics(selection):
    dual = GeneralizedRidge(order=2, solver="dual").fit(selection.Z, selection.Y).set_ridge(1.0)
    rows = {(row["measure"], row["method"], row["size"]): row["value"]
            for row in selection.diagnostics(dual, ["residual:1", "random_pairs:1", "weights_high:4", "pca:3"])}
    assert rows[("share", "residual:1", 1)] > 0.5 > rows[("share", "random_pairs:1", 1)]
    assert 0 < rows[("effective_parameters", "full_std", 78)] <= 900


@pytest.mark.parametrize("method, extra", [("squares", 12), ("random_pairs:n", 12), ("ofr:5", 5), ("leverage:7", 7),
                                           ("maclaurin:7", 7), ("nystrom:9", 9), ("r2sp:6", 6)])
def test_features_of_the_alternatives(selection, method, extra):
    F, inputs = selection.features(method)
    assert F.shape == (900, 12 + extra)
    np.testing.assert_allclose(inputs(selection.X), F, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(F[:, 12:].mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(F[:, 12:].std(axis=0), 1, atol=1e-9)


@pytest.mark.parametrize("method, kept, pairs", [
    ("half_squares", [0, 2, 4, 6, 8, 10], [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9), (11, 11)]),
    ("neighbors_t2", [0, 1, 3, 5, 7, 9, 11], [(1, 0), (3, 2), (5, 4), (7, 6), (9, 8)]),
    ("neighbors_t3", [0, 2, 4, 6, 8, 10, 11], [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)])])
def test_replaced_units(selection, method, kept, pairs):
    # (as many inputs as units: a unit out of two replaced by a product)
    F, inputs = selection.features(method)
    Z = selection.Z
    assert F.shape == (900, 12)
    np.testing.assert_allclose(F[:, :len(kept)], Z[:, kept])
    products = np.column_stack([Z[:, i] * Z[:, j] for i, j in pairs])
    np.testing.assert_allclose(F[:, len(kept):], (products - products.mean(axis=0)) / products.std(axis=0))
    np.testing.assert_allclose(inputs(selection.X), F, rtol=1e-9, atol=1e-9)


def test_forward_orthogonal_regression(selection):
    # the same pairs as the textbook algorithm on the explicit matrix of the candidates, orthogonalized at each step
    Z, T = selection.Z, len(selection.Z)
    i, j = np.triu_indices(12)
    candidates = Z[:, i] * Z[:, j]
    candidates = (candidates - candidates.mean(axis=0)) / candidates.std(axis=0)
    Q = np.linalg.qr(np.hstack([np.ones((T, 1)), Z]))[0]
    targets = selection.Y - selection.Y.mean(axis=0)
    W = candidates - Q @ (Q.T @ candidates)
    expected = []
    for _ in range(6):
        err = (((W.T @ targets) ** 2) / (targets ** 2).sum(axis=0)).sum(axis=1) / np.maximum((W ** 2).sum(axis=0), 1e-9)
        err[[np.flatnonzero((i == a) & (j == b))[0] for a, b in expected]] = -1
        best = err.argmax()
        expected.append((i[best], j[best]))
        q = W[:, best] / np.linalg.norm(W[:, best])
        W = W - np.outer(q, q @ W)
    a, b = selection.unit_pairs("ofr:6")
    assert list(zip(a, b)) == expected and expected[0] == (0, 1)  # (the product which gives the classes: first)
    a, b = selection.unit_pairs("ofr:2")  # (the first products of the longer selection)
    assert list(zip(a, b)) == expected[:2]


def test_leverage_scores(selection):
    # the diagonal of Phi (Phi^T Phi + ridge I)^-1 Phi^T, Phi the matrix of z and all the standardized products
    i, j = np.triu_indices(12)
    products = selection.Z[:, i] * selection.Z[:, j]
    Phi = np.hstack([selection.Z, (products - products.mean(axis=0)) / products.std(axis=0)])
    mu, V = np.linalg.eigh(Phi.T @ Phi)
    scores = selection.leverage_scores(20)[i, j]
    ridges = 10.0 ** np.linspace(-6, 8, 5000)
    ridge = ridges[np.argmin([abs(np.sum(mu / (mu + r)) - 20) for r in ridges])]
    expected = np.einsum("ir,r,ir->i", V, mu / (mu + ridge), V)[12:]
    np.testing.assert_allclose(scores, expected, rtol=2e-2)
    assert np.all(scores > 0) and np.all(scores < 1)
    a, b = selection.unit_pairs("leverage:7")
    assert len(set(zip(a, b))) == 7 and np.all(a <= b)
