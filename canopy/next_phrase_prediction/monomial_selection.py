"""
Selection of the monomials of degree 2 of the generalized readout (canopy.models.generalized_readout, "ridge:2" of
canopy.next_phrase_prediction.readout_rules: N + N (N + 1) / 2 monomials of the N units, 502 502 at 1001 units): can a
few hundred or thousand products, chosen before the regression, replace all of them? Exploratory: all the readouts
read the same reservoirs, those of the studies of the variables of the environment (HAG_CANARY_HPO_RULE=ridge:2: the
reservoirs optimized for the full readout of order 2, which they approximate), not reservoirs optimized for each.

Readouts ("methods"), ridge regressions (unpenalized bias) on the last state x of the window of each prediction point
(the readout "last" -> "last"), z = x standardized with the training rows:
  - "linear": on z;
  - "full": the generalized readout of order 2 on x (the rule "ridge:2"); "full_std": on z (all the products of z,
    the limit of the selections);
  - on z and selected products of two components, each product standardized with the training rows:
      "weights_high:m" products of the m units of largest weight norm ||w_i|| in the linear readout (heredity);
      "weights_low:m"  of the m units of smallest weight norm (the opposite, control);
      "random:m"       of m random units (control);
      "residual:M"     the M products z_i z_j the most correlated with the residual of the linear readout: largest
                       sum over the outputs k of Cov(z_i z_j, r_k)^2 / Var(z_i z_j), r the out-of-fold residual of the
                       linear readout on the training rows (INNER_FOLDS folds by song);
      "random_pairs:M" M random products (control);
      "pca:m"          products of the m first principal components of z (the training rows: the states at the
                       prediction points);
      "pca_frames:m"   products of the m first principal components of the states of all the frames of the phrases of
                       the training songs (the history of the units, without labels), standardized with these frames;
      "scores"         products of the outputs of the linear readout (out-of-fold on the training rows): 435 for 29
                       outputs.
    m in SIZES (default 32, 64, 128), M = m (m + 1) / 2 (528, 2080, 8256): the same numbers of products.
  - published or theoretical alternatives, for the comparison with the random pairs (not in the default methods; M
    can be "n", the number of units), each extra input standardized with the training rows:
      "squares"        z and all the squares z_i^2: N products (Jaeger 2002; Pathak et al. 2018);
      "half_squares"   a unit out of two replaced by its square: N inputs (Pathak et al. 2017);
      "neighbors_t2", "neighbors_t3"  a unit out of two replaced by the product of two neighbors by the index,
                       z_(j-1) z_(j-2) or z_(j-1) z_(j+1): N inputs (T2 and T3 of Chattopadhyay et al. 2020);
      "ofr:M"          M products chosen one by one by forward orthogonal regression: at each step, the product of
                       largest error reduction ratio (sum over the outputs) once orthogonalized against the constant,
                       z and the products already chosen (Chen, Billings & Luo 1989; on reservoir states: Lupascu &
                       Coca 2026);
      "leverage:M"     M products drawn without replacement with probabilities proportional to their ridge leverage
                       scores in the matrix of z and all the standardized products, the coefficient such that its
                       effective dimension is min(M, T / 2) (the sampling of the columns of a ridge regression, El
                       Alaoui & Mahoney 2015; for the polynomial kernels: Woodruff & Zandieh 2020, whose embedding is
                       not this exact computation);
      "maclaurin:M"    M products (w . z)(w' . z) of two random projections of signs +-1 / sqrt(N) (the terms of
                       degree 2 of the random Maclaurin features of Kar & Karnick 2012, z adjoined as they propose);
      "nystrom:M"      the Nystrom features of the kernel of the products, 1/2 ((a . b)^2 + sum_i a_i^2 b_i^2), on M
                       random training rows (Williams & Seeger 2001);
      "r2sp:M"         M random units tanh(w . z + b), w and b normal (variances 1 / N and 1): a random nonlinearity
                       which is not a product (the static projection of R2SP, Butcher et al. 2013).
    All on z, not on the raw states of the papers (a large common mean here: their squares would be almost linear).
The selections are computed on the training rows only (those of the fold in the cross-validation). The coefficient
of the linear readout used by the selections (weights, residual, scores) is that of "linear" in the cross-validation.

Protocol of canopy.next_phrase_prediction.readout_rules: the ridge coefficient of each method chosen by the 3-fold
cross-validation by song on the train songs (reservoirs of fold k drawn with the seed CV_SEED + k), then each method
trained on all the train songs and scored on the test songs for each seed (reservoirs drawn with the seed).
Diagnostics (seed 0, from the trained "full_std"): the importance of each product, sum over the outputs of its squared
weight times its variance; the share of the total importance of the M most important products ("top_pairs"), of the
products of each selection of units or pairs ("share"), and the effective number of parameters of "full_std"
("effective_parameters": sum of mu / (mu + ridge) over the eigenvalues mu of the centered kernel).

Results: outputs/canary/<prefix>monomial_selection.csv (one row per condition, size, seed and method),
<prefix>monomial_selection_cv.csv (out-of-fold accuracy of each method and coefficient) and
<prefix>monomial_selection_diagnostics.csv.
Usage (HAG_CANARY_TASK=next_phrase HAG_CANARY_HPO_RULE=ridge:2, with the variables of the studies):
  python -m canopy.next_phrase_prediction.monomial_selection run <condition> <size> [nb_seeds]   (results in _parts)
  python -m canopy.next_phrase_prediction.monomial_selection merge
  python -m canopy.next_phrase_prediction.monomial_selection list <sizes>      "<condition> <size>" of the pure models
"""
import os
import sys
import zlib
from functools import cached_property

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold

from canopy.common import OUTPUTS, RIDGES, load_dataset, pretrain_phrases
from canopy.hpo_utils import CV_SEED, FOLD_SEED
from canopy.models.generalized_readout import GeneralizedRidge
from canopy.models.reservoirs import build
from canopy.next_phrase_prediction.hpo import N_FOLDS, STUDY_PREFIX
from canopy.next_phrase_prediction.prediction import all_conditions, metrics, point_windows, prepare

SIZES = [int(m) for m in os.environ.get("HAG_CANARY_SELECTION_SIZES", "32,64,128").split(",")]
UNIT_KINDS = ("weights_high", "weights_low", "random")  # (all the products of m units)
PAIR_KINDS = ("residual", "random_pairs", "squares", "ofr", "leverage")  # (products of two units)
REPLACED_KINDS = ("half_squares", "neighbors_t2", "neighbors_t3")  # (a unit out of two replaced by a product)
COMPONENT_KINDS = ("pca", "pca_frames")  # (all the products of m components)
MAP_KINDS = ("nystrom", "r2sp")  # (M features which are not products of two components)
DEFAULT_METHODS = (["linear", "full", "full_std", "scores"]
                   + [f"{kind}:{m}" for kind in UNIT_KINDS + COMPONENT_KINDS for m in SIZES]
                   + [f"{kind}:{m * (m + 1) // 2}" for kind in ("residual", "random_pairs") for m in SIZES])
METHODS = os.environ.get("HAG_CANARY_SELECTION_METHODS", ",".join(DEFAULT_METHODS)).split(",")
RIDGES_STD = 10.0 ** np.arange(-10, 5)  # (coefficients of the standardized inputs: the best ones from 1e-6 to 1e2)
INNER_FOLDS = 3  # (out-of-fold outputs of the linear readout on the training rows)
TOP_PAIRS = [m * (m + 1) // 2 for m in SIZES] + [50000]  # (diagnostics)
JOBS = int(os.environ.get("HAG_CANARY_RULES_JOBS", "1"))  # processes (folds, then seeds)
RESULTS = f"{STUDY_PREFIX}monomial_selection{os.environ.get('HAG_CANARY_SELECTION_SUFFIX', '')}"
PARTS = OUTPUTS / f"{RESULTS}_parts"


def ridges_of(method):
    return RIDGES if method == "full" else RIDGES_STD


class Selection:
    """The selections of products of a set of training rows: states X (T, N), one-hot labels Y, the song of each row
    groups (T,), the moments of the frames of the training songs (number, sum (N,), Gram matrix (N, N)) or None, the
    coefficient of the linear readout rho_linear (None: chosen on the rows themselves, chosen_ridge; for the outputs
    of cross_scores, without the rows of the fold they are computed on)."""

    def __init__(self, X, Y, groups, moments, rho_linear, seed=0):
        self.X, self.Y, self.groups, self.moments, self.rho_linear, self.seed = X, Y, groups, moments, rho_linear, seed
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-8
        self.Z = self.standardized(X)

    def standardized(self, X):
        return (X - self.mean) / self.std

    def chosen_ridge(self, rows):
        """Coefficient of RIDGES_STD of the best accuracy of the linear readout in the cross-validation by song
        (INNER_FOLDS folds) on the training rows of indices rows only."""
        Z, Y, labels = self.Z[rows], self.Y[rows], self.Y[rows].argmax(axis=1)
        correct = np.zeros(len(RIDGES_STD))
        for fit, held in GroupKFold(INNER_FOLDS).split(Z, groups=self.groups[rows]):
            readout = GeneralizedRidge(order=1, solver="primal").fit(Z[fit], Y[fit])
            for r, ridge in enumerate(RIDGES_STD):
                correct[r] += np.sum(readout.set_ridge(ridge).run(Z[held]).argmax(axis=1) == labels[held])
        return float(RIDGES_STD[correct.argmax()])

    @cached_property
    def linear_ridge(self):
        """Coefficient of the linear readout trained on all the training rows: rho_linear, or (None) chosen_ridge of
        all the training rows."""
        return self.rho_linear if self.rho_linear is not None else self.chosen_ridge(np.arange(len(self.Z)))

    def linear_readout(self, rows=slice(None), ridge=None):
        """Linear readout trained on the training rows of indices rows, coefficient ridge (default linear_ridge)."""
        return GeneralizedRidge(order=1, solver="primal").fit(self.Z[rows], self.Y[rows]).set_ridge(
            self.linear_ridge if ridge is None else ridge)

    @cached_property
    def linear(self):
        return self.linear_readout()

    @cached_property
    def weight_norms(self):
        """Norm (N,) of the weights of each unit in the linear readout."""
        return np.linalg.norm(self.linear.Wout, axis=1)

    @cached_property
    def cross_scores(self):
        """Outputs (T, outputs) of the linear readout on the training rows, each song by a readout trained without its
        fold: its weights and (rho_linear None) its coefficient, chosen by chosen_ridge on the other folds only (nested
        cross-validation: no label of a row in its own outputs)."""
        scores = np.empty_like(self.Y)
        for fit, held in GroupKFold(INNER_FOLDS).split(self.Z, groups=self.groups):
            ridge = self.rho_linear if self.rho_linear is not None else self.chosen_ridge(fit)
            scores[held] = self.linear_readout(fit, ridge).run(self.Z[held])
        return scores

    @cached_property
    def pair_variance(self):
        """Variance (N, N) of each product z_i z_j over the training rows."""
        T = len(self.Z)
        return np.maximum((self.Z ** 2).T @ (self.Z ** 2) / T - (self.Z.T @ self.Z / T) ** 2, 0)

    @cached_property
    def residual_scores(self):
        """Score (N, N) of each product: sum over the outputs k of Cov(z_i z_j, r_k)^2 / Var(z_i z_j)."""
        residual = self.Y - self.cross_scores
        residual -= residual.mean(axis=0)
        squared = np.zeros((self.Z.shape[1],) * 2)
        for k in range(residual.shape[1]):
            squared += ((self.Z * residual[:, k:k + 1]).T @ self.Z / len(self.Z)) ** 2
        return squared / np.maximum(self.pair_variance, 1e-12)

    @staticmethod
    def principal_axes(correlation):
        """Eigenvectors (N, N) of the correlation matrix, by decreasing eigenvalue."""
        values, vectors = np.linalg.eigh(correlation)
        return vectors[:, ::-1]

    @cached_property
    def point_axes(self):
        return self.principal_axes(self.Z.T @ self.Z / len(self.Z))

    @cached_property
    def frame_statistics(self):
        """(mean (N,), standard deviation (N,), principal axes (N, N)) of the states of the frames."""
        n, total, gram = self.moments
        mean = total / n
        covariance = gram / n - np.outer(mean, mean)
        std = np.sqrt(np.maximum(np.diag(covariance), 0)) + 1e-8
        return mean, std, self.principal_axes(covariance / np.outer(std, std))

    def rng(self, method):
        return np.random.default_rng([self.seed, zlib.crc32(method.encode())])

    def size_of(self, method):
        """Number after the ":" of a method ("n": the number of units)."""
        size = method.partition(":")[2]
        return self.Z.shape[1] if size == "n" else int(size)

    @cached_property
    def product_statistics(self):
        """(mean (N, N), standard deviation (N, N)) of each product z_i z_j over the training rows (a constant
        product: an infinite standard deviation, its standardized values 0)."""
        std = np.sqrt(self.pair_variance)
        std[std < 1e-8] = np.inf
        return self.Z.T @ self.Z / len(self.Z), std

    def product_correlations(self, V):
        """(k, N, N): for each column v of V (T, k), the scalar products of v with the standardized products of the
        training rows, sum_t v_t (z_ti z_tj - mean_ij) / std_ij."""
        mean, std = self.product_statistics
        return np.stack([(self.Z.T @ (self.Z * v[:, None]) - mean * v.sum()) / std for v in V.T])

    def ofr_pairs(self, size):
        """(i, j) units of the size products chosen by forward orthogonal regression (see the module docstring): the
        scalar products with all the candidates updated at each step, without their orthogonalized matrix. The first
        products of a longer selection already computed are reused (the choice of a step does not depend on the next)."""
        if len(getattr(self, "_ofr", ((),))[0]) >= size:
            return tuple(units[:size] for units in self._ofr)
        Z, T = self.Z, len(self.Z)
        mean, std = self.product_statistics
        targets = self.Y - self.Y.mean(axis=0)
        energy = np.maximum((targets ** 2).sum(axis=0), 1e-12)
        Q = np.linalg.qr(np.hstack([np.ones((T, 1)), Z]))[0]  # (the constant and z: always in the readout)
        residual = targets - Q @ (Q.T @ targets)
        norms = np.where(np.isfinite(std), float(T), 0.0)  # (squared norms of the orthogonalized candidates)
        for start in range(0, Q.shape[1], 64):
            norms -= (self.product_correlations(Q[:, start:start + 64]) ** 2).sum(axis=0)
        G = self.product_correlations(residual)  # (outputs, N, N)
        allowed = np.triu(np.ones_like(norms, dtype=bool))
        chosen = []
        for _ in range(size):
            valid = allowed & (norms > 1e-8 * T)
            if not valid.any():
                break
            err = np.where(valid, ((G ** 2) / energy[:, None, None]).sum(axis=0) / np.maximum(norms, 1e-300), -1.0)
            i, j = np.unravel_index(err.argmax(), err.shape)
            chosen.append((i, j))
            allowed[i, j] = False
            q = (Z[:, i] * Z[:, j] - mean[i, j]) / std[i, j]
            q -= Q @ (Q.T @ q)
            q -= Q @ (Q.T @ q)  # (a second pass: the orthogonality kept over thousands of steps)
            q /= np.linalg.norm(q)
            c = self.product_correlations(q[:, None])[0]
            gain = q @ residual
            norms -= c ** 2
            G -= c[None] * gain[:, None, None]
            residual -= np.outer(q, gain)
            Q = np.hstack([Q, q[:, None]])
        self._ofr = tuple(np.array(chosen).T)
        return self._ofr

    def product_block(self, i):
        """Standardized products (T, N - i) of the unit i with the units j >= i, on the training rows."""
        mean, std = self.product_statistics
        return (self.Z[:, i:i + 1] * self.Z[:, i:] - mean[i, i:]) / std[i, i:]

    @cached_property
    def product_spectrum(self):
        """(eigenvalues (T,), eigenvectors (T, T)) of the kernel of the training rows for the matrix of z and all the
        standardized products."""
        kernel = self.Z @ self.Z.T
        for i in range(self.Z.shape[1]):
            S = self.product_block(i)
            kernel += S @ S.T
        mu, U = np.linalg.eigh(kernel)
        return np.maximum(mu, 0), U

    def leverage_scores(self, size):
        """Ridge leverage score (N, N) of each standardized product z_i z_j in the matrix of z and all the
        standardized products of the training rows, the ridge coefficient such that the effective dimension of this
        matrix is min(size, T / 2)."""
        (T, n), block, (mu, U) = self.Z.shape, self.product_block, self.product_spectrum
        target, low, high = min(size, T / 2), 1e-12 * mu.max(), mu.max() * T
        for _ in range(200):  # (bisection on the logarithm: the effective dimension decreases with the coefficient)
            ridge = np.sqrt(low * high)
            low, high = (ridge, high) if np.sum(mu / (mu + ridge)) > target else (low, ridge)
        scores = np.zeros((n, n))
        U = U / np.sqrt(mu + ridge)
        for i in range(n):
            scores[i, i:] = ((U.T @ block(i)) ** 2).sum(axis=0)
        return scores

    def replaced(self, kind):
        """(units kept as linear inputs, (i, j) units of the products replacing the others) of a kind of
        REPLACED_KINDS (the parity of the papers: the indices of the units of a random reservoir are arbitrary)."""
        n = self.Z.shape[1]
        if kind == "half_squares":
            j = np.arange(1, n, 2)
            pairs = (j, j)
        elif kind == "neighbors_t2":
            j = np.arange(2, n, 2)
            pairs = (j - 1, j - 2)
        else:
            j = np.arange(1, n - 1, 2)
            pairs = (j - 1, j + 1)
        return np.setdiff1d(np.arange(n), j), pairs

    def unit_pairs(self, method):
        """(i, j) units of the products z_i z_j of a method of UNIT_KINDS, PAIR_KINDS or REPLACED_KINDS."""
        kind, n = method.partition(":")[0], self.Z.shape[1]
        if kind == "squares":
            return np.arange(n), np.arange(n)
        if kind in REPLACED_KINDS:
            return self.replaced(kind)[1]
        size = self.size_of(method)
        if kind == "ofr":
            return self.ofr_pairs(size)
        if kind in UNIT_KINDS:
            if kind == "random":
                units = self.rng(method).choice(n, size, replace=False)
            else:
                order = np.argsort(self.weight_norms)
                units = order[:size] if kind == "weights_low" else order[::-1][:size]  # ("weights_high")
            a, b = np.triu_indices(size)
            units = np.sort(units)
            return units[a], units[b]
        i, j = np.triu_indices(n)
        if kind == "residual":
            chosen = np.argpartition(-self.residual_scores[i, j], size - 1)[:size]
        elif kind == "random_pairs":
            chosen = self.rng(method).choice(len(i), size, replace=False)
        elif kind == "leverage":
            scores = self.leverage_scores(size)[i, j]
            chosen = self.rng(method).choice(len(i), size, replace=False, p=scores / scores.sum())
        else:
            raise ValueError(f"Unknown method '{method}'")
        return i[chosen], j[chosen]

    def components(self, method):
        """(components of the training rows (T, c), function of the states giving their components, (a, b) indices of
        the components of each product) of a method."""
        kind, _, size = method.partition(":")
        if kind in UNIT_KINDS + PAIR_KINDS + REPLACED_KINDS:
            return self.Z, self.standardized, self.unit_pairs(method)
        if kind == "maclaurin":
            m, n = self.size_of(method), self.Z.shape[1]
            W = self.rng(method).choice([-1.0, 1.0], (n, 2 * m)) / np.sqrt(n)
            return self.Z @ W, lambda X: self.standardized(X) @ W, (np.arange(m), m + np.arange(m))
        if kind == "scores":
            return (self.cross_scores, lambda X: self.linear.run(self.standardized(X)),
                    np.triu_indices(self.Y.shape[1]))
        if kind == "pca":
            axes = self.point_axes[:, :int(size)]
            return self.Z @ axes, lambda X: self.standardized(X) @ axes, np.triu_indices(int(size))
        if kind == "pca_frames":
            mean, std, axes = self.frame_statistics
            axes = axes[:, :int(size)]

            def project(X):
                return ((X - mean) / std) @ axes
            return project(self.X), project, np.triu_indices(int(size))
        raise ValueError(f"Unknown method '{method}'")

    def maps(self, method):
        """(features of the training rows (T, M), function of the states giving them) of a method of MAP_KINDS."""
        kind, (T, n), rng = method.partition(":")[0], self.Z.shape, self.rng(method)
        size = self.size_of(method)
        if kind == "r2sp":
            W, b = rng.normal(size=(n, size)) / np.sqrt(n), rng.normal(size=size)
            return np.tanh(self.Z @ W + b), lambda X: np.tanh(self.standardized(X) @ W + b)
        landmarks = self.Z[rng.choice(T, min(size, T), replace=False)]

        def kernel(A):  # (the kernel of the products of two units, with the landmarks)
            return 0.5 * ((A @ landmarks.T) ** 2 + (A ** 2) @ (landmarks ** 2).T)
        values, vectors = np.linalg.eigh(kernel(landmarks))
        kept = values > 1e-10 * values.max()
        projection = vectors[:, kept] / np.sqrt(values[kept])
        return kernel(self.Z) @ projection, lambda X: kernel(self.standardized(X)) @ projection

    def features(self, method):
        """(inputs of the readout of the training rows (T, N + extra inputs): z (REPLACED_KINDS: its units kept) and
        the standardized products or features of the method, function of the states giving these inputs)."""
        kind = method.partition(":")[0]
        kept = self.replaced(kind)[0] if kind in REPLACED_KINDS else slice(None)
        if kind in MAP_KINDS:
            extra, extra_of = self.maps(method)
        else:
            train, components, (a, b) = self.components(method)
            extra = train[:, a] * train[:, b]

            def extra_of(X):
                c = components(X)
                return c[:, a] * c[:, b]
        mean, std = extra.mean(axis=0), extra.std(axis=0)
        std[std < 1e-8] = np.inf  # (a constant product: no input)

        def inputs(X):
            return np.hstack([self.standardized(X)[:, kept], (extra_of(X) - mean) / std])
        return np.hstack([self.Z[:, kept], (extra - mean) / std]), inputs

    def readout(self, method):
        """(readout of the method fitted on the training rows, to solve with set_ridge; function of the states giving
        its inputs; number of inputs)."""
        n = self.Z.shape[1]
        if method == "full":
            return GeneralizedRidge(order=2).fit(self.X, self.Y), lambda X: X, n + n * (n + 1) // 2
        if method == "full_std":
            return GeneralizedRidge(order=2).fit(self.Z, self.Y), self.standardized, n + n * (n + 1) // 2
        if method == "linear":
            return GeneralizedRidge(order=1, solver="primal").fit(self.Z, self.Y), self.standardized, n
        F, inputs = self.features(method)
        return GeneralizedRidge(order=1).fit(F, self.Y), inputs, F.shape[1]

    def diagnostics(self, full_std, methods):
        """[dict(measure, method, size, value)] from the readout "full_std" trained (see the module docstring)."""
        A, n = full_std.dual_coef, self.Z.shape[1]
        importance = np.zeros((n, n))
        for k in range(A.shape[1]):  # (weight of z_i z_j for the output k: sum over the rows t of A[t, k] z_ti z_tj)
            importance += ((self.Z * A[:, k:k + 1]).T @ self.Z) ** 2
        importance *= self.pair_variance
        values = np.sort(importance[np.triu_indices(n)])[::-1]
        total = values.sum()
        rows = [dict(measure="top_pairs", method="", size=m, value=values[:m].sum() / total) for m in TOP_PAIRS]
        for method in methods:
            if method.partition(":")[0] in UNIT_KINDS + PAIR_KINDS:
                i, j = self.unit_pairs(method)
                rows.append(dict(measure="share", method=method, size=len(i), value=importance[i, j].sum() / total))
        mu = full_std.eigenvalues
        rows.append(dict(measure="effective_parameters", method="full_std", size=len(values),
                         value=float(np.sum(mu / (mu + full_std.ridge)))))
        return rows


def reservoir_data(reservoirs, songs, waves, classes, frame_songs):
    """({wave: last state of the window of each point (points, N)}, {wave: one-hot next labels}, moments (number, sum,
    Gram matrix) of the states of all the frames of the windows of the songs frame_songs)."""
    X, Y, frame_songs = {}, {}, set(frame_songs)
    n, total, gram = 0, 0.0, 0.0
    for w, windows in point_windows(reservoirs, songs, waves):
        X[w] = np.array([x[-1] for x in windows], dtype=np.float64)
        Y[w] = (songs[w]["next"][:, None] == classes[None, :]).astype(np.float64)
        if w in frame_songs:
            frames = np.vstack(windows).astype(np.float64)
            n, total, gram = n + len(frames), total + frames.sum(axis=0), gram + frames.T @ frames
    return X, Y, (n, total, gram)


def stacked(X, Y, waves):
    """(states, labels, index of the song of each row) of the songs waves."""
    return (np.vstack([X[w] for w in waves]), np.vstack([Y[w] for w in waves]),
            np.concatenate([np.full(len(X[w]), i) for i, w in enumerate(waves)]))


def correct_counts(selection, methods, X_val, true, classes):
    """{method: correct predictions on the rows X_val for each coefficient of ridges_of(method)}."""
    counts = {}
    for method in methods:
        readout, inputs, _ = selection.readout(method)
        F = inputs(X_val)
        counts[method] = np.array([np.sum(classes[readout.set_ridge(ridge).run(F).argmax(axis=1)] == true)
                                   for ridge in ridges_of(method)])
    return counts


def fold_counts(data, fit, val, true, classes, rho_linear, methods, fold):
    X, Y, moments = data
    selection = Selection(*stacked(X, Y, fit), moments, rho_linear, seed=CV_SEED + fold)
    return correct_counts(selection, methods, np.vstack([X[w] for w in val]), true, classes)


def choose(model, params, size, songs, train_songs, frame, classes, methods):
    """({method: (ridge coefficient, out-of-fold accuracy)}, out-of-fold accuracy of each method and coefficient,
    coefficient of the linear readout of the selections), in the 3-fold cross-validation by song of the studies."""
    folds = []
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        data = reservoir_data(reservoirs, songs, fit + val, classes, fit)
        folds.append((data, fit, val, np.concatenate([songs[w]["next"] for w in val])))
    total = sum(len(true) for *_, true in folds)
    linear = sum(fold_counts(data, fit, val, true, classes, 1.0, ["linear"], k)["linear"]
                 for k, (data, fit, val, true) in enumerate(folds))
    rho_linear = float(RIDGES_STD[linear.argmax()])
    counts = Parallel(n_jobs=min(JOBS, len(folds)))(
        delayed(fold_counts)(data, fit, val, true, classes, rho_linear, methods, k)
        for k, (data, fit, val, true) in enumerate(folds))
    correct = {method: sum(count[method] for count in counts) for method in methods}
    chosen = {method: (float(ridges_of(method)[c.argmax()]), c.max() / total) for method, c in correct.items()}
    curves = pd.DataFrame([dict(method=method, ridge=ridge, cv_accuracy=count / total)
                           for method, c in correct.items() for ridge, count in zip(ridges_of(method), c)])
    return chosen, curves, rho_linear


def seed_results(data, train_songs, test_songs, true, classes, rho_linear, chosen, seed, diagnose):
    """([dict(method, inputs, metrics on the test songs)], diagnostics) of the methods trained on the train songs."""
    X, Y, moments = data
    selection = Selection(*stacked(X, Y, train_songs), moments, rho_linear, seed=seed)
    X_test = np.vstack([X[w] for w in test_songs])
    rows, diagnostics = [], []
    for method, (ridge, _) in chosen.items():
        readout, inputs, n_inputs = selection.readout(method)
        scores = readout.set_ridge(ridge).run(inputs(X_test))
        rows.append(dict(method=method, inputs=n_inputs, **metrics(scores, true, classes)))
        if diagnose and method == "full_std" and hasattr(readout, "dual_coef"):
            diagnostics = selection.diagnostics(readout, list(chosen))
    return rows, diagnostics


def run(todo, nb_seeds, methods=METHODS):
    """(results of the methods for each condition and seed, their cross-validation, the diagnostics of seed 0)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    results, curves, diagnostics = [], [], []
    for condition, model, params, _, size in todo:
        keys = dict(condition=condition, network_size=size)
        chosen, curve, rho_linear = choose(model, params, size, songs, train_songs, frame, classes, methods)
        curves.append(curve.assign(**keys))
        print(f"{condition} ({size}): linear readout of the selections {rho_linear:.0e}", flush=True)
        for method, (ridge, accuracy) in chosen.items():
            print(f"  {method}: ridge {ridge:.0e}, cross-validation accuracy {100 * accuracy:.2f} %", flush=True)
        true = np.concatenate([songs[w]["next"] for w in test_songs])
        seed_data = []
        for seed in range(nb_seeds):
            np.random.seed(seed)
            reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
            seed_data.append(reservoir_data(reservoirs, songs, train_songs + test_songs, classes, train_songs))
        outputs = Parallel(n_jobs=min(JOBS, nb_seeds))(
            delayed(seed_results)(seed_data[seed], train_songs, test_songs, true, classes, rho_linear, chosen, seed,
                                  seed == 0)
            for seed in range(nb_seeds))
        for seed, (rows, diagnostic) in enumerate(outputs):
            results += [dict(**keys, seed=seed, **row, ridge=chosen[row["method"]][0],
                             cv_accuracy=chosen[row["method"]][1]) for row in rows]
            diagnostics += [dict(**keys, seed=seed, **row) for row in diagnostic]
            print(f"  seed {seed}: " + ", ".join(f"{row['method']} {100 * row['accuracy']:.1f} %" for row in rows),
                  flush=True)
    return pd.DataFrame(results), pd.concat(curves, ignore_index=True), pd.DataFrame(diagnostics)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "list":
        for condition, _, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            if size and "+" not in condition and not condition.endswith("_proj"):
                print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size]) if c[0] == condition and c[4] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        PARTS.mkdir(parents=True, exist_ok=True)
        for table, suffix in zip(run(todo, nb_seeds), ("", "_cv", "_diagnostics")):
            table.to_csv(PARTS / f"{condition}_{size}{suffix}.csv", index=False)
    elif command == "merge":
        for suffix in ("", "_cv", "_diagnostics"):
            parts = sorted(part for part in PARTS.glob("*.csv")
                           if part.stem.endswith(suffix) and (suffix or not part.stem.endswith(("_cv", "_diagnostics"))))
            if parts:
                pd.concat([pd.read_csv(part) for part in parts], ignore_index=True).to_csv(
                    OUTPUTS / f"{RESULTS}{suffix}.csv", index=False)
                print(f"{len(parts)} conditions: {OUTPUTS / f'{RESULTS}{suffix}.csv'}")
    else:
        raise ValueError(f"Unknown command {command!r}: run, merge or list")
