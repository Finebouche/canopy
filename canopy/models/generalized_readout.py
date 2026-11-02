"""
Generalized readout of Ohkubo & Inubushi (2024), "Reservoir computing with generalized readout based on generalized
synchronization", Sci. Rep. 14, 30918 (https://doi.org/10.1038/s41598-024-81880-3), as a reservoirpy node.

The output is a polynomial of the reservoir state r: its linear term W r and all its unique monomials of degree 2 to
ORDER (r^T W^Q r for the quadratic-form readout, order 2, plus the cubic form for order 3), the higher-order terms of
the Taylor expansion of the map h of the generalized synchronization that the readout approximates (y_t = h(r_t)):

    y = W r + r^T W^Q r (+ sum W^C_ijk r_i r_j r_k) + b

It stays linear in the weights: a ridge regression on the monomials (those of reservoirpy.nodes.NVAR, without delay),
solved in one of two equivalent ways (the same solution for a positive ridge):
  - "primal": on the monomials themselves, when there are fewer of them than samples (small reservoirs, as in the
    paper: N = 10 neurons, or the MFCC);
  - "dual": kernel ridge regression, the inner products of the monomials of two states being computed without the
    monomials (for N = 1000 neurons, 500 500 quadratic monomials): sum over the degrees d of h_d(a * b), the complete
    homogeneous symmetric polynomials of the products of their coordinates, by Newton's identities from the power
    sums p_k = (a^k) . (b^k). The bias is the unpenalized constant term of the primal one (centered kernel).
In both, the spectrum of the centered problem is kept: set_ridge solves it again for another ridge coefficient without
recomputing it (choice of the coefficient by cross-validation).

Example (states of a reservoir, one-hot targets):
    >>> readout = GeneralizedRidge(order=2, ridge=1e-4).fit(states, targets)
    >>> scores = readout.run(test_states)
"""
import itertools as it
from math import comb
from typing import Iterable, Optional

import numpy as np
from reservoirpy.node import ParallelNode

BLOCK = 2048  # rows of the kernel computed at once (memory)


def monomials(x, order):
    """Unique monomials of degree 1 to order of the rows of x (n, N): (n, sum of comb(N + d - 1, d))."""
    features = [x]
    for degree in range(2, order + 1):
        index = np.array(list(it.combinations_with_replacement(range(x.shape[1]), degree)))
        features.append(np.prod(x[:, index], axis=2))
    return np.hstack(features)


def monomial_kernel(a, b, order):
    """Inner products (len(a), len(b)) of the unique monomials of degree 1 to order of the rows of a and b."""
    kernel = np.empty((len(a), len(b)))
    powers_b = [b ** k for k in range(1, order + 1)]
    for i in range(0, len(a), BLOCK):
        block = a[i:i + BLOCK]
        p = [None] + [(block ** k) @ powers_b[k - 1].T for k in range(1, order + 1)]  # power sums of a_i * b_j
        h = [1.0]  # complete homogeneous symmetric polynomials: h_d = sum_k p_k h_(d-k) / d
        for degree in range(1, order + 1):
            h.append(sum(p[k] * h[degree - k] for k in range(1, degree + 1)) / degree)
        kernel[i:i + BLOCK] = sum(h[1:])
    return kernel


class GeneralizedRidge(ParallelNode):
    """Ridge readout on the unique monomials of degree 1 to ``order`` of its inputs (see the module docstring).

    Parameters
    ----------
    order : int, default to 2
        Highest degree of the monomials (1: the linear ridge readout, 2: quadratic form, 3: cubic form).
    ridge : float, default to 0.0
        L2 regularization parameter, the same for all the monomials (as in the paper).
    fit_bias : bool, default to True
        If True, an unpenalized bias is learned.
    solver : {"auto", "primal", "dual"}, default to "auto"
        "auto": "primal" if there are no more monomials than samples, else "dual".
    input_dim, output_dim : int, optional
        Inferred at the first call.
    name : str, optional
    """

    def __init__(self, order: int = 2, ridge: float = 0.0, fit_bias: bool = True, solver: str = "auto",
                 input_dim: Optional[int] = None, output_dim: Optional[int] = None, name: Optional[str] = None):
        if solver not in ("auto", "primal", "dual"):
            raise ValueError(f"Unknown solver '{solver}': must be 'auto', 'primal' or 'dual'.")
        self.order = order
        self.ridge = ridge
        self.fit_bias = fit_bias
        self.solver = solver
        self.name = name
        self.state = {}
        self.input_dim = input_dim
        self.output_dim = output_dim

    def initialize(self, x, y=None):
        self._set_input_dim(x)
        self._set_output_dim(y)
        self.n_monomials = sum(comb(self.input_dim + d - 1, d) for d in range(1, self.order + 1))
        self.state = {"out": np.zeros((self.output_dim,))}
        self.initialized = True

    def _outputs(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.fitted_solver == "primal":
            return monomials(x, self.order) @ self.Wout + self.bias
        kernel = monomial_kernel(x, self.samples, self.order)
        if self.fit_bias:  # (centered as the kernel of the training samples)
            kernel += self.kernel_mean - kernel.mean(axis=1, keepdims=True) - self.kernel_column_means
        return kernel @ self.dual_coef + self.bias

    def _step(self, state, x):
        return {"out": self._outputs(x[None])[0]}

    def _run(self, state, x):
        out = self._outputs(x)
        return {"out": out[-1]}, out

    def worker(self, x, y):
        return x, y

    def master(self, generator: Iterable):
        xs, ys = zip(*generator)
        X, Y = np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64)
        self.y_mean = Y.mean(axis=0) if self.fit_bias else np.zeros(Y.shape[1])
        Yc = Y - self.y_mean
        self.fitted_solver = self.solver
        if self.solver == "auto":
            self.fitted_solver = "primal" if self.n_monomials <= len(X) else "dual"
        if self.fitted_solver == "primal":
            F = monomials(X, self.order)
            self.feature_mean = F.mean(axis=0) if self.fit_bias else np.zeros(F.shape[1])
            F -= self.feature_mean
            gram = F.T @ F
            projected = F.T @ Yc
        else:
            self.samples = X
            gram = monomial_kernel(X, X, self.order)
            if self.fit_bias:  # kernel of the centered monomials
                self.kernel_column_means, self.kernel_mean = gram.mean(axis=0), gram.mean()
                gram += self.kernel_mean - self.kernel_column_means[:, None] - self.kernel_column_means[None]
            projected = Yc
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(gram)
        self.eigenvalues = np.maximum(self.eigenvalues, 0)  # (rounding errors of a positive semidefinite matrix)
        self.projected = self.eigenvectors.T @ projected
        self.set_ridge(self.ridge)

    def set_ridge(self, ridge: float) -> "GeneralizedRidge":
        """Solves the trained regression again with this ridge coefficient (from its kept spectrum)."""
        self.ridge = ridge
        # (without ridge, the negligible eigenvalues ignored, as np.linalg.pinv: no division by 0)
        denominator = self.eigenvalues + ridge
        tolerance = self.eigenvalues.max() * np.finfo(float).eps * len(self.eigenvalues)
        inverse = np.divide(1.0, denominator, out=np.zeros_like(denominator), where=denominator > tolerance)
        coef = self.eigenvectors @ (self.projected * inverse[:, None])
        if self.fitted_solver == "primal":
            self.Wout = coef
            self.bias = self.y_mean - self.feature_mean @ coef
        else:
            self.dual_coef = coef
            self.bias = self.y_mean
        return self
