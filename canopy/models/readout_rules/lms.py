"""
Online least mean squares readout (the rule of reservoirpy's LMS, the LMS-based FORCE rule of Sussillo & Abbott 2009)
trained in JAX: the steps of a pass over the rows in a compiled lax.scan, about 6 times faster than reservoirpy's loop
over the steps (500 inputs, 30 outputs), the same weights (float64):

    e = W^T x + b - y,    dW = -eta x e^T,    db = -eta e

normalized (NLMS): the learning rate of each row divided by its squared norm (+ 1, the input of the bias), eta =
learning_rate / (||x||^2 + 1), stable for a learning rate below 2 whatever the norms of the rows.

Example (rows of all the training songs, the order of the rows at each epoch):
    >>> readout = LMS(learning_rate=1e-3)
    >>> for epoch in range(10):
    ...     readout.partial_fit(X, Y, rng.permutation(len(X)))
    >>> scores = readout.run(X_test)
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # float64, to match reservoirpy


@partial(jax.jit, static_argnames=("dual",))
def lms_epoch(W, b, X, Y, order, rates, dual=False):
    """(W, b) after the LMS steps on the rows order of (X, Y), the learning rate of row i rates[i]. dual: the rule in
    its dual form, X the kernel of the training rows (n, n) and W the coefficients of the rows (n, outputs), the
    weights sum_j phi(x_j) W[j] on the features phi of the kernel (canopy.models.readout_rules.kernel): the update
    -eta phi(x_i) e^T of the weights is that of the coefficient of row i."""
    def step(carry, i):
        W, b = carry
        error = X[i] @ W + b - Y[i]
        W = W.at[i].add(-rates[i] * error) if dual else W - rates[i] * jnp.outer(X[i], error)
        return (W, b - rates[i] * error), None
    return jax.lax.scan(step, (W, b), order)[0]


def row_rates(learning_rate, squared_norms, normalized):
    """Learning rate of each row: learning_rate, or learning_rate / (squared norm + 1) (normalized)."""
    squared_norms = jnp.asarray(squared_norms)
    return learning_rate / (squared_norms + 1) if normalized else jnp.full(squared_norms.shape, learning_rate)


class LMS:
    """LMS readout, initial weights and bias zero (reservoirpy's LMS with its defaults); normalized: NLMS."""

    def __init__(self, learning_rate, normalized=False):
        self.learning_rate, self.normalized = learning_rate, normalized
        self.Wout, self.bias, self.rates = None, None, None

    def partial_fit(self, X, Y, order):
        """Trained further on the rows order of (X, Y) (arrays, JAX ones to avoid a copy at each call)."""
        if self.Wout is None:
            self.Wout, self.bias = jnp.zeros((X.shape[1], Y.shape[1])), jnp.zeros(Y.shape[1])
            self.rates = row_rates(self.learning_rate, jnp.sum(jnp.asarray(X) ** 2, axis=1), self.normalized)
        self.Wout, self.bias = lms_epoch(self.Wout, self.bias, X, Y, jnp.asarray(order), self.rates)
        return self

    def run(self, X):
        return np.asarray(X @ self.Wout + self.bias)
