"""
Online least mean squares readout (the rule of reservoirpy's LMS, the LMS-based FORCE rule of Sussillo & Abbott 2009)
trained in JAX: the steps of a pass over the rows in a compiled lax.scan, about 6 times faster than reservoirpy's loop
over the steps (500 inputs, 30 outputs), the same weights (float64):

    e = W^T x + b - y,    dW = -eta x e^T,    db = -eta e

Example (rows of all the training songs, the order of the rows at each epoch):
    >>> readout = LMS(learning_rate=1e-3)
    >>> for epoch in range(10):
    ...     readout.partial_fit(X, Y, rng.permutation(len(X)))
    >>> scores = readout.run(X_test)
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # float64, to match reservoirpy


@jax.jit
def lms_epoch(W, b, X, Y, order, learning_rate):
    """(W, b) after the LMS steps on the rows order of (X, Y)."""
    def step(carry, i):
        W, b = carry
        error = X[i] @ W + b - Y[i]
        return (W - learning_rate * jnp.outer(X[i], error), b - learning_rate * error), None
    return jax.lax.scan(step, (W, b), order)[0]


class LMS:
    """LMS readout, initial weights and bias zero (reservoirpy's LMS with its defaults)."""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.Wout, self.bias = None, None

    def partial_fit(self, X, Y, order):
        """Trained further on the rows order of (X, Y) (arrays, JAX ones to avoid a copy at each call)."""
        if self.Wout is None:
            self.Wout, self.bias = jnp.zeros((X.shape[1], Y.shape[1])), jnp.zeros(Y.shape[1])
        self.Wout, self.bias = lms_epoch(self.Wout, self.bias, X, Y, jnp.asarray(order), self.learning_rate)
        return self

    def run(self, X):
        return np.asarray(X @ self.Wout + self.bias)
