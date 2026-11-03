"""
Online readouts on the monomials of the state up to degree order (the features of the generalized readout of Ohkubo &
Inubushi 2024, canopy.models.generalized_readout: 125 750 of them at 500 neurons for order 2), trained in their dual
form: the weights are a combination sum_j phi(x_j) A[j] of the features of the training rows, the output of a row is
K[i] @ A + b (K the kernel of the training rows: the inner products of their monomials, monomial_kernel) and the update
of the weights by row i, phi(x_i) times a vector, is that of the coefficient A[i]: the same rule as on the monomials
themselves (LMS, EH: canopy.models.readout_rules.lms and eh, dual=True), without computing them. The monomials are
centered (the kernel of the centered monomials, with their mean on the training rows, as the dual GeneralizedRidge),
and the learning rate of each row is normalized by its squared norm (NLMS: the norms of the monomials spread much
more than those of the states, and a single rate diverges on the rows of the largest norms).

Example (standardized states X of all the training rows, the order of the rows at each epoch):
    >>> readout = KernelLMS(X, order=2, learning_rate=0.1)
    >>> for epoch in range(10):
    ...     readout.partial_fit(Y, rng.permutation(len(X)))
    >>> scores = readout.run(X_test)
"""
import jax.numpy as jnp
import numpy as np

from canopy.models.generalized_readout import monomial_kernel
from canopy.models.readout_rules.eh import eh_epoch
from canopy.models.readout_rules.lms import lms_epoch, row_rates


class KernelReadout:
    """The centered kernel of the monomials of the training rows X (n, N) up to degree order, and the dual weights
    (A, bias) of a readout trained on it (by the subclasses)."""

    def __init__(self, X, order):
        self.X, self.order = np.asarray(X, dtype=np.float64), order
        K = monomial_kernel(self.X, self.X, order)
        self.column_means, self.mean = K.mean(axis=0), K.mean()
        K += self.mean - self.column_means[:, None] - self.column_means[None]
        self.K = jnp.asarray(K)
        self.A, self.bias = None, None

    def kernel(self, X):
        """Centered kernel (len(X), n) of the rows X with the training rows."""
        K = monomial_kernel(np.asarray(X, dtype=np.float64), self.X, self.order)
        return K + self.mean - K.mean(axis=1, keepdims=True) - self.column_means[None]

    def run(self, X):
        return self.kernel(X) @ np.asarray(self.A) + np.asarray(self.bias)

    @property
    def Wout(self):  # (the dual weights: finite when the weights are)
        return self.A


class KernelLMS(KernelReadout):
    """LMS on the monomials, normalized (NLMS: the learning rate of row i learning_rate / (||phi(x_i)||^2 + 1))."""

    def __init__(self, X, order, learning_rate):
        super().__init__(X, order)
        self.rates = row_rates(learning_rate, jnp.diag(self.K), True)

    def partial_fit(self, Y, order):
        """Trained further on the rows order of the training rows, their targets Y."""
        if self.A is None:
            self.A, self.bias = jnp.zeros((len(self.X), Y.shape[1])), jnp.zeros(Y.shape[1])
        self.A, self.bias = lms_epoch(self.A, self.bias, self.K, jnp.asarray(Y), jnp.asarray(order), self.rates,
                                      dual=True)
        return self


class KernelEH(KernelReadout):
    """EH on the monomials (the options of canopy.models.readout_rules.eh.EH), normalized as KernelLMS; the exploration
    noise drawn as EH (with the same generator, the same outputs as EH on the monomials themselves)."""

    def __init__(self, X, order, learning_rate=1e-3, decay=None, noise=0.5, baseline="filtered", alpha=0.2,
                 reward="per_output", modulation="binary", fit_bias=True, seed=None):
        super().__init__(X, order)
        self.rates = row_rates(learning_rate, jnp.diag(self.K), True)
        self.decay, self.noise, self.alpha = decay, noise, alpha
        self.baseline, self.reward, self.modulation, self.fit_bias = baseline, reward, modulation, fit_bias
        self.rng = np.random.default_rng(seed)
        self.carry = None

    def partial_fit(self, Y, order):
        """Trained further on the rows order of the training rows, their targets Y."""
        n_out = Y.shape[1]
        if self.carry is None:
            self.carry = (jnp.zeros((len(self.X), n_out)), jnp.zeros(n_out), jnp.zeros(n_out), jnp.zeros(n_out), 0)
        noise = self.rng.uniform(-self.noise, self.noise, (len(order), n_out))
        self.carry = eh_epoch(self.carry, self.K, jnp.asarray(Y), jnp.asarray(order), jnp.asarray(noise), self.rates,
                              self.alpha, baseline=self.baseline, reward=self.reward, modulation=self.modulation,
                              fit_bias=self.fit_bias, decay=self.decay, dual=True)
        self.A, self.bias = self.carry[0], self.carry[1]
        return self
