"""
Online recursive least squares readout (the RLS-based FORCE rule of Sussillo & Abbott 2009), with the bias learned as
the weight of a constant input 1: reservoirpy's RLS node on the inputs (x, 1), fit_bias=False.

    P_0 = I / alpha,    P <- P - P x~ x~^T P / (1 + x~^T P x~),    W~ <- W~ - P x~ e^T,    x~ = (x, 1),  e = W~^T x~ - y

After one pass over the rows, its weights are those of the ridge regression of the same rows with the coefficient alpha
(on the weights and on the bias): W~ = (X~^T X~ + alpha I)^-1 X~^T Y, whatever the order of the rows.

Why not the bias of reservoirpy's node (fit_bias=True): it is updated apart from the weights, as a running mean of the
errors (b <- b - e / t), while the weights are updated as if there were no bias. This is not the least squares solution,
and with a small alpha the readout is much worse than the ridge regression (products of the linear scores of random_ee
500 without leak, alpha = 1e-6: 5 % against 79 %; the states alone, alpha = 1e-8: 74 % against 77 %). With the constant
input, RLS gives the accuracy of the ridge regression down to alpha = 1e-6 on these inputs; at 1e-8 the recursion
itself loses its precision (float64, eigenvalues of the inputs down to 1e-14).

Example (the rows of each training song in turn):
    >>> readout = RLS(alpha=1e-4)
    >>> for X, Y in songs:
    ...     readout.partial_fit(X, Y)
    >>> scores = readout.run(X_test)
"""
import numpy as np
from reservoirpy.nodes import RLS as RLSNode


def with_constant(X):
    """Rows (T, N + 1): the inputs X (T, N) and the constant input 1."""
    X = np.asarray(X, dtype=np.float64)
    return np.hstack([X, np.ones((len(X), 1))])


class RLS:
    """RLS readout, initial P = I / alpha; Wout (N, outputs) and bias (outputs,) after the first partial_fit."""

    def __init__(self, alpha=1e-6):
        self.alpha = alpha
        self.node = RLSNode(alpha=alpha, fit_bias=False)

    def partial_fit(self, X, Y):
        """Trained further on the rows X (T, N), their targets Y (T, outputs), in their order."""
        self.node.partial_fit(with_constant(X), np.asarray(Y, dtype=np.float64))
        return self

    @property
    def Wout(self):
        return self.node.Wout[:-1]

    @property
    def bias(self):
        return self.node.Wout[-1]

    def run(self, X):
        """Outputs (T, outputs) of the rows X (T, N)."""
        return with_constant(X) @ self.node.Wout
