"""
Readout learned by reward-modulated Hebbian learning, the "exploratory Hebbian" (EH) rule of Hoerzer, Legenstein &
Maass (2014), "Emergence of complex computational structures from chaotic neural networks through reward-modulated
Hebbian learning", Cereb. Cortex 24(3), 677-690 (https://doi.org/10.1093/cercor/bhs348), as a reservoirpy node.

No teacher: the target only gives a scalar reward. During training, the output explores with a noise xi, and its
weights follow the Hebbian product of their input and of the exploratory part of the output, gated by a modulatory
signal (dopamine-like) telling whether the performance is better than usual:

    z = W^T x + b + xi,                                xi uniform in [-noise, noise]
    P = -(z - y)^2                                     performance (reward), per output or summed over them
    dW = eta * x (z - z_bar) M,  db = eta (z - z_bar) M
    M = 1 if P > P_bar else 0                          ("binary" modulation, the paper's; "continuous": M = P - P_bar)

With baseline="filtered" (the paper's), z_bar and P_bar are the low-pass filtered output and performance (z_bar <-
(1 - alpha) z_bar + alpha z, the same for P_bar, updated after the step): the input must change slowly (a time
series), so that z - z_bar is mostly the exploration. With baseline="sample", for independent samples (one state per
prediction point), they are the output without noise and its performance on the same input (z - z_bar = xi, the node
perturbation of Fiete & Seung 2006, the reward compared with the one expected for this input). eta decays as
eta_0 / (1 + t / decay). Run (and partial_fit's returned outputs aside), the output is the deterministic W^T x + b.

Example (states of a reservoir, one-hot targets, several passes):
    >>> readout = EH(learning_rate=1e-3, noise=0.5, seed=0)
    >>> for epoch in range(10):
    ...     readout.partial_fit(states, targets)
    >>> scores = readout.run(test_states)
"""
from typing import Callable, Literal, Optional, Union

import numpy as np
from reservoirpy.mat_gen import zeros
from reservoirpy.node import OnlineNode
from reservoirpy.type import NodeInput, State, Timeseries, Timestep, Weights, is_array


class EH(OnlineNode):
    """Readout learned by the exploratory Hebbian rule (see the module docstring).

    Parameters
    ----------
    learning_rate : float, default 1e-3
        Initial learning rate eta_0.
    decay : float, optional
        Time constant (learning steps) of the decay of the learning rate, eta_0 / (1 + t / decay); None: constant.
    noise : float, default 0.5
        Amplitude of the exploration noise (uniform in [-noise, noise]) added to each output during training.
    baseline : {"filtered", "sample"}, default "filtered"
        z_bar and P_bar: low-pass filtered output and performance (the paper's, for time series), or the output
        without noise and its performance on the same input (independent samples).
    alpha : float, default 0.2
        Coefficient of the low-pass filters of the output and of the performance (1: no memory), baseline "filtered".
    reward : {"per_output", "global"}, default "per_output"
        Performance of each output (-(z_k - y_k)^2, a modulatory signal per output), or a single one for all of them
        (-||z - y||^2, one global signal).
    modulation : {"binary", "continuous"}, default "binary"
        M = 1 if P > P_bar else 0 (the paper's), or M = P - P_bar.
    Wout, bias : callable or array, default zeros
        Initial weights (input_dim, output_dim) and bias (output_dim,).
    fit_bias : bool, default True
        Learn the bias too.
    seed : int or numpy Generator, optional
        Generator of the exploration noise.
    """

    #: Learned output weights.
    Wout: Weights
    #: Learned bias.
    bias: Weights

    def __init__(
        self,
        learning_rate: float = 1e-3,
        decay: Optional[float] = None,
        noise: float = 0.5,
        baseline: Literal["filtered", "sample"] = "filtered",
        alpha: float = 0.2,
        reward: Literal["per_output", "global"] = "per_output",
        modulation: Literal["binary", "continuous"] = "binary",
        Wout: Union[Weights, Callable] = zeros,
        bias: Union[Weights, Callable] = zeros,
        fit_bias: bool = True,
        seed: Optional[Union[int, np.random.Generator]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        if reward not in ("per_output", "global"):
            raise ValueError(f"Unknown reward '{reward}': must be 'per_output' or 'global'.")
        if baseline not in ("filtered", "sample"):
            raise ValueError(f"Unknown baseline '{baseline}': must be 'filtered' or 'sample'.")
        if modulation not in ("binary", "continuous"):
            raise ValueError(f"Unknown modulation '{modulation}': must be 'binary' or 'continuous'.")
        self.learning_rate = learning_rate
        self.decay = decay
        self.noise = noise
        self.baseline = baseline
        self.alpha = alpha
        self.reward = reward
        self.modulation = modulation
        self.Wout = Wout
        self.bias = bias
        self.fit_bias = fit_bias
        self.rng = np.random.default_rng(seed)
        self.state = {}
        self.name = name
        self.input_dim = Wout.shape[0] if is_array(Wout) else input_dim
        self.output_dim = Wout.shape[1] if is_array(Wout) else output_dim

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        out = x @ self.Wout + self.bias
        return {"out": out[-1]}, out

    def _step(self, state: State, x: Timestep) -> State:
        return {"out": x @ self.Wout + self.bias}

    def initialize(self, x: Union[NodeInput, Timestep], y: Optional[Union[NodeInput, Timestep]] = None):
        self._set_input_dim(x)
        self._set_output_dim(y)
        if isinstance(self.Wout, Callable):
            self.Wout = self.Wout(self.input_dim, self.output_dim)
        if isinstance(self.bias, Callable):
            self.bias = self.bias(self.output_dim)
        self.state = {"out": np.zeros((self.output_dim,))}
        # low-pass filtered output and performance (None: set by the first step), learning steps done
        self.z_bar, self.P_bar, self.steps = None, None, 0
        self.initialized = True

    def _learning_step(self, x: Timestep, y: Optional[Timestep]) -> Timestep:
        if y is None:
            raise ValueError("EH needs the targets (they give the reward).")
        def performance(out):
            return -(out - y) ** 2 if self.reward == "per_output" else np.full(self.output_dim, -np.sum((out - y) ** 2))
        z = x @ self.Wout + self.bias + self.rng.uniform(-self.noise, self.noise, self.output_dim)
        P = performance(z)
        if self.baseline == "sample":
            self.z_bar = x @ self.Wout + self.bias
            self.P_bar = performance(self.z_bar)
        elif self.z_bar is None:
            self.z_bar, self.P_bar = z.copy(), P.copy()
        M = (P > self.P_bar).astype(float) if self.modulation == "binary" else P - self.P_bar
        eta = self.learning_rate / (1 + self.steps / self.decay) if self.decay else self.learning_rate
        exploration = eta * (z - self.z_bar) * M  # (output_dim,)
        self.Wout = self.Wout + np.outer(x, exploration)
        if self.fit_bias:
            self.bias = self.bias + exploration
        if self.baseline == "filtered":
            self.z_bar = (1 - self.alpha) * self.z_bar + self.alpha * z
            self.P_bar = (1 - self.alpha) * self.P_bar + self.alpha * P
        self.steps += 1
        return z
