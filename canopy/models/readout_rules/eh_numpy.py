"""
Readout learned by reward-modulated Hebbian learning, the "exploratory Hebbian" (EH) rule, as a reservoirpy node (the
reference of the JAX version canopy.models.readout_rules.eh: the same weights).

No teacher: the target only gives a reward. During training, the output explores with a noise xi, and its weights
follow the Hebbian product of their input and of the exploratory part of the output, gated by a modulatory signal
(dopamine-like) telling whether the performance is better than expected:

    z = W^T x + b + xi                       xi uniform in [-noise, noise]; run: the deterministic W^T x + b
    dW = eta x (z - z_bar) M,  db = eta (z - z_bar) M

Four ingredients make the versions, one option each (one method of EH each):

  1. baseline, the estimate of the noise z - z_bar and the expected performance P_bar (_baselines):
       "filtered": z_bar and P_bar low-pass filtered over the steps (z_bar <- (1 - alpha) z_bar + alpha z, the same
                   for P_bar, after the step). z - z_bar is the noise only if the input changes slowly (a time series).
       "sample":   the output without noise and its performance on the same input: z - z_bar = xi exactly (node
                   perturbation, Fiete & Seung 2006), for independent samples.
  2. reward, the performance P (_performance):
       "global":     -||z - y||^2, one signal for all the outputs.
       "per_output": -(z_k - y_k)^2, a signal for each output.
  3. modulation, M (_modulation):
       "binary":     M = 1 if P > P_bar else 0.
       "continuous": M = P - P_bar.
  4. normalized, the learning rate (_learning_rate):
       False: eta = learning_rate (decaying as learning_rate / (1 + t / decay) with decay).
       True:  eta = learning_rate / (||x||^2 + 1), the step of NLMS (canopy.models.readout_rules.lms).

VERSIONS, each one ingredient away from the previous (the rules of canopy.next_phrase_prediction.readout_rules; "np":
node perturbation, not NumPy):

  version        baseline  reward      modulation  normalized
  "legenstein"   filtered  global      continuous  no    Legenstein, Chase, Schwartz & Maass (2010, J. Neurosci.
                                                         30(25), 8400-8410), the origin of the rule
  "eh"           filtered  global      binary      no    Hoerzer, Legenstein & Maass (2014, Cereb. Cortex 24(3),
                                                         677-690): learns slowly changing time series; on
                                                         independent samples it does not learn (about chance level
                                                         on the next phrase)
  "eh_np"        sample    global      binary      no    ours: the noise known as in Fiete & Seung (2006) and the
                                                         basic rule of Burms, Caluwaerts & Dambre (2015, Front.
                                                         Neurorobot. 9:9, eq. 3; their P_bar: the mean of the last 50
                                                         rewards for the same input). A single reward for all the
                                                         outputs does not tell which one improved: it does not learn
  "eh_np_local"  sample    per_output  binary      no    ours: learns (on average over the noise, LMS with the error
                                                         e |e| / noise, saturated at noise / 4 for |e| > noise / 2)
  "neh"          sample    per_output  binary      yes   ours: eh_np_local with the step of NLMS. Closest published
                                                         rule: the "decorrelated" rule of Burms et al. (2015, eq. 4,
                                                         dW = alpha (r - r_bar) Z^T X (X^T X + I)^-1, the same step for
                                                         a trial of a single step, by Sherman-Morrison), presented as
                                                         a decorrelation close to ridge regression

The defaults of EH are those of "eh" (Hoerzer et al.); EH.version(name, ...) builds a version.

Example (states of a reservoir, one-hot targets, several passes):
    >>> readout = EH.version("eh_np_local", learning_rate=1e-3, noise=0.5, seed=0)
    >>> for epoch in range(10):
    ...     readout.partial_fit(states, targets)
    >>> scores = readout.run(test_states)
"""
from typing import Callable, Literal, Optional, Union

import numpy as np
from reservoirpy.mat_gen import zeros
from reservoirpy.node import OnlineNode
from reservoirpy.type import NodeInput, State, Timeseries, Timestep, Weights, is_array

VERSIONS = {
    "legenstein": dict(baseline="filtered", reward="global", modulation="continuous", normalized=False),
    "eh": dict(baseline="filtered", reward="global", modulation="binary", normalized=False),
    "eh_np": dict(baseline="sample", reward="global", modulation="binary", normalized=False),
    "eh_np_local": dict(baseline="sample", reward="per_output", modulation="binary", normalized=False),
    "neh": dict(baseline="sample", reward="per_output", modulation="binary", normalized=True),
}


def check_options(baseline, reward, modulation):
    for name, value, allowed in (("baseline", baseline, ("filtered", "sample")),
                                 ("reward", reward, ("global", "per_output")),
                                 ("modulation", modulation, ("binary", "continuous"))):
        if value not in allowed:
            raise ValueError(f"Unknown {name} '{value}': must be one of {', '.join(map(repr, allowed))}.")


def version_options(name):
    """Options (baseline, reward, modulation, normalized) of the version name of VERSIONS."""
    if name not in VERSIONS:
        raise ValueError(f"Unknown EH version '{name}': must be one of {', '.join(VERSIONS)}.")
    return dict(VERSIONS[name])


class EH(OnlineNode):
    """Readout learned by the exploratory Hebbian rule (see the module docstring; defaults: the version "eh").

    Parameters
    ----------
    learning_rate : float, default 1e-3
        Initial learning rate eta_0.
    decay : float, optional
        Time constant (learning steps) of the decay of the learning rate, eta_0 / (1 + t / decay); None: constant.
    noise : float, default 0.5
        Amplitude of the exploration noise (uniform in [-noise, noise]) added to each output during training.
    baseline : {"filtered", "sample"}, default "filtered"
        Ingredient 1: z_bar and P_bar low-pass filtered, or the output without noise and its performance.
    alpha : float, default 0.2
        Coefficient of the low-pass filters of the output and of the performance (1: no memory), baseline "filtered".
    reward : {"global", "per_output"}, default "global"
        Ingredient 2: a single performance for all the outputs, or one for each.
    modulation : {"binary", "continuous"}, default "binary"
        Ingredient 3: M = 1 if P > P_bar else 0, or M = P - P_bar.
    normalized : bool, default False
        Ingredient 4: the learning rate of each step divided by ||x||^2 + 1 (the step of NLMS).
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
        reward: Literal["global", "per_output"] = "global",
        modulation: Literal["binary", "continuous"] = "binary",
        normalized: bool = False,
        Wout: Union[Weights, Callable] = zeros,
        bias: Union[Weights, Callable] = zeros,
        fit_bias: bool = True,
        seed: Optional[Union[int, np.random.Generator]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        check_options(baseline, reward, modulation)
        self.learning_rate = learning_rate
        self.decay = decay
        self.noise = noise
        self.baseline = baseline
        self.alpha = alpha
        self.reward = reward
        self.modulation = modulation
        self.normalized = normalized
        self.Wout = Wout
        self.bias = bias
        self.fit_bias = fit_bias
        self.rng = np.random.default_rng(seed)
        self.state = {}
        self.name = name
        self.input_dim = Wout.shape[0] if is_array(Wout) else input_dim
        self.output_dim = Wout.shape[1] if is_array(Wout) else output_dim

    @classmethod
    def version(cls, name, **kwargs):
        """EH of the version name of VERSIONS (its baseline, reward, modulation and normalized), with the other
        parameters kwargs."""
        return cls(**version_options(name), **kwargs)

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
        # filtered output and performance (baseline "filtered"; None: set by the first step), learning steps done
        self.z_bar, self.P_bar, self.steps = None, None, 0
        self.initialized = True

    # The four ingredients

    def _performance(self, out, y):
        """2. reward: the performance of each output (-(out_k - y_k)^2), or the global one for all of them."""
        if self.reward == "per_output":
            return -(out - y) ** 2
        return np.full(self.output_dim, -np.sum((out - y) ** 2))

    def _baselines(self, x, y, z, P):
        """1. baseline: (z_bar, P_bar) of this step: the output without noise and its performance ("sample"), or the
        filtered ones of the previous steps ("filtered"; at the first step, z and P)."""
        if self.baseline == "sample":
            z_clean = x @ self.Wout + self.bias
            return z_clean, self._performance(z_clean, y)
        if self.z_bar is None:
            return z.copy(), P.copy()
        return self.z_bar, self.P_bar

    def _modulation(self, P, P_bar):
        """3. modulation: M = 1 if P > P_bar else 0 ("binary"), or P - P_bar ("continuous")."""
        return (P > P_bar).astype(float) if self.modulation == "binary" else P - P_bar

    def _learning_rate(self, x):
        """4. learning rate: learning_rate (decaying with decay), divided by ||x||^2 + 1 if normalized."""
        eta = self.learning_rate / (x @ x + 1) if self.normalized else self.learning_rate
        return eta / (1 + self.steps / self.decay) if self.decay else eta

    def _learning_step(self, x: Timestep, y: Optional[Timestep]) -> Timestep:
        if y is None:
            raise ValueError("EH needs the targets (they give the reward).")
        z = x @ self.Wout + self.bias + self.rng.uniform(-self.noise, self.noise, self.output_dim)
        P = self._performance(z, y)
        self.z_bar, self.P_bar = self._baselines(x, y, z, P)
        exploration = self._learning_rate(x) * (z - self.z_bar) * self._modulation(P, self.P_bar)  # (output_dim,)
        self.Wout = self.Wout + np.outer(x, exploration)
        if self.fit_bias:
            self.bias = self.bias + exploration
        if self.baseline == "filtered":  # (after the step)
            self.z_bar = (1 - self.alpha) * self.z_bar + self.alpha * z
            self.P_bar = (1 - self.alpha) * self.P_bar + self.alpha * P
        self.steps += 1
        return z
