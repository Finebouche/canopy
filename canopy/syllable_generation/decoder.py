"""
Decoder of the syllable type of 1 s samples (the perceptual model of Pagliarini, Leblois & Hinaut 2021, an ESN after
the canary decoder of Trouvain & Hinaut 2020): MFCC of the sample (13 coefficients and their deltas and delta-deltas,
FEATURES, standardized with the means and standard deviations of the training samples), read frame by frame by an echo
state network (UNITS leaky tanh neurons) whose ridge readout is trained to give the type of the sample at every frame
(the whole second, silence included: "notrim"). Outputs: the readout at every frame (raw, Pagliarini et al.'s auditory
activation is computed from it, canopy.syllable_generation.sensorimotor) and the type of the largest mean output over
the frames (predict).

Types: the 16 syllable types and, optionally, a garbage type X (Pagliarini et al.'s "EXT" decoder: samples of the GAN
early in its training and white noise, the sounds that are not canary syllables).

Example:
    >>> decoder = Decoder(seed=0).fit(x, labels)  # x (n, 16384) audio at 16 kHz, labels (n,) type names
    >>> decoder.predict(x_test), decoder.raw(x_test)  # (n,) names, (n, frames, types)
"""
import numpy as np
import librosa
from scipy import sparse

from canopy.syllable_generation.syllables import SAMPLING_RATE

FEATURES = dict(n_mfcc=13, n_fft=512, hop_length=160, fmin=500, fmax=8000, n_mels=40)  # (frames of 10 ms)
UNITS = 1000
SPECTRAL_RADIUS = 0.5
LEAK_RATE = 0.05  # (time constant of ~200 ms at 10 ms per frame)
INPUT_SCALING = 0.5
BIAS_SCALING = 0.1
CONNECTIVITY = 0.1
RIDGES = 10.0 ** np.arange(-4, 6)  # (accuracy flat below 1e2 on the real syllables)
GARBAGE = "X"
BATCH = 1024  # samples whose states are computed at once


def features(x):
    """MFCC, deltas and delta-deltas (n, frames, 39) of the samples x (n, samples) at SAMPLING_RATE."""
    mfcc = librosa.feature.mfcc(y=np.asarray(x, np.float32), sr=SAMPLING_RATE, **FEATURES)
    return np.concatenate([mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)],
                          axis=1).transpose(0, 2, 1)


class Decoder:
    def __init__(self, units=UNITS, seed=0):
        rng = np.random.default_rng(seed)
        W = sparse.random(units, units, CONNECTIVITY, random_state=rng, data_rvs=lambda n: rng.uniform(-1, 1, n))
        W = W.toarray()
        self.W = (W * SPECTRAL_RADIUS / np.max(np.abs(np.linalg.eigvals(W)))).astype(np.float32)
        self.W_in = rng.uniform(-INPUT_SCALING, INPUT_SCALING, (3 * FEATURES["n_mfcc"], units)).astype(np.float32)
        self.bias = rng.uniform(-BIAS_SCALING, BIAS_SCALING, units).astype(np.float32)
        self.mean, self.std, self.W_out, self.classes = None, None, None, None

    def states(self, x):
        """Generator of the states (batch, frames, units) of the batches of BATCH samples of x."""
        for k in range(0, len(x), BATCH):
            u = (features(x[k:k + BATCH]) - self.mean) / self.std
            h, states = np.zeros((len(u), len(self.bias)), np.float32), np.empty(u.shape[:2] + (len(self.bias),),
                                                                                   np.float32)
            drive = u @ self.W_in + self.bias
            for t in range(u.shape[1]):
                h = (1 - LEAK_RATE) * h + LEAK_RATE * np.tanh(drive[:, t] + h @ self.W)
                states[:, t] = h
            yield states

    def fit(self, x, labels, x_val=None, labels_val=None):
        """Trained on the samples x (n, samples) of types labels (names), the ridge coefficient of RIDGES of the best
        accuracy on the samples x_val of types labels_val (without them: the median one)."""
        self.classes = np.array(sorted(set(labels), key=lambda c: (c == GARBAGE, c)))
        sample = features(x[:: max(1, len(x) // 2000)])
        self.mean, self.std = sample.mean(axis=(0, 1)), sample.std(axis=(0, 1)) + 1e-8
        index = {c: i for i, c in enumerate(self.classes)}
        units = len(self.bias) + 1
        XtX, XtY = np.zeros((units, units)), np.zeros((units, len(self.classes)))
        for k, states in zip(range(0, len(x), BATCH), self.states(x)):
            X = np.concatenate([states.reshape(-1, units - 1), np.ones((states.size // (units - 1), 1), np.float32)], 1)
            Y = np.repeat(np.eye(len(self.classes))[[index[c] for c in labels[k:k + BATCH]]], states.shape[1], axis=0)
            XtX += X.T @ X
            XtY += X.T @ Y
        self.solutions = {ridge: np.linalg.solve(XtX + ridge * np.eye(units), XtY) for ridge in RIDGES}
        if x_val is None:
            self.ridge = float(np.median(RIDGES))
        else:
            # (the mean output over the frames is the readout of the mean state)
            mean_states = np.concatenate([states.mean(axis=1) for states in self.states(x_val)])
            self.validation = {r: np.mean(self.classes[(mean_states @ self.solutions[r][:-1] + self.solutions[r][-1])
                                                       .argmax(axis=1)] == labels_val) for r in RIDGES}
            self.ridge = max(self.validation, key=self.validation.get)
        self.W_out = self.solutions[self.ridge]
        return self

    def raw(self, x):
        """Readout (n, frames, types) at every frame of the samples x."""
        return np.concatenate([s @ self.W_out[:-1] + self.W_out[-1] for s in self.states(x)])

    def predict(self, x):
        """Type (n,) of the largest mean output over the frames of each sample."""
        return self.classes[self.raw(x).mean(axis=1).argmax(axis=1)]
