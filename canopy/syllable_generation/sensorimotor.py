"""
Sensorimotor model of Pagliarini, Leblois & Hinaut (2021), "Canary vocal sensorimotor model with RNN decoder and
low-dimensional GAN generator" (ICDL; code: https://github.com/spagliarini/canary-vocal-sensorimotor-model): an
inverse model, from the perceptual space (the 16 syllable types) to the motor space (the latent space of the GAN,
canopy.syllable_generation.wavegan), learned by a Hebbian rule from a random babbling:
  - motor exploration: latent vectors z ~ U(-1, 1)^3, their sounds G(z) heard by the decoder
    (canopy.syllable_generation.decoder);
  - auditory activation of a sound (their "max" option): the outputs of the decoder at the frame of its largest
    output in the first HALF_SECOND frames, below FLOOR set to 0 and above 1 to 1, each type divided by the 95th
    percentile of its activation over the exploration (scaling) and capped at 1: A (16,);
  - Hebbian rule: W (3, 16), initialized in U(-W_SEED, W_SEED), W += eta z A^T at each trial (a random sound of the
    exploration);
  - production of type s: the GAN at the motor command clip(W[:, s], -1, 1) (their piecewise linear activation), its
    score the auditory activation of type s.
Their settings: 3001 trials, eta 0.1 or 0.01, W_SEED 0.001, the decoder with the garbage type ("EXT").

Example:
    >>> z, A, scaling = explore(generator, decoder, 16000)
    >>> W = learn(z, A, eta=0.01, trials=3001)  # (trials + 1, 3, 16)
    >>> audio, scores = produce(W[-1], generator, decoder, scaling)
"""
import numpy as np

from canopy.syllable_generation.syllables import LABELS
from canopy.syllable_generation.wavegan import LATENT_DIM, generate

HALF_SECOND = 50  # frames of the decoder (10 ms)
FLOOR = 0.01
TRIALS = 3001
W_SEED = 0.001


def auditory_activation(raw, classes, labels=LABELS):
    """Activation (n, len(labels)) of the decoder outputs raw (n, frames, len(classes)): the outputs at the frame of
    the largest one in the first HALF_SECOND frames, clipped (FLOOR, 1)."""
    raw = raw[:, :HALF_SECOND]
    frame = raw.max(axis=2).argmax(axis=1)
    peak = raw[np.arange(len(raw)), frame]
    peak = np.where(peak < FLOOR, 0, np.minimum(peak, 1))
    return peak[:, [list(classes).index(c) for c in labels]]


def explore(generator, decoder, n, seed=0):
    """Motor exploration: (z (n, LATENT_DIM), A (n, 16) the scaled auditory activations, scaling (16,) the 95th
    percentiles of the activations of each type)."""
    z = np.random.default_rng(seed).uniform(-1, 1, (n, LATENT_DIM)).astype(np.float32)
    activation = auditory_activation(decoder.raw(generate(generator, z)), decoder.classes)
    scaling = np.maximum(np.percentile(activation, 95, axis=0), 1e-8)
    return z, np.minimum(activation / scaling, 1), scaling


def learn(z, A, eta, trials=TRIALS, seed=0):
    """Weights (trials + 1, LATENT_DIM, 16) along the Hebbian learning (W += eta z A^T) on random sounds of the
    exploration (z, A), from W in U(-W_SEED, W_SEED)."""
    rng = np.random.default_rng(seed)
    W0 = rng.uniform(-W_SEED, W_SEED, (z.shape[1], A.shape[1]))
    trial = rng.choice(len(z), trials)
    return np.concatenate([W0[None], W0 + eta * np.cumsum(z[trial, :, None] * A[trial, None, :], axis=0)])


def produce(W, generator, decoder, scaling):
    """(audio (16, samples), scores (16,)) of the productions of the 16 types by the weights W (LATENT_DIM, 16): the
    GAN at clip(W[:, s], -1, 1), the score its scaled auditory activation of type s."""
    audio = generate(generator, np.clip(W, -1, 1).T)
    activation = np.minimum(auditory_activation(decoder.raw(audio), decoder.classes) / scaling, 1)
    return audio, np.diag(activation)
