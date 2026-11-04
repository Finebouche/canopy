import numpy as np  # (before torch, as in canopy.syllable_generation.wavegan)
import torch

from canopy.syllable_generation.syllables import SLICE_LENGTH, padded, syllable_bounds
from canopy.syllable_generation.wavegan import (BATCH_SIZE, LATENT_DIM, Critic, Generator, PhaseShuffle,
                                                gradient_penalty, latent)


def test_shapes():
    generator, critic = Generator(), Critic()
    x = generator(latent(4, "cpu"))
    assert x.shape == (4, 1, SLICE_LENGTH) and x.abs().max() <= 1
    assert critic(x).shape == (4,)


def test_phase_shuffle_is_wavegan_s():
    """The shift of WaveGAN (TensorFlow): the sequence padded by reflection, then cut at the shift."""
    x, shuffle, shifts = torch.arange(10.0).view(1, 1, 10), PhaseShuffle(2), set()
    for seed in range(20):
        torch.manual_seed(seed)
        shift = int(torch.randint(-2, 3, ()))  # (the draw of the shuffle with this seed)
        torch.manual_seed(seed)
        expected = np.pad(x.numpy(), [(0, 0), (0, 0), (max(shift, 0), max(-shift, 0))], mode="reflect")
        np.testing.assert_array_equal(shuffle(x).numpy(), expected[..., max(-shift, 0):max(-shift, 0) + 10])
        shifts.add(shift)
    assert shifts == {-2, -1, 0, 1, 2}


def test_gradient_penalty_of_a_linear_critic():
    """A linear critic w.x has gradient w everywhere: the penalty is (||w|| - 1)^2."""
    w = torch.randn(1, 1, 8)
    penalty = gradient_penalty(lambda x: (x * w).sum(dim=(1, 2)), torch.randn(5, 1, 8), torch.randn(5, 1, 8))
    torch.testing.assert_close(penalty, (w.norm() - 1) ** 2)


def test_syllable_bounds():
    power = np.zeros(1000)
    power[100:200], power[203:300], power[600:604] = 1.0, 1.0, 1.0  # (a gap of 3 samples, a segment of 4)
    onsets, offsets = syllable_bounds(power, rate=1000)  # (gaps of 3 ms merged: not > MIN_SILENCE; 4 ms removed)
    assert list(onsets) == [100] and list(offsets) == [300]
    assert padded([np.ones(3), np.ones(20)], length=10).sum(axis=1).tolist() == [3, 10]


def test_training_step_runs():
    generator, critic = Generator(dim=8), Critic(dim=8)
    real = torch.rand(BATCH_SIZE // 8, 1, SLICE_LENGTH) * 2 - 1
    loss = -(critic(real).mean() - critic(generator(latent(len(real), "cpu")).detach()).mean())
    loss = loss + 10 * gradient_penalty(critic, real, generator(latent(len(real), "cpu")).detach())
    loss.backward()
    assert all(p.grad is not None for p in critic.parameters())
    assert LATENT_DIM == 3


def test_auditory_activation_and_hebbian_learning():
    from canopy.syllable_generation.sensorimotor import HALF_SECOND, auditory_activation, learn
    classes = ["B", "A", "X"]
    raw = np.zeros((1, HALF_SECOND + 10, 3))
    raw[0, 5] = [0.5, 0.005, 0.2]  # (the frame of the largest output: A below the floor)
    raw[0, HALF_SECOND + 5] = [3.0, 3.0, 3.0]  # (after the first half second: ignored)
    np.testing.assert_allclose(auditory_activation(raw, classes, labels=["A", "B"]), [[0.0, 0.5]])
    z, A = np.array([[1.0, -1.0]]), np.array([[0.5, 1.0]])
    W = learn(z, A, eta=0.1, trials=3)
    np.testing.assert_allclose(W[-1] - W[0], 3 * 0.1 * np.outer(z[0], A[0]))
