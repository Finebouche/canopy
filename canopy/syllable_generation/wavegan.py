"""
Low-dimensional WaveGAN of Pagliarini et al. (2021): the WaveGAN of Donahue, McAuley & Puckette (2019) ("Adversarial
audio synthesis", ICLR) with a latent space of dimension 3, generating SLICE_LENGTH samples (~1 s at 16 kHz) of audio,
trained on single canary syllables (canopy.syllable_generation.syllables). After their TensorFlow code (WaveGAN of
Donahue et al.) and the PyTorch version of N. Trouvain (canarygan), with the settings of WaveGAN:
  - generator: z ~ U(-1, 1)^LATENT_DIM -> dense (16 x 16 DIM, ReLU) -> 5 transposed convolutions (kernel 25, stride 4:
    16 DIM -> 8 DIM -> 4 DIM -> 2 DIM -> DIM -> 1 channel, length 16 -> 16384), ReLU, tanh at the output;
  - critic: 5 convolutions (kernel 25, stride 4: 1 -> DIM -> ... -> 16 DIM channels), LeakyReLU 0.2, phase shuffle
    (the whole batch shifted by a random number of samples in [-PHASE_SHUFFLE, PHASE_SHUFFLE], reflected at the edges)
    after the first 4, dense to a scalar;
  - WGAN-GP loss (Gulrajani et al. 2017, gradient penalty LAMBDA), CRITIC_STEPS updates of the critic per update of the
    generator, each on its own batch of real syllables (as WaveGAN; canarygan reuses the same batch), Adam (LR, BETAS),
    batches of BATCH_SIZE; weights initialized as TensorFlow's layers (Glorot uniform, zero bias).
canarygan differs: no phase shuffle by default (rad 0, WaveGAN's is 2) and He initialization.

An epoch is len(dataset) / BATCH_SIZE updates of the generator (canarygan's, 250 for the 16 000 syllables): the
critic sees each syllable CRITIC_STEPS times per epoch. Checkpoints (generator, critic, optimizers, generator of the
epoch) in <run>/checkpoints every SAVE_EVERY epochs and the last one in <run>/last.pt (training resumed from it), the
losses every LOG_EVERY updates in <run>/losses.csv. Device: CUDA, else CPU (device).

Usage (from the root of canopy):
  python -m canopy.syllable_generation.wavegan train <run> [epochs]       (default EPOCHS; resumed from <run>/last.pt)
  python -m canopy.syllable_generation.wavegan generate <checkpoint> <n> <file.npz> [seed]   x (n, SLICE_LENGTH), z
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np  # (before torch: the reverse order loads two OpenMP runtimes in hag_env)
import torch
from torch import nn
from torch.nn import functional as F

from canopy.syllable_generation.syllables import SLICE_LENGTH, syllable_dataset

LATENT_DIM = 3
DIM = 64
KERNEL, STRIDE = 25, 4
PHASE_SHUFFLE = 2
LAMBDA = 10.0
CRITIC_STEPS = 5
LR, BETAS = 1e-4, (0.5, 0.9)
BATCH_SIZE = 64
EPOCHS = 1000
SAVE_EVERY = 15  # epochs (canarygan's)
LOG_EVERY = 100  # updates of the generator
PADDING = (KERNEL - STRIDE + 1) // 2  # ("same" padding of TensorFlow, stride 4: 11)


def initialized(layer):
    """layer with TensorFlow's initialization (Glorot uniform weights, zero bias)."""
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class PhaseShuffle(nn.Module):
    """Shift of the time axis of the batch (batch, channels, time) by a random number of samples in [-radius, radius],
    reflection at the edges."""

    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def forward(self, x):
        shift = int(torch.randint(-self.radius, self.radius + 1, ()))
        if shift == 0:
            return x
        if shift > 0:
            return F.pad(x[..., :-shift], (shift, 0), mode="reflect")
        return F.pad(x[..., -shift:], (0, -shift), mode="reflect")


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, dim=DIM):
        super().__init__()
        self.dim = dim
        self.dense = initialized(nn.Linear(latent_dim, 16 * 16 * dim))
        channels = [16 * dim, 8 * dim, 4 * dim, 2 * dim, dim, 1]
        self.deconvolutions = nn.ModuleList(
            initialized(nn.ConvTranspose1d(a, b, KERNEL, STRIDE, padding=PADDING, output_padding=1))
            for a, b in zip(channels[:-1], channels[1:]))

    def forward(self, z):
        x = F.relu(self.dense(z)).view(len(z), 16 * self.dim, 16)
        for k, deconvolution in enumerate(self.deconvolutions):
            x = deconvolution(x)
            x = torch.tanh(x) if k == len(self.deconvolutions) - 1 else F.relu(x)
        return x


class Critic(nn.Module):
    def __init__(self, dim=DIM, phase_shuffle=PHASE_SHUFFLE):
        super().__init__()
        channels = [1, dim, 2 * dim, 4 * dim, 8 * dim, 16 * dim]
        self.convolutions = nn.ModuleList(initialized(nn.Conv1d(a, b, KERNEL, STRIDE, padding=PADDING))
                                          for a, b in zip(channels[:-1], channels[1:]))
        self.shuffle = PhaseShuffle(phase_shuffle)
        self.dense = initialized(nn.Linear(16 * dim * SLICE_LENGTH // STRIDE ** 5, 1))

    def forward(self, x):
        for k, convolution in enumerate(self.convolutions):
            x = F.leaky_relu(convolution(x), 0.2)
            if k < len(self.convolutions) - 1 and self.shuffle.radius:
                x = self.shuffle(x)
        return self.dense(x.flatten(1)).squeeze(1)


def latent(n, device, generator=None):
    """n latent vectors z ~ U(-1, 1)^LATENT_DIM."""
    return torch.rand(n, LATENT_DIM, device=device, generator=generator) * 2 - 1


def gradient_penalty(critic, real, fake):
    """Mean of (||grad of the critic at x||_2 - 1)^2 at random interpolations x of real and fake (WGAN-GP)."""
    alpha = torch.rand(len(real), 1, 1, device=real.device)
    x = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grad, = torch.autograd.grad(critic(x).sum(), x, create_graph=True)
    return ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()


def device():
    """CUDA, else CPU (not MPS: the gradient penalty is NaN on it for batches of 64, torch 2.13)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batches(x, rng):
    """Endless batches of BATCH_SIZE rows of x (a tensor on the device), the rows shuffled at each pass."""
    while True:
        order = torch.as_tensor(rng.permutation(len(x)), device=x.device)
        for k in range(len(x) // BATCH_SIZE):
            yield x[order[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]]


def train(run, epochs=EPOCHS, seed=0):
    """Train the GAN on the syllables of the train songs, results in the directory run (resumed from run/last.pt)."""
    run = Path(run)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    on = device()
    torch.manual_seed(seed)
    x = torch.as_tensor(syllable_dataset("train")["x"][:, None], device=on)
    generator, critic = Generator().to(on), Critic().to(on)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)
    epoch = 0
    if (run / "last.pt").exists():
        state = torch.load(run / "last.pt", map_location=on)
        for module, key in [(generator, "generator"), (critic, "critic"), (g_optimizer, "g_optimizer"),
                            (c_optimizer, "c_optimizer")]:
            module.load_state_dict(state[key])
        epoch = state["epoch"]
        print(f"resumed at epoch {epoch}", flush=True)
    real_batches = batches(x, np.random.default_rng([seed, epoch]))
    steps_per_epoch = len(x) // BATCH_SIZE
    log = open(run / "losses.csv", "a", newline="")
    writer = csv.writer(log)
    if log.tell() == 0:
        writer.writerow(["epoch", "step", "critic_loss", "wasserstein", "gradient_penalty", "generator_loss", "seconds"])
    start = time.time()
    while epoch < epochs:
        for step in range(steps_per_epoch):
            for _ in range(CRITIC_STEPS):
                real = next(real_batches)
                with torch.no_grad():
                    fake = generator(latent(len(real), on))
                wasserstein = critic(real).mean() - critic(fake).mean()
                penalty = gradient_penalty(critic, real, fake)
                c_loss = -wasserstein + LAMBDA * penalty
                c_optimizer.zero_grad(set_to_none=True)
                c_loss.backward()
                c_optimizer.step()
            g_loss = -critic(generator(latent(BATCH_SIZE, on))).mean()
            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()
            if step % LOG_EVERY == 0:
                writer.writerow([epoch, epoch * steps_per_epoch + step, c_loss.item(), wasserstein.item(),
                                 penalty.item(), g_loss.item(), round(time.time() - start, 1)])
                log.flush()
        epoch += 1
        print(f"epoch {epoch}: critic {c_loss.item():.3f}, wasserstein {wasserstein.item():.3f}, "
              f"generator {g_loss.item():.3f} ({time.time() - start:.0f} s)", flush=True)
        state = dict(generator=generator.state_dict(), critic=critic.state_dict(), g_optimizer=g_optimizer.state_dict(),
                     c_optimizer=c_optimizer.state_dict(), epoch=epoch)
        torch.save(state, run / "last.pt")
        if epoch % SAVE_EVERY == 0 or epoch == epochs:
            torch.save(dict(generator=state["generator"], epoch=epoch), run / "checkpoints" / f"epoch_{epoch:04d}.pt")
    log.close()


def load_generator(checkpoint, on=None):
    """Generator of a checkpoint (eval mode, on the device on)."""
    on = on or device()
    generator = Generator().to(on)
    generator.load_state_dict(torch.load(checkpoint, map_location=on)["generator"])
    return generator.eval()


def generate(generator, z, batch_size=256):
    """Audio (len(z), SLICE_LENGTH) generated from the latent vectors z (n, LATENT_DIM)."""
    on = next(generator.parameters()).device
    with torch.no_grad():
        return np.concatenate([generator(torch.as_tensor(z[k:k + batch_size], dtype=torch.float32, device=on))
                               [:, 0].cpu().numpy() for k in range(0, len(z), batch_size)])


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS)
    elif sys.argv[1] == "generate":
        n, seed = int(sys.argv[3]), int(sys.argv[5]) if len(sys.argv) > 5 else 0
        z = np.random.default_rng(seed).uniform(-1, 1, (n, LATENT_DIM)).astype(np.float32)
        np.savez(sys.argv[4], x=generate(load_generator(sys.argv[2]), z), z=z)
    else:
        raise ValueError(f"Unknown command {sys.argv[1]}: train or generate")
