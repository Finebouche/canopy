<p align="center"><h1 align="center">canopy</h1></p>
<p align="center">
	<em>Reservoir computing with Hebbian adaptive growth (HAG) on canary songs</em>
</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [🗂 Repository structure](#-repository-structure)
- [🚀 Setup](#-setup)
- [▶️ Usage](#️-usage)
- [🎗 License](#-license)

---

## 📍 Overview

Experiments of [HAG](https://github.com/Finebouche/hag) (Hebbian adaptive growth of reservoirs) on the songs of the
canary marron1 ([M1-2016-spring](https://zenodo.org/records/6521932): 459 annotated songs recorded at 44.1 kHz), from
their MFCC:

- **next-phrase prediction**: the reservoir runs over the whole song and, at the end of each phrase, the readout
  predicts the label of the next one (and the boundaries between the phrases);
- **syllable classification**: the label of every frame, as [canapy](https://github.com/birds-canopy/canapy) (Trouvain &
  Hinaut).

HAG reservoirs (mean HAG, variance HAG) are compared with random reservoirs (ESN) and hybrids, their hyperparameters
optimized with Optuna on the train songs, and evaluated on the test songs with paired comparisons over seeds.

---
## 🗂 Repository structure

| Folder | Content |
|:--|:--|
| `canopy/datasets` | The canary songs (`songs.py`): annotations, phrases, train/test split by song, MFCC |
| `canopy/datasets/marron1` | The dataset: its annotations (`annotations/`, tracked), those of the Zenodo record corrected (5 shifted songs, 21 relabeled phrases then 41 relabeled segments, see `songs.py`); its audio and MFCC caches (not tracked) |
| `canopy` | Shared parts: the settings of the studies from the variables `HAG_CANARY_*`, their names and their output folders (`settings.py`), MFCC and HAG pretraining phrases (`common.py`) |
| `canopy/models` | Reservoirs of the tasks (`reservoirs.py`, in JAX); the readouts trained online and of order 2 are studied in [bachue](https://github.com/Finebouche/bachue) |
| `canopy/studies` | One folder per study, self-contained: each file imports only the shared modules (`settings`, `common`, `datasets`, `models`, `studies/common.py`) and the files of its own study (`tests/test_layout.py`). `common.py`: the shared parts of the studies (storage of the Optuna studies, search space of the reservoirs, run of a study, conditions of the evaluations, paired comparisons, certainty along the phrases) |
| `canopy/studies/phrase_boundaries` | Phrase boundaries: `task.py` (targets, ridge readout, scores), `study.py` (study of the reservoirs, with its cross-validation by song), `evaluation.py` (final evaluation on the test songs) |
| `canopy/studies/next_phrase` | Prediction of the next phrase: `task.py`, `study.py`, `evaluation.py`, the n-grams of the previous phrases (`ngrams.py`) and the certainty along the phrases (`certainty.py`) |
| `canopy/studies/syllables_classification` | Syllable classification, frame by frame, as canapy: `task.py`, `study.py`, `evaluation.py`, `certainty.py` and the reference of canapy itself (`canapy_reference.py`) |
| `canopy/studies/syllable_generation` | Generation of syllables: single syllables cut from the phrases of marron1 (`syllables.py`), the low-dimensional WaveGAN of Pagliarini et al. (2021) in PyTorch (`wavegan.py`, trained on a GPU of PlaFRIM with `slurm/submit_gan.sh`), the ESN decoder of the syllable type (`decoder.py`) and the Hebbian sensorimotor model of Pagliarini, Leblois & Hinaut (2021) (`sensorimotor.py`) |
| `canopy/studies/song_generation` | Generation of songs phrase by phrase (`generator.py`): the predictor of the next phrase chooses the next phrase of the song its reservoir hears, a working memory holds it while a voice sings it; stage 1, recorded phrases of the train songs as the voice (`python -m canopy.studies.song_generation.generator`, results in `outputs/song_generation/`), then the GAN |
| `outputs/<study>` | One folder per study: `hpo/` (the Optuna databases, one per study of the reservoirs, not tracked), `performances/` (the result CSVs of the final evaluations), `analysis/`; `outputs/next_phrase/ablation_deltas_leak` (the deltas of the MFCC), `outputs/syllable_generation` (the runs of the GAN), `outputs/song_generation` |
| `slurm` | Jobs on PlaFRIM: HPO, evaluations (`submit_canary.sh`, `submit_next_phrase.sh`, `submit_syllables.sh`, `canary.sbatch`) |
| `ETAT_DES_TRAVAUX.md` | State of the work (in French): what was tested, the conclusions, what to rerun with the final annotations |
| `1. Exploration.ipynb` | Exploration of the songs: statistics, split, next-phrase prediction along a song, space of the phrases (UMAP, t-SNE), outliers and corrections of the annotations |

---
## 🚀 Setup

The environment (`environment.yml`) installs, in editable mode, next to the repository:

- reservoirpy (`../../Autres/reservoirpy`): a branch based on `v0.4.3-dev` with `HAGReservoir` and the solvers of
  `Ridge` (branch `ridge-solvers` of the fork, reservoirpy PR #261);
- then hag (`../hag`, branch `main`, the [hag repository](https://github.com/Finebouche/hag)) and canopy, without
  their dependencies (hag pins another version of reservoirpy).

```bash
micromamba env create -f environment.yml
micromamba activate canopy_env
pip install --no-deps -e ../hag -e .
```

The environment of hag (`hag_env`) works too, with `PYTHONPATH=$PWD`.

The audio (~855 MB) is downloaded from Zenodo at the first loading of the songs, and the MFCC are computed once and
cached (`canopy/datasets/marron1/boundary_features_*.pkl`).

---
## ▶️ Usage

From the root of the repository (`PYTHONPATH=$PWD`), for example:

```bash
HAG_CANARY_CALLS=1 HAG_CANARY_LEAK=1 python -m canopy.studies.next_phrase.study mean_hag 1000
HAG_CANARY_CALLS=1 HAG_CANARY_LEAK=1 python -m canopy.studies.next_phrase.evaluation 500,1000 10
python -m canopy.studies.syllables_classification.evaluation 250,500,1000 10
```

The variables `HAG_CANARY_*` (variants of the studies, their storage) are documented in `canopy/settings.py`. On PlaFRIM, the whole experiments are submitted with the scripts of `slurm/` (`PYTHONPATH` = the checkout and
hag, `HAG_DIR`, default `~/hag_core`).

---
## 🎗 License

MIT
