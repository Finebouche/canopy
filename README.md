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
| `canary/datasets` | The canary songs (`songs.py`): annotations, phrases, train/test split by song, MFCC |
| `canary/datasets/Canary` | The dataset: its annotations (`annotations/`, tracked), those of the Zenodo record corrected (5 shifted songs, 21 relabeled phrases, see `songs.py`); its audio and MFCC caches (not tracked) |
| `canary` | Shared parts: reservoirs of the tasks (`models.py`, in JAX), MFCC and HAG pretraining phrases (`common.py`), HPO (`hpo.py`), conditions and paired comparisons (`evaluate.py`), certainty along the phrases, generalized readout |
| `canary/next_phrase_prediction` | Next-phrase prediction (`prediction.py`), phrase boundaries (`boundaries.py`, `evaluate.py`), their HPO and certainty |
| `canary/syllables_classification` | Frame-by-frame syllable classification (`classification.py`), its HPO and certainty, canapy reference |
| `outputs/canary` | Results (CSV); Optuna studies in `outputs/canary/studies` (not tracked) |
| `slurm` | Jobs on PlaFRIM: HPO, evaluations (`submit_canary.sh`, `submit_next_phrase.sh`, `submit_syllables.sh`, `canary.sbatch`) |
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
cached (`canary/datasets/Canary/boundary_features_*.pkl`).

---
## ▶️ Usage

From the root of the repository (`PYTHONPATH=$PWD`), for example:

```bash
HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_LEAK=1 python -m canary.next_phrase_prediction.hpo mean_hag 1000
HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_LEAK=1 python -m canary.next_phrase_prediction.prediction 500,1000 10
python -m canary.syllables_classification.classification 250,500,1000 10
```

The variables `HAG_CANARY_*` (task, variants of the studies, readouts) are documented in the docstrings of the
modules. On PlaFRIM, the whole experiments are submitted with the scripts of `slurm/` (`PYTHONPATH` = the checkout and
hag, `HAG_DIR`, default `~/hag_core`).

---
## 🎗 License

MIT
