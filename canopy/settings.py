"""
Settings of the canary tasks, from the variables of the environment: the variants of the studies of the reservoirs and
their names, their storage, the cross-validation.

  CANOPY_LEAKY_RESERVOIRS=1: leaky reservoirs, the leak rate in the search (next phrase and syllables; studies
      "..._leak_");
  CANOPY_WITH_CALLS=1: the calls are phrases too (next phrase; studies "next_phrase_calls_");
  CANOPY_RESET_EACH_PHRASE=1: the reservoirs hear the current phrase only, from a zero state (next phrase;
      "next_phrase_reset_");
  CANOPY_SEARCH_INPUT_GAINS=1: the gain of each group of input features in the search (syllables; "syllables_gains_");
  CANOPY_LABELS_WITH_DURATIONS=1: the reservoirs reading the labels also read the duration of each phrase and the
    silence before it (next phrase from the labels; studies "next_phrase_labels_..._durations_");
  CANOPY_N_TRIALS: trials of a study (400); CANOPY_WORKERS: processes optimizing a study (1);
  CANOPY_JOB_STUDIES_DIR: folder of the Optuna database of a job (set by slurm/canary.sbatch, in /dev/shm during the
      job);
    without it, the database of the dataset of each study, outputs/<study>/hpo/marron1_db.sqlite3;
  CANOPY_RIDGE_SOLVER: solver of reservoirpy's Ridge, the readout of the next phrase ("svd": the singular value
    decomposition of the centered states, stable with the smallest coefficients);
  CANOPY_DATASET: the canary of the studies, "marron1" (default), or "llb3", "llb11" or "llb16" (three other canaries,
    Cohen et al. 2022, eLife; canopy.common.load_dataset). Only the next phrase from the audio and from the labels run
    on them (DATASET_STUDIES), without calls (they have none); their Optuna studies, in
    outputs/<study>/hpo/<dataset>_db.sqlite3, and their results are prefixed with "<dataset>_" (PREFIXES,
    DATASET_PREFIX);
  CANOPY_XENO_CANTO_KEY: the personal key of the API of xeno-canto (https://xeno-canto.org/account), read by
    canopy.datasets.xeno_canto to list the recordings it downloads.
(CANOPY_TASK: the study optimized by slurm/canary.sbatch, "boundaries", "next_phrase" or "syllables"; the scripts of the
studies do not read it.)

The studies of canopy.studies ("phrase_boundaries", "next_phrase", "next_phrase_labels", "next_syllable",
"syllables_classification")
write in one folder each,
outputs/<study> (outputs_dir): hpo/ (one Optuna database per dataset with all the Optuna studies of the reservoirs,
database), performances/ (the results of the final evaluations), analysis/, figures/; the generations, in
outputs/syllable_generation and outputs/song_generation.
Optuna studies: PREFIXES[study] + "<model>_<size>" (study_name): the phrase boundaries without prefix,
"next_phrase_[calls_][reset_][leak_]", "next_phrase_labels_[calls_][durations_][leak_]",
"next_syllable_calls_[phrases_][change_][leak_]" and "syllables_[gains_][leak_]".
"""
import os
from pathlib import Path

from canopy.common import OUTPUTS

LEAK = os.environ.get("CANOPY_LEAKY_RESERVOIRS") == "1"
CALLS = os.environ.get("CANOPY_WITH_CALLS") == "1"
RESET = os.environ.get("CANOPY_RESET_EACH_PHRASE") == "1"
INPUT_GAINS = os.environ.get("CANOPY_SEARCH_INPUT_GAINS") == "1"
LABELS_WITH_DURATIONS = os.environ.get("CANOPY_LABELS_WITH_DURATIONS") == "1"
# (canopy.studies.next_syllable: the readout trained on the next syllable of every step, "syllables", or on the next
# phrase at the ends of the phrases only, "phrases"; the score of its studies at the ends of the phrases, "raw" or
# knowing that the phrase changes, "change_known")
SYLLABLE_READOUT = os.environ.get("CANOPY_SYLLABLE_READOUT", "syllables")
SYLLABLE_SCORE = os.environ.get("CANOPY_SYLLABLE_SCORE", "raw")
if SYLLABLE_READOUT not in ("syllables", "phrases") or SYLLABLE_SCORE not in ("raw", "change_known"):
    raise ValueError(f"CANOPY_SYLLABLE_READOUT={SYLLABLE_READOUT} (syllables or phrases), "
                     f"CANOPY_SYLLABLE_SCORE={SYLLABLE_SCORE} (raw or change_known)")
N_TRIALS = int(os.environ.get("CANOPY_N_TRIALS", 400))
WORKERS = int(os.environ.get("CANOPY_WORKERS", 1))
JOB_STUDIES = os.environ.get("CANOPY_JOB_STUDIES_DIR")
RIDGE_SOLVER = os.environ.get("CANOPY_RIDGE_SOLVER", "svd")
XENO_CANTO_KEY = os.environ.get("CANOPY_XENO_CANTO_KEY")

DATASET_NAME = os.environ.get("CANOPY_DATASET", "marron1")
DATASETS = ("marron1", "llb3", "llb11", "llb16")
DATASET_STUDIES = ("next_phrase", "next_phrase_labels")  # (the studies which run on the other datasets than marron1)
if DATASET_NAME not in DATASETS:
    raise ValueError(f"CANOPY_DATASET={DATASET_NAME}: one of {DATASETS}")
if DATASET_NAME != "marron1" and CALLS:
    raise ValueError(f"CANOPY_DATASET={DATASET_NAME} has no calls: CANOPY_WITH_CALLS=1 is for marron1 only")
DATASET_PREFIX = "" if DATASET_NAME == "marron1" else f"{DATASET_NAME}_"  # (names of the studies and of the results)
DATABASE_NAME = f"{DATASET_NAME}_db.sqlite3"

N_FOLDS = 3  # (cross-validation by song on the train songs)
FOLD_SEED = 0
CV_SEED = 1000  # (the matrices of fold k drawn with np.random.seed(CV_SEED + k), the same for all the trials)

PREFIXES = {"phrase_boundaries": "",  # (the boundary studies have no variant: no leak)
            "next_phrase": "next_phrase_" + "calls_" * CALLS + "reset_" * RESET + "leak_" * LEAK,
            "next_phrase_labels": ("next_phrase_labels_" + "calls_" * CALLS + "durations_" * LABELS_WITH_DURATIONS
                                   + "leak_" * LEAK),
            "next_syllable": ("next_syllable_" + "calls_" * CALLS + "phrases_" * (SYLLABLE_READOUT == "phrases")
                              + "change_" * (SYLLABLE_SCORE == "change_known") + "leak_" * LEAK),
            "syllables_classification": "syllables_" + "gains_" * INPUT_GAINS + "leak_" * LEAK}


PREFIXES = {study: DATASET_PREFIX + prefix for study, prefix in PREFIXES.items()}


def study_name(study, model, size):
    """Name of the Optuna study of the reservoirs of a model and size, for a study of canopy.studies."""
    return f"{PREFIXES[study]}{model}_{size}"


def outputs_dir(study, kind=None):
    """outputs/<study>[/<kind>] (kind: "hpo", "performances" or "analysis")."""
    return OUTPUTS / study / kind if kind else OUTPUTS / study


def database(study):
    """Optuna database of the dataset for a study: outputs/<study>/hpo/<dataset>_db.sqlite3, all its Optuna studies."""
    return outputs_dir(study, "hpo") / DATABASE_NAME


def studies_database(study):
    """Optuna database read and written: that of the job (CANOPY_JOB_STUDIES_DIR) if set, else that of the dataset."""
    return Path(JOB_STUDIES) / DATABASE_NAME if JOB_STUDIES else database(study)
