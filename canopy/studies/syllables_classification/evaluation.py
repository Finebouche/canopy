"""
Final evaluation of the syllable classification on the test songs (canopy.studies.syllables_classification.task), after
the studies of canopy.studies.syllables_classification.study (with the same variables: CANOPY_SEARCH_INPUT_GAINS,
CANOPY_LEAKY_RESERVOIRS). Conditions, for each size: "mfcc" (no reservoir, once) and canopy.studies.common.conditions.
The ridge coefficient is chosen by the cross-validation of the task
(canopy.studies.syllables_classification.study.choose_ridge), then the readout is trained on all the train songs and
scored on the test songs for each seed (paired comparisons of the frame accuracy, canopy.studies.common).

Results: outputs/syllables_classification/performances/<prefix of the studies>hpo.csv and ..._comparisons.csv
(syllables_hpo.csv, syllables_gains_hpo.csv with CANOPY_SEARCH_INPUT_GAINS=1, syllables_gains_leak_hpo.csv with
CANOPY_LEAKY_RESERVOIRS=1 too). Usage:
  python -m canopy.studies.syllables_classification.evaluation [sizes] [nb_seeds]   all the conditions (default
      250,500,1000, 10 seeds)
  python -m canopy.studies.syllables_classification.evaluation run <condition> <size> [nb_seeds]   one condition (one
      job of
      slurm/submit_syllables.sh), results in outputs/syllables_classification/performances/<prefix>hpo_parts
  python -m canopy.studies.syllables_classification.evaluation merge   results of the parts and the comparisons
  python -m canopy.studies.syllables_classification.evaluation list <sizes>   "<condition> <size>" of all the conditions
"""
import sys

import numpy as np
import pandas as pd

from canopy.studies.common import compare, conditions
from canopy.settings import PREFIXES, outputs_dir
from canopy.studies.syllables_classification.study import choose_ridge
from canopy.studies.syllables_classification.task import evaluate, load_task_dataset

SIZES = [250, 500, 1000]
REFERENCE = "random_ee"  # (condition of the paired comparisons)
NAME = f"{PREFIXES['syllables_classification']}hpo"
PERFORMANCES = outputs_dir("syllables_classification", "performances")
PARTS = PERFORMANCES / f"{NAME}_parts"


def all_conditions(sizes):
    """[(condition, model, params, size)], mfcc first (size 0)."""
    todo = [("mfcc", "mfcc", {}, 0)]
    for size in sizes:
        todo += [(condition, model, params, size)
                 for condition, model, params, _, _ in conditions(size, "syllables_classification")]
    return todo


def run(todo, nb_seeds):
    songs, train_songs, test_songs, frame, classes = load_task_dataset()
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, {len(classes)} classes, "
          f"{sum(len(songs[w]['labels']) for w in test_songs)} test frames", flush=True)
    rows = []
    for condition, model, params, size in todo:
        ridge, cv_accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
        for seed in range(nb_seeds):
            result = evaluate(model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed)
            rows.append(dict(condition=condition, network_size=size, seed=seed, ridge=ridge,
                             cv_frame_accuracy=cv_accuracy, **result))
        accuracy = [row["frame_accuracy"] for row in rows[-nb_seeds:]]
        print(f"{condition} ({size}): frame accuracy {100 * np.mean(accuracy):.2f} ± {100 * np.std(accuracy):.2f} % "
              f"(ridge {ridge:.0e}, cross-validation {100 * cv_accuracy:.2f} %)", flush=True)
    return pd.DataFrame(rows)


def save(results):
    comparisons = compare(results, "frame_accuracy", REFERENCE)
    pd.set_option("display.width", 200)
    print(comparisons.round(4).to_string(index=False))
    PERFORMANCES.mkdir(parents=True, exist_ok=True)
    results.to_csv(PERFORMANCES / f"{NAME}.csv", index=False)
    comparisons.to_csv(PERFORMANCES / f"{NAME}_comparisons.csv", index=False)
    print("Saved to", PERFORMANCES / f"{NAME}.csv", "and", PERFORMANCES / f"{NAME}_comparisons.csv")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "list":
        for condition, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size] if size else []) if c[0] == condition and c[3] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        PARTS.mkdir(parents=True, exist_ok=True)
        run(todo, nb_seeds).to_csv(PARTS / f"{condition}_{size}.csv", index=False)
    elif command == "merge":
        parts = sorted(PARTS.glob("*.csv"))
        print(f"{len(parts)} conditions")
        save(pd.concat([pd.read_csv(part) for part in parts], ignore_index=True))
    else:
        sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SIZES
        nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        save(run(all_conditions(sizes), nb_seeds))

