"""
Final evaluation of the prediction of the next phrase on the test songs (canopy.studies.next_phrase.task), after the
studies of canopy.studies.next_phrase.study (with the same variables: CANOPY_WITH_CALLS, CANOPY_RESET_EACH_PHRASE,
CANOPY_LEAKY_RESERVOIRS). Conditions, for each size: "mfcc" (no reservoir, once) and canopy.studies.common.conditions.
The ridge coefficient is that of the best trial of the study, or, for the conditions without a study, chosen by the same
cross-validation (canopy.studies.next_phrase.study.choose_ridge). For each seed (paired), the reservoirs are drawn with
np.random.seed(seed), the readout is trained on all the train songs and scored on the test songs
(canopy.studies.next_phrase.task.metrics); then the paired comparisons by seed of the accuracy (canopy.studies.common).

Results: outputs/next_phrase/performances/<prefix of the studies>hpo.csv (e.g. next_phrase_calls_leak_hpo.csv) and
..._comparisons.csv.
Usage:
  python -m canopy.studies.next_phrase.evaluation [sizes] [nb_seeds]   all the conditions (default 500,1000, 10 seeds)
  python -m canopy.studies.next_phrase.evaluation run <condition> <size> [nb_seeds]   one condition (one job of
      slurm/submit_next_phrase.sh), results in outputs/next_phrase/performances/<prefix>hpo_parts
  python -m canopy.studies.next_phrase.evaluation merge   results of the parts and the comparisons
  python -m canopy.studies.next_phrase.evaluation list <sizes>   "<condition> <size>" of all the conditions
"""
import sys

import numpy as np
import pandas as pd

from canopy.studies.common import compare, conditions
from canopy.settings import PREFIXES, outputs_dir
from canopy.studies.next_phrase.study import choose_ridge
from canopy.studies.next_phrase.task import evaluate, load_task_dataset

SIZES = [500, 1000]
REFERENCE = "random_ee"  # (condition of the paired comparisons)
NAME = f"{PREFIXES['next_phrase']}hpo"
PERFORMANCES = outputs_dir("next_phrase", "performances")
PARTS = PERFORMANCES / f"{NAME}_parts"


def all_conditions(sizes):
    """[(condition, model, params, ridge, size)], mfcc first (size 0); ridge None when chosen by cross-validation."""
    todo = [("mfcc", "mfcc", {}, None, 0)]
    for size in sizes:
        todo += [(condition, model, params, ridge, size)
                 for condition, model, params, ridge, _ in conditions(size, "next_phrase")]
    return todo


def run(todo, nb_seeds):
    """Results of each condition and seed."""
    songs, train_songs, test_songs, frame, classes = load_task_dataset()
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, {len(classes)} classes, "
          f"{sum(len(songs[w]['next']) for w in test_songs)} test points", flush=True)
    rows = []
    for condition, model, params, ridge, size in todo:
        cv_accuracy = np.nan  # (that of the study is in its best trial)
        if ridge is None:
            ridge, cv_accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
        accuracies = []
        for seed in range(nb_seeds):
            result = evaluate(model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed)
            rows.append(dict(condition=condition, network_size=size, seed=seed, ridge=ridge, cv_accuracy=cv_accuracy,
                             **result))
            accuracies.append(result["accuracy"])
        print(f"{condition} ({size}): accuracy {100 * np.mean(accuracies):.2f} ± {100 * np.std(accuracies):.2f} % "
              f"(ridge {ridge:.0e})", flush=True)
    return pd.DataFrame(rows)


def save(results):
    """The results and the paired comparisons of the conditions."""
    pd.set_option("display.width", 200)
    PERFORMANCES.mkdir(parents=True, exist_ok=True)
    results.to_csv(PERFORMANCES / f"{NAME}.csv", index=False)
    comparisons = compare(results, "accuracy", REFERENCE)
    print(comparisons.round(4).to_string(index=False))
    comparisons.to_csv(PERFORMANCES / f"{NAME}_comparisons.csv", index=False)
    print("Saved to", PERFORMANCES / f"{NAME}.csv", "and", PERFORMANCES / f"{NAME}_comparisons.csv")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "list":
        for condition, _, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size] if size else []) if c[0] == condition and c[4] == size]
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
