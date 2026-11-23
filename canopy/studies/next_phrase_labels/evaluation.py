"""
Final evaluation of the reservoirs that read the labels of the phrases on the test songs
(canopy.studies.next_phrase_labels.task), after the studies of canopy.studies.next_phrase_labels.study (with the same
variables: CANOPY_WITH_CALLS, CANOPY_LABELS_WITH_DURATIONS, CANOPY_LEAKY_RESERVOIRS). Conditions, for each size:
"labels" (the inputs without reservoir, once: the readout on the last label alone, a first-order model) and
canopy.studies.common.conditions. The ridge coefficient is that of the best trial of the study, or, for the conditions
without a study, chosen by the same cross-validation (canopy.studies.next_phrase_labels.study.choose_ridge). For each
seed (paired), the reservoirs are drawn with np.random.seed(seed), the readout is trained on all the train songs and
scored on the test songs with the metrics of the task on the audio; then the paired comparisons by seed of the accuracy.

Results: outputs/next_phrase_labels/performances/<prefix of the studies>hpo.csv (e.g.
next_phrase_labels_calls_leak_hpo.csv) and ..._comparisons.csv, to compare with those of the audio,
outputs/next_phrase/performances.

Usage: python -m canopy.studies.next_phrase_labels.evaluation [sizes] [nb_seeds]   (default 500,1000 and 10 seeds)
"""
import sys

import numpy as np
import pandas as pd

from canopy.settings import PREFIXES, outputs_dir
from canopy.studies.common import compare, conditions
from canopy.studies.next_phrase_labels.study import choose_ridge
from canopy.studies.next_phrase_labels.task import evaluate, load_task_dataset

SIZES = [500, 1000]
REFERENCE = "random_ee"  # (condition of the paired comparisons)
NAME = f"{PREFIXES['next_phrase_labels']}hpo"
PERFORMANCES = outputs_dir("next_phrase_labels", "performances")

if __name__ == "__main__":
    sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SIZES
    nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    songs, train_songs, test_songs, classes = load_task_dataset()
    todo = [("labels", "mfcc", {}, None, 0)]  # (model "mfcc" of canopy.models.reservoirs.build: no reservoir)
    for size in sizes:
        todo += [(condition, model, params, ridge, size)
                 for condition, model, params, ridge, _ in conditions(size, "next_phrase_labels")]
    rows = []
    for condition, model, params, ridge, size in todo:
        cv_accuracy = np.nan  # (that of the study is in its best trial)
        if ridge is None:
            ridge, cv_accuracy = choose_ridge(model, params, size, songs, train_songs, classes)
        for seed in range(nb_seeds):
            result = evaluate(model, params, size, ridge, songs, train_songs, test_songs, classes, seed)
            rows.append(dict(condition=condition, network_size=size, seed=seed, ridge=ridge, cv_accuracy=cv_accuracy,
                             **result))
        accuracy = [row["accuracy"] for row in rows[-nb_seeds:]]
        print(f"{condition} ({size}): accuracy {100 * np.mean(accuracy):.2f} ± {100 * np.std(accuracy):.2f} % "
              f"(ridge {ridge:.0e})", flush=True)
    results = pd.DataFrame(rows)
    # (compare adds the condition without reservoir, which it names "mfcc", to each size)
    comparisons = compare(results.replace({"condition": {"labels": "mfcc"}}), "accuracy", REFERENCE)
    comparisons = comparisons.replace("mfcc", "labels")
    pd.set_option("display.width", 200)
    print(comparisons.round(4).to_string(index=False))
    PERFORMANCES.mkdir(parents=True, exist_ok=True)
    results.to_csv(PERFORMANCES / f"{NAME}.csv", index=False)
    comparisons.to_csv(PERFORMANCES / f"{NAME}_comparisons.csv", index=False)
    print("Saved to", PERFORMANCES / f"{NAME}.csv", "and", PERFORMANCES / f"{NAME}_comparisons.csv")
