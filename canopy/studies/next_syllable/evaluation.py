"""
Final evaluation of the reservoirs that read one syllable per step on the test songs (canopy.studies.next_syllable.task),
after the studies of canopy.studies.next_syllable.study (with the same variables: CANOPY_WITH_CALLS=1,
CANOPY_SYLLABLE_READOUT, CANOPY_SYLLABLE_SCORE, CANOPY_LEAKY_RESERVOIRS). Conditions, for each size: "syllables" (the inputs without reservoir, once: the readout on the
current syllable alone, which predicts the same type at every step) and canopy.studies.common.conditions. The ridge
coefficient is that of the best trial of the study, or, for the conditions without a study, chosen by the same
cross-validation (canopy.studies.next_syllable.study.choose_ridge). For each seed (paired), the reservoirs are drawn with
np.random.seed(seed), the readout is trained on all the steps of the train songs and scored on the test songs: the
metrics of the next phrase at its prediction points, and the accuracy of the next syllable on all the steps
(syllable_accuracy, the readout of the syllables only) and that of the next phrase knowing that the phrase changes
(accuracy_change_known); then the paired comparisons by seed of the score of the studies (accuracy, or
accuracy_change_known with CANOPY_SYLLABLE_SCORE=change_known).

Results: outputs/next_syllable/performances/<prefix of the studies>hpo.csv (e.g. next_syllable_calls_hpo.csv) and
..._comparisons.csv, to compare with outputs/next_phrase/performances and outputs/next_phrase_labels/performances.

Usage: CANOPY_WITH_CALLS=1 python -m canopy.studies.next_syllable.evaluation [sizes] [nb_seeds]   (default 500,1000
  and 10 seeds)
"""
import sys

import numpy as np
import pandas as pd

from canopy.settings import PREFIXES, SYLLABLE_SCORE, outputs_dir
from canopy.studies.common import compare, conditions
from canopy.studies.next_syllable.study import choose_ridge
from canopy.studies.next_syllable.task import evaluate, load_task_dataset

SIZES = [500, 1000]
REFERENCE = "random_ee"  # (condition of the paired comparisons)
NAME = f"{PREFIXES['next_syllable']}hpo"
PERFORMANCES = outputs_dir("next_syllable", "performances")

if __name__ == "__main__":
    sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SIZES
    nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    songs, train_songs, test_songs, classes = load_task_dataset()
    todo = [("syllables", "mfcc", {}, None, 0)]  # (model "mfcc" of canopy.models.reservoirs.build: no reservoir)
    for size in sizes:
        todo += [(condition, model, params, ridge, size)
                 for condition, model, params, ridge, _ in conditions(size, "next_syllable")]
    rows = []
    for condition, model, params, ridge, size in todo:
        cv_accuracy = np.nan  # (that of the study is in its best trial)
        if ridge is None:
            ridge, cv_scores = choose_ridge(model, params, size, songs, train_songs, classes)
            cv_accuracy = cv_scores["accuracy_change_known" if SYLLABLE_SCORE == "change_known" else "accuracy"]
        for seed in range(nb_seeds):
            result = evaluate(model, params, size, ridge, songs, train_songs, test_songs, classes, seed)
            rows.append(dict(condition=condition, network_size=size, seed=seed, ridge=ridge, cv_accuracy=cv_accuracy,
                             **result))
        last = pd.DataFrame(rows[-nb_seeds:])
        print(f"{condition} ({size}): next phrase {100 * last['accuracy'].mean():.2f} ± {100 * last['accuracy'].std(ddof=0):.2f} %"
              f", knowing the change {100 * last['accuracy_change_known'].mean():.2f} %, next syllable "
              f"{100 * last['syllable_accuracy'].mean():.2f} % (ridge {ridge:.0e})", flush=True)
    results = pd.DataFrame(rows)
    # (compare adds the condition without reservoir, which it names "mfcc", to each size)
    score = "accuracy_change_known" if SYLLABLE_SCORE == "change_known" else "accuracy"
    comparisons = compare(results.replace({"condition": {"syllables": "mfcc"}}), score, REFERENCE)
    comparisons = comparisons.replace("mfcc", "syllables")
    pd.set_option("display.width", 200)
    print(comparisons.round(4).to_string(index=False))
    PERFORMANCES.mkdir(parents=True, exist_ok=True)
    results.to_csv(PERFORMANCES / f"{NAME}.csv", index=False)
    comparisons.to_csv(PERFORMANCES / f"{NAME}_comparisons.csv", index=False)
    print("Saved to", PERFORMANCES / f"{NAME}.csv", "and", PERFORMANCES / f"{NAME}_comparisons.csv")
