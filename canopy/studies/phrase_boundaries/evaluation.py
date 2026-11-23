"""
Final evaluation of the phrase boundaries on the test songs (used only here), after the studies of
canopy.studies.phrase_boundaries.study. Conditions, for each size (number of neurons in all): "mfcc" (no reservoir,
once) and canopy.studies.common.conditions (the pure models with the best parameters of their study, the projections of
the HAG models, the hybrids). The ridge coefficient and the peak threshold are those of the best trial of the study, or,
for the conditions without a study (mfcc, projections, hybrids), chosen with the same cross-validation on the train
songs (canopy.studies.phrase_boundaries.study.cross_validate). For each seed (the same seeds for all the conditions,
i.e. paired), the reservoirs are drawn with np.random.seed(seed) and learn (HAG) on phrases of all the train songs, the
readout is trained on all the train songs and the boundaries of the test songs are scored; then the paired comparisons
by seed of the F1 (canopy.studies.common).

Results: outputs/phrase_boundaries/performances/evaluation.csv (one row per condition, size and seed) and
comparisons.csv.
Usage: python -m canopy.studies.phrase_boundaries.evaluation [sizes] [nb_seeds]   (default 250,500,1000 and 10)
"""
import sys

import numpy as np
import pandas as pd

from canopy.settings import outputs_dir
from canopy.studies.common import compare, conditions
from canopy.studies.phrase_boundaries.study import cross_validate
from canopy.studies.phrase_boundaries.task import evaluate, load_task_dataset

SIZES = [250, 500, 1000]
REFERENCE = "random_ee"  # (condition of the paired comparisons)
PERFORMANCES = outputs_dir("phrase_boundaries", "performances")

if __name__ == "__main__":
    sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SIZES
    nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    songs, train_songs, test_songs, frame = load_task_dataset()
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, "
          f"{sum(len(songs[w]['boundaries']) for w in test_songs)} test boundaries", flush=True)
    todo = [("mfcc", "mfcc", {}, None, None, 0)]
    for size in sizes:
        todo += [(*condition, size) for condition in conditions(size, "phrase_boundaries")]
    rows = []
    for condition, model, params, ridge, threshold, size in todo:
        if ridge is None:
            _, ridge, threshold = cross_validate(model, params, size, songs, train_songs, frame)
        for seed in range(nb_seeds):
            result = evaluate(model, params, size, ridge, threshold, songs, train_songs, test_songs, frame, seed)
            rows.append(dict(condition=condition, network_size=size, seed=seed, ridge=ridge, threshold=threshold,
                             **result))
        f1 = [row["f1"] for row in rows[-nb_seeds:]]
        print(f"{condition} ({size}): F1 {np.mean(f1):.3f} ± {np.std(f1):.3f} (ridge {ridge:.0e}, threshold "
              f"{threshold:.2f})", flush=True)
    results = pd.DataFrame(rows)
    comparisons = compare(results, "f1", REFERENCE)
    pd.set_option("display.width", 200)
    print(comparisons.round(4).to_string(index=False))
    PERFORMANCES.mkdir(parents=True, exist_ok=True)
    results.to_csv(PERFORMANCES / "evaluation.csv", index=False)
    comparisons.to_csv(PERFORMANCES / "comparisons.csv", index=False)
