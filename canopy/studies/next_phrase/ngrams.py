"""
References of the prediction of the next phrase (canopy.studies.next_phrase.task): n-grams of the true labels of the previous
phrases, fixed order n (the n previous labels, START before the first phrase; unseen context: the most frequent next
label) and adaptive (the longest context seen in the train songs, up to MAX_ORDER), scored at the test points with the
metrics of the reservoirs (canopy.studies.next_phrase.task.metrics), and the share of the points whose context was not seen in
the train songs (marginal frequencies then).

Results: outputs/next_phrase/performances/next_phrase_ngrams.csv (next_phrase_calls_ngrams.csv with
HAG_CANARY_CALLS=1).
Usage: python -m canopy.studies.next_phrase.ngrams
"""
from collections import Counter

import numpy as np
import pandas as pd

from canopy.settings import CALLS, outputs_dir
from canopy.studies.next_phrase.task import START, load_task_dataset, metrics

MAX_ORDER = 10

if __name__ == "__main__":
    songs, train_songs, test_songs, frame, classes = load_task_dataset()
    true = np.concatenate([songs[w]["next"] for w in test_songs])
    index = {label: i for i, label in enumerate(classes)}

    def contexts(song, n):
        history = [START] * n + [label for _, _, label in song["next_phrases"]]
        return [tuple(history[j:j + n]) for j in range(len(song["next"]))]

    rows = []
    for order in [*range(0, MAX_ORDER + 1), None]:
        orders = list(range(MAX_ORDER, -1, -1)) if order is None else sorted({order, 0}, reverse=True)
        counts = {n: {} for n in orders}
        for w in train_songs:
            for n in orders:
                for context, label in zip(contexts(songs[w], n), songs[w]["next"]):
                    counts[n].setdefault(context, Counter())[label] += 1
        scores, unseen = [], 0
        for w in test_songs:
            by_order = {n: contexts(songs[w], n) for n in orders}
            for j in range(len(songs[w]["next"])):
                n = next(n for n in orders if by_order[n][j] in counts[n])
                unseen += n == 0 and (order is None or order > 0)
                row = np.zeros(len(classes))
                for label, count in counts[n][by_order[n][j]].items():
                    row[index[label]] = count
                scores.append(row / row.sum())
        rows.append(dict(condition="ngram_adaptive" if order is None else f"ngram_{order}", order=order,
                         unseen_context=unseen / len(scores), **metrics(np.array(scores), true, classes)))
    results = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    outputs_dir("next_phrase", "performances").mkdir(parents=True, exist_ok=True)
    file_name = outputs_dir("next_phrase", "performances") / f"next_phrase_{'calls_' * CALLS}ngrams.csv"
    results.to_csv(file_name, index=False)
    print("Saved to", file_name)
