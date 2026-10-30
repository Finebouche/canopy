"""
Learning rules of the readout of the next-phrase prediction (canary.next_phrase_prediction.prediction), compared on the
same reservoirs: from the least to the most biologically plausible,
  - "ridge": ridge regression on all the training rows at once (reservoirpy's Ridge, solver RIDGE_SOLVER), the
    reference of the other tasks;
  - "rls": online recursive least squares (reservoirpy's RLS, the RLS-based FORCE rule of Sussillo & Abbott 2009,
    non-local: each synapse uses the inverse correlation matrix of all the inputs), initial P = I / alpha;
  - "lms": online least mean squares (the rule of reservoirpy's LMS, the LMS-based FORCE rule: local, but each readout
    needs its exact error; trained in JAX, readout_rules.lms), learning rate c / mean ||x||^2 (c the normalized rate,
    stable below 2);
  - "eh": the exploratory Hebbian rule of Hoerzer et al. (2014) (the rule of canary.eh_readout.EH, the paper's: filtered
    baselines, a single global binary reward for all the readouts, no teacher; trained in JAX, readout_rules.eh),
    learning rate c / mean ||x||^2;
  - "eh_np": the same with the baseline "sample" (node perturbation of Fiete & Seung 2006: the reward compared with
    that of the output without noise on the same input; not the paper's rule, which does not know the noise), global
    reward; "eh_np_local": with a reward for each readout (its own modulatory signal).
The online rules see the training rows song by song, each song in its time order, the songs in a random order at each
epoch (pass over the training songs). Their inputs are centered ("center", RLS only) or standardized ("standardize")
with the mean and standard deviation of each unit on the training rows (a homeostatic estimate of the mean activity and of the
gain of each input, local to its synapses): the states have a large common mean and small fluctuations (mean 0.16,
standard deviation 0.01 per unit for random_ee 500), and without this the online rules learn the direction of the
mean (accuracy at the level of chance). The paper's EH rule needs slowly changing inputs (its filtered baseline
estimates the noise from the change of the output): on the independent prediction points, it does not learn.

Reservoirs and readout of each condition: those of the studies of the readout (HAG_CANARY_TAIL_HPO=1, with the
variables of the studies: HAG_CANARY_CALLS, HAG_CANARY_LEAK), with the readout (training rows, decision, tail) of the
condition in their evaluation (column hpo of outputs/canary/<prefix>hpo_readouts.csv, e.g.
next_phrase_calls_tail_hpo_readouts.csv). The hyperparameters of each rule (ridge coefficient, alpha, normalized
learning rate, noise, number of epochs) are chosen in the 3-fold cross-validation by song of the studies (out-of-fold
accuracy, reservoirs of fold k drawn with the seed CV_SEED + k), then each rule is trained on all the train songs and
scored on the test songs for each seed (paired comparisons with "ridge"). Rules: HAG_CANARY_READOUT_RULES
(comma-separated, default all).

Results: outputs/canary/<prefix>readout_rules<suffix>.csv (one row per condition, size, seed and rule),
<prefix>readout_rules<suffix>_comparisons.csv (each rule - ridge, paired by seed) and <prefix>readout_rules<suffix>_cv.csv
(out-of-fold accuracy of each configuration and number of epochs: the learning curves), suffix
HAG_CANARY_RULES_SUFFIX (default none).
Usage (HAG_CANARY_TASK=next_phrase HAG_CANARY_TAIL_HPO=1, with the variables of the studies):
  python -m canary.next_phrase_prediction.readout_rules run <condition> <size> [nb_seeds]   one condition
                                                       (slurm/submit_readout_rules.sh), results in <name>_parts
  python -m canary.next_phrase_prediction.readout_rules merge          results of the parts and the comparisons
  python -m canary.next_phrase_prediction.readout_rules list <sizes>   "<condition> <size>" of the conditions (no _proj)
"""
import os
import sys

import numpy as np
import pandas as pd
import jax.numpy as jnp
from reservoirpy.nodes import RLS
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold

from canary.common import OUTPUTS, RIDGES, load_dataset, pretrain_phrases
from canary.hpo import CV_SEED, FOLD_SEED, N_FOLDS
from canary.models import build
from canary.next_phrase_prediction.hpo import STUDY_PREFIX, TAIL_HPO, TAIL_SEPARATE
from canary.next_phrase_prediction.prediction import (NAME, all_conditions, decision_of, fitted_readouts, metrics,
                                                      point_windows, prepare, readout_of, training_of)
from canary.next_phrase_prediction.readout_rules.eh import EH
from canary.next_phrase_prediction.readout_rules.lms import LMS

RULES = os.environ.get("HAG_CANARY_READOUT_RULES", "ridge,rls,lms,eh,eh_np,eh_np_local").split(",")
NORMALIZATIONS = ["center", "standardize"]  # (inputs of RLS; LMS and EH: "standardize", better in all the conditions)
RLS_ALPHAS = 10.0 ** np.arange(-10, -2)  # (ridge coefficient of the standardized inputs)
LMS_RATES = [0.003, 0.01, 0.03, 0.1, 0.3]  # (normalized: c / mean ||x||^2)
EH_RATES = [0.03, 0.1, 0.3, 1.0]
EH_NOISE = 0.5  # (the paper's)
EPOCHS = [1, 3, 10, 30, 100, 300, 1000, 3000]  # (scores of the online rules after these numbers of epochs, the same run)
RLS_EPOCHS = [1]
RESULTS = f"{STUDY_PREFIX}readout_rules{os.environ.get('HAG_CANARY_RULES_SUFFIX', '')}"
PARTS = OUTPUTS / f"{RESULTS}_parts"


def song_data(reservoirs, songs, waves, readout, classes):
    """({wave: states of the decision of its points}, {wave: (training rows, one-hot next labels)}) of the readout
    (as canary.next_phrase_prediction.prediction.point_data, by song: the rows of a song in its time order)."""
    (points, tail), (decision, decision_tail) = training_of(readout), decision_of(readout)
    states, rows = {}, {}
    for w, windows in point_windows(reservoirs, songs, waves):
        def selection(mode, length):
            return [x[-1:] for x in windows] if mode == "last" else [x[-length:] for x in windows]
        states[w] = np.array([x.mean(axis=0) for x in selection(decision, decision_tail)])
        next_labels = (songs[w]["next"][:, None] == classes[None, :]).astype(np.float64)
        selected = selection(points, tail)
        if points == "tail":
            rows[w] = (np.vstack(selected), np.repeat(next_labels, [len(x) for x in selected], axis=0))
        else:
            rows[w] = (np.array([x.mean(axis=0) for x in selected]), next_labels)
    return states, rows


def configurations(rule):
    """[(hyperparameters, epochs)] of a rule: a run of the online rules for each hyperparameters, scored after each
    number of epochs."""
    if rule == "ridge":
        return [(dict(ridge=float(r)), [None]) for r in RIDGES]
    if rule == "rls":
        return [(dict(normalization=n, alpha=float(a)), RLS_EPOCHS) for n in NORMALIZATIONS for a in RLS_ALPHAS]
    if rule == "lms":
        return [(dict(normalization="standardize", rate=c), EPOCHS) for c in LMS_RATES]
    return [(dict(normalization="standardize", rate=c, noise=EH_NOISE), EPOCHS) for c in EH_RATES]


def normalized(hyper, rows, states, fit):
    """(rows, states) with the inputs normalized as hyper["normalization"] with the statistics of the rows of the songs
    fit (unchanged without normalization: ridge)."""
    if "normalization" not in hyper:
        return rows, states
    X = np.vstack([rows[w][0] for w in fit])
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-8 if hyper["normalization"] == "standardize" else 1.0
    return ({w: ((x - mean) / scale, y) for w, (x, y) in rows.items()},
            {w: (x - mean) / scale for w, x in states.items()})


def trained(rule, hyper, rows, fit, rng, epochs):
    """Yields (epochs, readout) of the rule trained on the rows of the songs fit: the ridge readout (epochs None), or
    the online readout after each number of epochs of epochs (the same node, trained further)."""
    if rule == "ridge":
        data = (np.vstack([rows[w][0] for w in fit]), np.vstack([rows[w][1] for w in fit]))
        yield None, next(fitted_readouts(data, [hyper["ridge"]]))[1]
        return
    mean_norm = np.mean(np.concatenate([np.sum(rows[w][0] ** 2, axis=1) for w in fit])) + 1  # (with the bias)
    if rule == "rls":
        node = RLS(alpha=hyper["alpha"])
    elif rule == "lms":
        node = LMS(learning_rate=hyper["rate"] / mean_norm)
    else:
        node = EH(learning_rate=hyper["rate"] / mean_norm, noise=hyper["noise"],
                  reward="per_output" if rule == "eh_np_local" else "global",
                  baseline="filtered" if rule == "eh" else "sample", seed=rng)
    # (JAX rules: the rows of all the songs fit, the songs in the order of rng at each epoch, as one pass over them)
    X, Y = (jnp.asarray(np.vstack([rows[w][i] for w in fit])) for i in (0, 1))
    starts = np.cumsum([0] + [len(rows[w][0]) for w in fit])
    with np.errstate(all="ignore"):  # (diverging rates: their scores are NaN, never chosen)
        for epoch in range(1, max(epochs) + 1):
            if rule == "rls":
                for w in rng.permutation(fit):
                    node.partial_fit(*rows[w])
            else:
                order = np.concatenate([np.arange(starts[s], starts[s + 1]) for s in rng.permutation(len(fit))])
                node.partial_fit(X, Y, order)
            if epoch in epochs:
                yield epoch, node


def accuracy_counts(scores, true, classes):
    with np.errstate(all="ignore"):
        return int((classes[np.nan_to_num(scores, nan=-np.inf).argmax(axis=1)] == true).sum())


def choose(model, params, size, songs, train_songs, frame, classes, readout, rules):
    """({rule: (hyperparameters with "epochs", out-of-fold accuracy)} of the best configuration of each rule, out-of-fold
    accuracy of each configuration and number of epochs), in the 3-fold cross-validation by song of the studies."""
    correct, total = {}, 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        all_states, all_rows = song_data(reservoirs, songs, fit + val, readout, classes)
        true = np.concatenate([songs[w]["next"] for w in val])
        for rule in rules:
            for c, (hyper, epochs) in enumerate(configurations(rule)):
                rng = np.random.default_rng([CV_SEED + k, c])
                rows, states = normalized(hyper, all_rows, all_states, fit)
                X_val = np.vstack([states[w] for w in val])
                for epoch, node in trained(rule, hyper, rows, fit, rng, epochs):
                    key = (rule, c, epoch)
                    correct[key] = correct.get(key, 0) + accuracy_counts(node.run(X_val), true, classes)
        total += len(true)
    chosen = {}
    for rule in rules:
        (_, c, epoch), count = max(((key, count) for key, count in correct.items() if key[0] == rule),
                                   key=lambda item: item[1])
        chosen[rule] = (dict(configurations(rule)[c][0], epochs=epoch), count / total)
    curves = pd.DataFrame([dict(rule=rule, **configurations(rule)[c][0], epochs=epoch, cv_accuracy=count / total)
                           for (rule, c, epoch), count in correct.items()])
    return chosen, curves


def run(todo, nb_seeds):
    """(results of the rules (RULES) for each condition and seed, their learning curves in the cross-validation)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    studies = pd.read_csv(OUTPUTS / f"{NAME}.csv")
    rows, curves = [], []
    for condition, model, params, _, size in todo:
        study = studies[(studies.condition == condition) & (studies.network_size == size) & studies.hpo.astype(bool)]
        if study.empty:
            raise ValueError(f"{condition} ({size}): no readout in {OUTPUTS / f'{NAME}.csv'}")
        points, decision, tail = study.iloc[0][["points", "decision", "tail"]]
        readout = readout_of(points, decision, None if pd.isna(tail) else int(tail))
        print(f"{condition} ({size}): readout {readout}", flush=True)
        chosen, curve = choose(model, params, size, songs, train_songs, frame, classes, readout, RULES)
        curves.append(curve.assign(condition=condition, network_size=size))
        for rule, (hyper, cv_accuracy) in chosen.items():
            print(f"  {rule}: {hyper}, cross-validation accuracy {100 * cv_accuracy:.2f} %", flush=True)
        for seed in range(nb_seeds):
            np.random.seed(seed)
            reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
            all_states, all_data = song_data(reservoirs, songs, train_songs + test_songs, readout, classes)
            true = np.concatenate([songs[w]["next"] for w in test_songs])
            for rule, (hyper, cv_accuracy) in chosen.items():
                config = {key: value for key, value in hyper.items() if key != "epochs"}
                data, states = normalized(config, all_data, all_states, train_songs)
                X_test = np.vstack([states[w] for w in test_songs])
                epochs = [hyper["epochs"]] if hyper["epochs"] is not None else [None]
                _, node = next(trained(rule, config, data, train_songs, np.random.default_rng([seed, 1]), epochs))
                with np.errstate(all="ignore"):
                    scores = np.nan_to_num(node.run(X_test), nan=0.0)
                rows.append(dict(condition=condition, network_size=size, seed=seed, rule=rule, points=points,
                                 decision=decision, tail=tail, **hyper, cv_accuracy=cv_accuracy,
                                 **metrics(scores, true, classes)))
            print(f"  seed {seed}: " + ", ".join(f"{r['rule']} {100 * r['accuracy']:.1f} %" for r in rows[-len(chosen):]),
                  flush=True)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def compare_rules(results, metric="accuracy"):
    """Paired comparisons by seed of the metric of each rule with that of ridge (rule - ridge), for each condition."""
    keys = ["condition", "network_size", "seed"]
    base = results[results.rule == "ridge"].set_index(keys)[metric]
    rows = []
    for (condition, size, rule), group in results[results.rule != "ridge"].groupby(["condition", "network_size", "rule"]):
        diff = (group.set_index(keys)[metric] - base).dropna().to_numpy()
        if len(diff):
            rows.append(dict(condition=condition, network_size=size, rule=rule, nb_seeds=len(diff),
                             mean=group[metric].mean(), mean_diff=diff.mean(),
                             wilcoxon_p=wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0,
                             a_better=int((diff > 0).sum())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if not TAIL_HPO or TAIL_SEPARATE:
        raise RuntimeError("HAG_CANARY_TASK=next_phrase HAG_CANARY_TAIL_HPO=1 is needed (studies and readouts)")
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "list":
        for condition, _, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            if not condition.endswith("_proj"):
                print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size] if size else []) if c[0] == condition and c[4] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        PARTS.mkdir(parents=True, exist_ok=True)
        results, curves = run(todo, nb_seeds)
        results.to_csv(PARTS / f"{condition}_{size}.csv", index=False)
        curves.to_csv(PARTS / f"{condition}_{size}_cv.csv", index=False)
    elif command == "merge":
        parts = sorted(part for part in PARTS.glob("*.csv") if not part.stem.endswith("_cv"))
        print(f"{len(parts)} conditions")
        results = pd.concat([pd.read_csv(part) for part in parts], ignore_index=True)
        results.to_csv(OUTPUTS / f"{RESULTS}.csv", index=False)
        curve_parts = sorted(PARTS.glob("*_cv.csv"))
        if curve_parts:
            pd.concat([pd.read_csv(part) for part in curve_parts], ignore_index=True).to_csv(
                OUTPUTS / f"{RESULTS}_cv.csv", index=False)
        comparisons = compare_rules(results)
        pd.set_option("display.width", 200)
        print(comparisons.round(4).to_string(index=False))
        comparisons.to_csv(OUTPUTS / f"{RESULTS}_comparisons.csv", index=False)
        print("Saved to", OUTPUTS / f"{RESULTS}.csv", "and", OUTPUTS / f"{RESULTS}_comparisons.csv")
    else:
        raise ValueError(f"Unknown command {command!r}: run, merge or list")
