"""
Online learning rules of the readout on selected monomials of degree 2: the selections of
canopy.next_phrase_prediction.monomial_selection (z, the standardized last state of each prediction point, and a few
hundred or thousand standardized products chosen before the training), read by the online rules of
canopy.next_phrase_prediction.readout_rules instead of the ridge regression. With few products, the rules learn their
weights themselves (primal form), without the kernel of the training rows of "nlms:2" and "neh:2", and RLS, whose
matrix has the size of the inputs, becomes possible. Exploratory, as monomial_selection: all the rules and methods read
the same reservoirs (those of the studies of the variables of the environment, HAG_CANARY_HPO_RULE=ridge:2), not
reservoirs optimized for each.

Rules (HAG_CANARY_SELECTION_RULES, default nlms, eh_np_local, rls): those of readout_rules.trained, on the inputs of
each method (HAG_CANARY_SELECTION_METHODS):
  - "linear": z alone (the rule on the states);
  - "full_std": all the products of z, in dual form (the rules "nlms:2" and "neh:2" on z: only nlms and neh, the dual
    form being always normalized; not RLS, whose matrix has the size of the inputs);
  - the selections of monomial_selection ("weights_high:m", "residual:M", "pca:m", "pca_frames:m", "random_pairs:M",
    "scores", ...): z and the selected products, in primal form.
The linear readout of the selections (weights, residual, scores) is the ridge one, its coefficient chosen as in
monomial_selection.

Protocol of readout_rules: the hyperparameters of each rule and method (learning rate of LMS, NLMS and of the versions
of EH: the grids of readout_rules; noise of the versions of EH: NEH_NOISES; alpha of RLS: RLS_ALPHAS; number of
epochs, at most HAG_CANARY_RULES_MAX_EPOCHS, default 1000) chosen by the 3-fold cross-validation by song on the train songs, then each rule and method trained on
all the train songs and scored on the test songs for each seed.

Results: outputs/canary/<prefix>monomial_selection_rules.csv (one row per condition, size, seed, rule and method,
with the seconds of its training) and
<prefix>monomial_selection_rules_cv.csv (out-of-fold accuracy of each configuration and number of epochs).
Usage (HAG_CANARY_TASK=next_phrase HAG_CANARY_HPO_RULE=ridge:2, with the variables of the studies):
  python -m canopy.next_phrase_prediction.monomial_selection_rules run <condition> <size> [nb_seeds]
  python -m canopy.next_phrase_prediction.monomial_selection_rules merge
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from canopy.common import OUTPUTS, load_dataset, pretrain_phrases
from canopy.hpo_utils import CV_SEED, FOLD_SEED
from canopy.models.reservoirs import build
from canopy.next_phrase_prediction.hpo import N_FOLDS, STUDY_PREFIX
from canopy.next_phrase_prediction.monomial_selection import RIDGES_STD, Selection, fold_counts, reservoir_data, stacked
from canopy.next_phrase_prediction.prediction import all_conditions, metrics, prepare
from canopy.next_phrase_prediction.readout_rules import NEH_NOISES, configurations, trained

RULES = os.environ.get("HAG_CANARY_SELECTION_RULES", "nlms,eh_np_local,rls").split(",")
METHODS = os.environ.get("HAG_CANARY_SELECTION_METHODS", "linear,full_std,scores,weights_high:64,residual:2080,pca:64,"
                         "pca_frames:64,random_pairs:2080").split(",")
RLS_ALPHAS = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]  # (initial P = I / alpha, on the standardized inputs)
MAX_EPOCHS = int(os.environ.get("HAG_CANARY_RULES_MAX_EPOCHS", "1000"))
JOBS = int(os.environ.get("HAG_CANARY_RULES_JOBS", "1"))  # processes training the configurations
RESULTS = f"{STUDY_PREFIX}monomial_selection_rules{os.environ.get('HAG_CANARY_SELECTION_SUFFIX', '')}"
PARTS = OUTPUTS / f"{RESULTS}_parts"


def rule_configurations(rule):
    """[(hyperparameters, epochs)] of a rule (the inputs are already standardized)."""
    if rule == "rls":
        return [(dict(alpha=alpha), [1]) for alpha in RLS_ALPHAS]
    noises = [None] if rule in ("lms", "nlms", "neh") else NEH_NOISES  # (neh: its own grid of noises, the same)
    return [({key: value for key, value in hyper.items() if key != "normalization"}, epochs)
            for noise in noises for hyper, epochs in configurations(rule, max_epochs=MAX_EPOCHS, noise=noise)]


def pairs(methods=METHODS, rules=RULES):
    """(method, rule) trained: all the products (dual form, always normalized) for nlms and neh only."""
    return [(method, rule) for method in methods for rule in rules if method != "full_std" or rule in ("nlms", "neh")]


def rule_inputs(selection, method):
    """(inputs of the training rows, function of the states giving the inputs, degree of the monomials of the inputs
    read by the rule: 2 for all the products in dual form)."""
    if method in ("linear", "full_std"):
        return selection.Z, selection.standardized, 2 if method == "full_std" else 1
    F, inputs = selection.features(method)
    return F, inputs, 1


def trained_readouts(selection, waves, lengths, method, rule, hyper, epochs, rng):
    """Yields (epochs, readout, function of the states giving its inputs, number of inputs) of the rule trained on the
    inputs of the method of the training rows (those of the songs waves, lengths rows each), as readout_rules.trained."""
    F, inputs, order = rule_inputs(selection, method)
    bounds = np.cumsum([0] + list(lengths))
    rows = {w: (F[bounds[i]:bounds[i + 1]], selection.Y[bounds[i]:bounds[i + 1]]) for i, w in enumerate(waves)}
    n = F.shape[1]
    for epoch, readout in trained(rule, hyper, rows, list(waves), rng, epochs, order):
        yield epoch, readout, inputs, n + n * (n + 1) // 2 if order == 2 else n


def timed(function, *args):
    """(result, seconds) of function(*args)."""
    start = time.time()
    return function(*args), time.time() - start


def configuration_counts(data, fit, val, true, classes, rho_linear, method, rule, c, hyper, epochs, fold):
    """{epochs: correct predictions on the songs val} of a configuration trained on the songs fit."""
    X, Y, moments = data
    selection = Selection(*stacked(X, Y, fit), moments, rho_linear, seed=CV_SEED + fold)
    X_val, F_val, counts = np.vstack([X[w] for w in val]), None, {}
    rng = np.random.default_rng([CV_SEED + fold, c])
    with np.errstate(all="ignore"):
        for epoch, readout, inputs, _ in trained_readouts(selection, fit, [len(X[w]) for w in fit], method, rule, hyper,
                                                          epochs, rng):
            if F_val is None:
                F_val = inputs(X_val)
            scores = np.nan_to_num(np.asarray(readout.run(F_val)), nan=-np.inf)
            counts[epoch] = int(np.sum(classes[scores.argmax(axis=1)] == true))
    return counts


def choose(model, params, size, songs, train_songs, frame, classes, methods=METHODS, rules=RULES):
    """({(method, rule): (hyperparameters with "epochs", out-of-fold accuracy, seconds of all its configurations)},
    out-of-fold accuracy of each configuration and number of epochs, coefficient of the linear readout of the
    selections), in the 3-fold cross-validation by song of the studies."""
    folds = []
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        data = reservoir_data(reservoirs, songs, fit + val, classes, fit)
        folds.append((data, fit, val, np.concatenate([songs[w]["next"] for w in val])))
    total = sum(len(true) for *_, true in folds)
    linear = sum(fold_counts(data, fit, val, true, classes, 1.0, ["linear"], k)["linear"]
                 for k, (data, fit, val, true) in enumerate(folds))
    rho_linear = float(RIDGES_STD[linear.argmax()])
    tasks = [(k, method, rule, c, hyper, epochs) for k in range(len(folds)) for method, rule in pairs(methods, rules)
             for c, (hyper, epochs) in enumerate(rule_configurations(rule))]
    counts = Parallel(n_jobs=min(JOBS, len(tasks)))(
        delayed(timed)(configuration_counts, *folds[k], classes, rho_linear, method, rule, c, hyper, epochs, k)
        for k, method, rule, c, hyper, epochs in tasks)
    correct, seconds = {}, {}
    for (k, method, rule, c, hyper, epochs), (count, duration) in zip(tasks, counts):
        seconds[(method, rule)] = seconds.get((method, rule), 0) + duration
        for epoch, value in count.items():
            correct[(method, rule, c, epoch)] = correct.get((method, rule, c, epoch), 0) + value
    chosen, curves = {}, []
    for method, rule in pairs(methods, rules):
        configs = rule_configurations(rule)
        (c, epoch), count = max((((c, epoch), count) for (m, r, c, epoch), count in correct.items()
                                 if (m, r) == (method, rule)), key=lambda item: item[1])
        chosen[(method, rule)] = (dict(configs[c][0], epochs=epoch), count / total, seconds[(method, rule)])
        curves += [dict(method=method, rule=rule, **configs[c][0], epochs=epoch, cv_accuracy=count / total)
                   for (m, r, c, epoch), count in correct.items() if (m, r) == (method, rule)]
    return chosen, pd.DataFrame(curves), rho_linear


def seed_metrics(data, train_songs, test_songs, true, classes, rho_linear, method, rule, hyper, seed):
    """dict(inputs, metrics on the test songs) of the rule trained on the inputs of the method of the train songs."""
    X, Y, moments = data
    selection = Selection(*stacked(X, Y, train_songs), moments, rho_linear, seed=seed)
    config = {key: value for key, value in hyper.items() if key != "epochs"}
    with np.errstate(all="ignore"):
        _, readout, inputs, n_inputs = next(trained_readouts(
            selection, train_songs, [len(X[w]) for w in train_songs], method, rule, config, [hyper["epochs"]],
            np.random.default_rng([seed, 1])))
        scores = np.nan_to_num(np.asarray(readout.run(inputs(np.vstack([X[w] for w in test_songs])))), nan=0.0)
    return dict(inputs=n_inputs, **metrics(scores, true, classes))


def run(todo, nb_seeds, methods=METHODS, rules=RULES):
    """(results of the rules on the methods for each condition and seed, their cross-validation)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    results, curves = [], []
    for condition, model, params, _, size in todo:
        keys = dict(condition=condition, network_size=size)
        chosen, curve, rho_linear = choose(model, params, size, songs, train_songs, frame, classes, methods, rules)
        curves.append(curve.assign(**keys))
        print(f"{condition} ({size}): linear readout of the selections {rho_linear:.0e}", flush=True)
        for (method, rule), (hyper, accuracy, seconds) in chosen.items():
            print(f"  {rule} on {method}: {hyper}, cross-validation accuracy {100 * accuracy:.2f} %, {seconds:.0f} s",
                  flush=True)
        true = np.concatenate([songs[w]["next"] for w in test_songs])
        seed_data = []
        for seed in range(nb_seeds):
            np.random.seed(seed)
            reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
            seed_data.append(reservoir_data(reservoirs, songs, train_songs + test_songs, classes, train_songs))
        tasks = [(seed, method, rule) for seed in range(nb_seeds) for method, rule in chosen]
        outputs = Parallel(n_jobs=min(JOBS, len(tasks)))(
            delayed(timed)(seed_metrics, seed_data[seed], train_songs, test_songs, true, classes, rho_linear, method,
                           rule, chosen[(method, rule)][0], seed)
            for seed, method, rule in tasks)
        for (seed, method, rule), (output, seconds) in zip(tasks, outputs):
            hyper, accuracy, _ = chosen[(method, rule)]
            results.append(dict(**keys, seed=seed, rule=rule, method=method, **hyper, cv_accuracy=accuracy, **output,
                                seconds=seconds))
        for seed in range(nb_seeds):
            print(f"  seed {seed}: " + ", ".join(
                f"{r['rule']} on {r['method']} {100 * r['accuracy']:.1f} % ({r['seconds']:.0f} s)"
                for r in results[-len(tasks):] if r["seed"] == seed), flush=True)
    return pd.DataFrame(results), pd.concat(curves, ignore_index=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size]) if c[0] == condition and c[4] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        PARTS.mkdir(parents=True, exist_ok=True)
        for table, suffix in zip(run(todo, nb_seeds), ("", "_cv")):
            table.to_csv(PARTS / f"{condition}_{size}{suffix}.csv", index=False)
    elif command == "merge":
        for suffix in ("", "_cv"):
            parts = sorted(part for part in PARTS.glob("*.csv") if part.stem.endswith("_cv") == bool(suffix))
            if parts:
                pd.concat([pd.read_csv(part) for part in parts], ignore_index=True).to_csv(
                    OUTPUTS / f"{RESULTS}{suffix}.csv", index=False)
                print(f"{len(parts)} conditions: {OUTPUTS / f'{RESULTS}{suffix}.csv'}")
    else:
        raise ValueError(f"Unknown command {command!r}: run or merge")
