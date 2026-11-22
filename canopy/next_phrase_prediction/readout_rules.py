"""
Learning rules of the readout of the next-phrase prediction (canopy.next_phrase_prediction.prediction), compared on the
same reservoirs: from the least to the most biologically plausible,
  - "ridge": ridge regression on all the training rows at once (reservoirpy's Ridge, solver RIDGE_SOLVER), the
    reference of the other tasks;
  - "rls": online recursive least squares (canopy.models.readout_rules.rls: reservoirpy's RLS on the inputs and a
    constant input 1 for the bias, the RLS-based FORCE rule of Sussillo & Abbott 2009, non-local: each synapse uses
    the inverse correlation matrix of all the inputs), initial P = I / alpha; after one epoch, the weights of the
    ridge regression of coefficient alpha (the bias of reservoirpy's node, a running mean of the errors, used until
    2026-11-19, gave much worse readouts with a small alpha);
  - "lms": online least mean squares (the rule of reservoirpy's LMS, the LMS-based FORCE rule: local, but each readout
    needs its exact error; trained in JAX, canopy.models.readout_rules.lms), learning rate c / mean ||x||^2 (c the normalized rate,
    stable below 2): LMS of Widrow & Hoff, a constant learning rate (the mean computed once on the training rows);
  - "eh": the exploratory Hebbian rule of Hoerzer et al. (2014) (the rule of canopy.models.readout_rules.eh_numpy.EH, the paper's: filtered
    baselines, a single global binary reward for all the readouts, no teacher; trained in JAX, canopy.models.readout_rules.eh),
    learning rate c / mean ||x||^2;
  - "eh_np": the same with the baseline "sample" (node perturbation of Fiete & Seung 2006: the reward compared with
    that of the output without noise on the same input; not the paper's rule, which does not know the noise), global
    reward; "eh_np_local": with a reward for each readout (its own modulatory signal);
  - "nlms" and "neh": LMS and eh_np_local with the learning rate of each row normalized by its squared norm (NLMS,
    learning rate c / (||x||^2 + 1)), needed on the monomials (order 2) whose norms spread much more than those of
    the states. "nlms" is NLMS (Nagumo & Noda 1967, Albert & Gardner 1967; as a reservoir readout: Yuenyong 2014, and
    the step of BPDC, Steil 2004); "eh_np", "eh_np_local" and "neh" are ours (closest published rule of neh: the
    decorrelated rule of Burms, Caluwaerts & Dambre 2015). Which version is which: canopy.models.readout_rules.
Each rule can read the monomials of the state up to degree n instead of the state ("<rule>:<n>", e.g. "nlms:2"; the
features of the generalized readout of Ohkubo & Inubushi 2024): "ridge:n" the generalized readout
(canopy.models.generalized_readout), "nlms:n" and "neh:n" trained in their dual form on the kernel of the monomials
(canopy.models.readout_rules.kernel, the same weights as on the monomials themselves).
Each rule can also read the state and selected products of degree 2 ("<rule>:<selection>", a selection of
canopy.next_phrase_prediction.monomial_selection chosen on the training rows before the training of the readout, e.g.
"nlms:pca:64": the 2080 products of the 64 first principal components; "eh_np_local:scores": the 435 products of the
outputs of the linear ridge readout; "ridge:random_pairs:2080"): inputs standardized by the selection (the state and
each product), in primal form; the linear readout of the selections "scores", "weights_high", "weights_low" and "residual" with its ridge
coefficient chosen on the training rows (Selection.chosen_ridge; nested for the out-of-fold outputs of the training
rows of "scores" and "residual": chosen without the fold of each row).
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
next_phrase_calls_tail_hpo_readouts.csv), or (HAG_CANARY_TAIL_HPO=0) those of the studies of the reservoirs with the
linear ridge readout, the readout "last" -> "last". The reservoirs are not optimized again for the rules: all the
rules read the same reservoirs (fold k of the cross-validation: drawn with the seed CV_SEED + k; seed s of the
evaluation: drawn with the seed s), and only the hyperparameters of each rule (ridge coefficient, alpha, normalized
learning rate, noise, number of epochs) are chosen, in the 3-fold cross-validation by song of the studies on the train
songs (out-of-fold accuracy; never on the test songs); then each rule is trained on all the train songs and scored on
the test songs for each seed (paired comparisons with "ridge"). Rules: HAG_CANARY_READOUT_RULES (comma-separated,
default all the rules on the states), their configurations trained in HAG_CANARY_RULES_JOBS processes (default 1),
for at most HAG_CANARY_RULES_MAX_EPOCHS epochs (default the last of EPOCHS).
These comparisons read reservoirs optimized for the ridge readout: they favor it (CLAUDE.md, fair comparisons). For a
fair comparison, each rule with its own studies: HAG_CANARY_HPO_RULE=<rule> (canopy.next_phrase_prediction.hpo,
reservoirs optimized for the rule, slurm/submit_hpo_rules.sh), then its evaluation here with the same variable: the
reservoirs of its studies, that rule alone, its rate, noise and alpha of the study (the hybrids, without study: those
of the grid), its number of epochs (and normalization of RLS, coefficient of ridge:<n>) chosen by the cross-validation.

Results: outputs/canary/<prefix>readout_rules<suffix>.csv (one row per condition, size, seed and rule),
<prefix>readout_rules<suffix>_comparisons.csv (each rule - ridge, paired by seed) and <prefix>readout_rules<suffix>_cv.csv
(out-of-fold accuracy of each configuration and number of epochs: the learning curves), suffix
HAG_CANARY_RULES_SUFFIX (default none).
Usage (HAG_CANARY_TASK=next_phrase HAG_CANARY_TAIL_HPO=1, with the variables of the studies):
  python -m canopy.next_phrase_prediction.readout_rules run <condition> <size> [nb_seeds]   one condition
                                                       (slurm/submit_readout_rules.sh), results in <name>_parts
  python -m canopy.next_phrase_prediction.readout_rules merge          results of the parts and the comparisons
  python -m canopy.next_phrase_prediction.readout_rules list <sizes>   "<condition> <size>" of the conditions (no _proj)
"""
import os
import sys

import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.stats import wilcoxon
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from canopy.common import OUTPUTS, RIDGES, load_dataset, pretrain_phrases
from canopy.hpo_utils import CV_SEED, FOLD_SEED
from canopy.models.generalized_readout import GeneralizedRidge
from canopy.models.readout_rules.eh import EH
from canopy.models.readout_rules.eh_numpy import version_options
from canopy.models.readout_rules.kernel import KernelEH, KernelLMS
from canopy.models.readout_rules.lms import LMS
from canopy.models.readout_rules.rls import RLS
from canopy.models.reservoirs import build
from canopy.next_phrase_prediction.hpo import HPO_RULE, N_FOLDS, STUDY_PREFIX, TAIL_HPO, TAIL_SEPARATE
from canopy.next_phrase_prediction.monomial_selection import RIDGES_STD, Selection
from canopy.next_phrase_prediction.prediction import (BASE_READOUT, NAME, all_conditions, decision_of, fitted_readouts,
                                                      metrics, point_windows, prepare, readout_of, training_of)

RULES = os.environ.get("HAG_CANARY_READOUT_RULES", "ridge,rls,lms,eh,eh_np,eh_np_local").split(",")
if HPO_RULE:  # (the studies of the rule, canopy.next_phrase_prediction.hpo: that rule alone)
    RULES = [HPO_RULE]
NORMALIZATIONS = ["center", "standardize"]  # (inputs of RLS; LMS and EH: "standardize", better in all the conditions)
RLS_ALPHAS = 10.0 ** np.arange(-10, -2)  # (ridge coefficient of the standardized inputs)
LMS_RATES = [0.003, 0.01, 0.03, 0.1, 0.3]  # (normalized: c / mean ||x||^2)
EH_RATES = [0.03, 0.1, 0.3, 1.0]
EH_NOISE = 0.5  # (the paper's)
NLMS_RATES = [0.03, 0.1, 0.3]  # (normalized per row: c / (||x||^2 + 1))
NEH_RATES = [0.01, 0.03, 0.1]
NEH_NOISES = [0.5, 1.5]
NORMALIZED_RULES = ("nlms", "neh")  # (the rules which can read the monomials of the state: order > 1, with ridge)
JOBS = int(os.environ.get("HAG_CANARY_RULES_JOBS", "1"))  # processes training the configurations
EPOCHS = [1, 3, 10, 30, 100, 300, 1000, 3000]  # (scores of the online rules after these numbers of epochs, the same run)
RLS_EPOCHS = [1]
MAX_EPOCHS = int(os.environ.get("HAG_CANARY_RULES_MAX_EPOCHS", EPOCHS[-1]))  # (of the online rules, at most: those of
# EPOCHS up to it)
RESULTS = f"{STUDY_PREFIX}readout_rules{os.environ.get('HAG_CANARY_RULES_SUFFIX', '')}"
PARTS = OUTPUTS / f"{RESULTS}_parts"


def song_data(reservoirs, songs, waves, readout, classes, frame_songs=()):
    """({wave: states of the decision of its points}, {wave: (training rows, one-hot next labels)}, moments) of the
    readout (as canopy.next_phrase_prediction.prediction.point_data, by song: the rows of a song in its time order);
    moments: (number, sum (N,), Gram matrix (N, N)) of the states of all the frames of the windows of the songs
    frame_songs (the selection "pca_frames"), None without frame_songs."""
    (points, tail), (decision, decision_tail) = training_of(readout), decision_of(readout)
    states, rows, frame_songs = {}, {}, set(frame_songs)
    n, total, gram = 0, 0.0, 0.0
    for w, windows in point_windows(reservoirs, songs, waves):
        if w in frame_songs:
            frames = np.vstack(windows).astype(np.float64)
            n, total, gram = n + len(frames), total + frames.sum(axis=0), gram + frames.T @ frames
        def selection(mode, length):
            return [x[-1:] for x in windows] if mode == "last" else [x[-length:] for x in windows]
        states[w] = np.array([x.mean(axis=0) for x in selection(decision, decision_tail)])
        next_labels = (songs[w]["next"][:, None] == classes[None, :]).astype(np.float64)
        selected = selection(points, tail)
        if points == "tail":
            rows[w] = (np.vstack(selected), np.repeat(next_labels, [len(x) for x in selected], axis=0))
        else:
            rows[w] = (np.array([x.mean(axis=0) for x in selected]), next_labels)
    return states, rows, (n, total, gram) if frame_songs else None


def frame_songs_of(rules, fit):
    """Songs whose frames give the moments of song_data: fit if a rule reads the selection "pca_frames"."""
    return fit if any((rule_inputs(label)[2] or "").startswith("pca_frames") for label in rules) else ()


def configurations(rule, rate=None, max_epochs=None, noise=None, alpha=None, selection=None):
    """[(hyperparameters, epochs)] of a rule: a run of the online rules for each hyperparameters, scored after each
    number of epochs of EPOCHS (at most max_epochs) (LMS and EH: the learning rates of the grid, or the single one
    rate; EH: the exploration noise EH_NOISE, or noise; RLS: the grid of NORMALIZATIONS and RLS_ALPHAS, or the single
    alpha with centered inputs). With a selection of products (inputs already standardized): no normalization, the
    ridge coefficients RIDGES_STD."""
    if selection:
        if rule == "ridge":
            return [(dict(ridge=float(r)), [None]) for r in RIDGES_STD]
        unique = {}  # (the configurations which differ only by the normalization: once)
        for hyper, epochs in configurations(rule, rate, max_epochs, noise, alpha):
            unique.setdefault(tuple((k, v) for k, v in hyper.items() if k != "normalization"), epochs)
        return [(dict(hyper), epochs) for hyper, epochs in unique.items()]
    epochs = [e for e in EPOCHS if max_epochs is None or e <= max_epochs]
    if rule == "ridge":
        return [(dict(ridge=float(r)), [None]) for r in RIDGES]
    if rule == "rls":
        if alpha:
            return [(dict(normalization="center", alpha=float(alpha)), RLS_EPOCHS)]
        return [(dict(normalization=n, alpha=float(a)), RLS_EPOCHS) for n in NORMALIZATIONS for a in RLS_ALPHAS]
    if rule == "lms":
        return [(dict(normalization="standardize", rate=c), epochs) for c in ([rate] if rate else LMS_RATES)]
    if rule == "nlms":
        return [(dict(normalization="standardize", rate=c), epochs) for c in ([rate] if rate else NLMS_RATES)]
    if rule == "neh":
        return [(dict(normalization="standardize", rate=c, noise=v), epochs) for c in ([rate] if rate else NEH_RATES) for v in ([noise] if noise else NEH_NOISES)]
    return [(dict(normalization="standardize", rate=c, noise=noise or EH_NOISE), epochs)
            for c in ([rate] if rate else EH_RATES)]


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


def rule_inputs(label):
    """(rule, order, selection) of a rule of HAG_CANARY_READOUT_RULES: "<rule>" (order 1, the states), "<rule>:<order>"
    (the monomials of the states up to degree order: ridge and NORMALIZED_RULES) or "<rule>:<selection>" (order 1, the
    states and the products of a selection of monomial_selection, e.g. "nlms:pca:64"; selection None otherwise)."""
    rule, _, inputs = label.partition(":")
    if inputs and not inputs.isdigit():
        return rule, 1, inputs
    order = int(inputs or 1)
    if order > 1 and rule != "ridge" and rule not in NORMALIZED_RULES:
        raise ValueError(f"{label}: only ridge and {', '.join(NORMALIZED_RULES)} read the monomials (order > 1)")
    return rule, order, None


def rule_order(label):
    """(rule, order) of a rule (rule_inputs)."""
    return rule_inputs(label)[:2]


def label_configurations(label, rate=None, max_epochs=None, noise=None, alpha=None):
    """configurations of the rule of a label of HAG_CANARY_READOUT_RULES."""
    rule, _, selection = rule_inputs(label)
    return configurations(rule, rate, max_epochs, noise, alpha, selection)


_selected = {}  # (the inputs of the last selection built: its key and themselves)


def selected_rows(selection, all_rows, fit, moments, seed):
    """({wave: (inputs of its rows, labels)} of the songs fit, function of the states giving their inputs) of a
    selection of monomial_selection chosen on the rows of the songs fit: the standardized states and products."""
    X, Y = (np.vstack([all_rows[w][i] for w in fit]) for i in (0, 1))
    key = (selection, seed, X.shape, float(X.sum()), float(X[:, 0] @ X[:, -1]))
    if _selected.get("key") != key:
        lengths = [len(all_rows[w][0]) for w in fit]
        groups = np.repeat(np.arange(len(fit)), lengths)
        F, inputs = Selection(X, Y, groups, moments, None, seed=seed).features(selection)
        bounds = np.cumsum([0] + lengths)
        _selected.update(key=key, value=({w: (F[bounds[i]:bounds[i + 1]], all_rows[w][1]) for i, w in enumerate(fit)}, inputs))
    return _selected["value"]


_generalized = {}  # (the last generalized ridge readout fitted, solved again for each coefficient: its key and itself)


def trained(rule, hyper, rows, fit, rng, epochs, order=1, selected=False):
    """Yields (epochs, readout) of the rule trained on the rows of the songs fit (order > 1: on the monomials of the
    rows up to degree order): the ridge readout (epochs None), or the online readout after each number of epochs of
    epochs (the same node, trained further), until it diverges (non-finite weights: stopped after the next number of
    epochs, its scores NaN). selected: the rows of a selection (ridge: a single fit for all the coefficients)."""
    if rule == "ridge":
        data = (np.vstack([rows[w][0] for w in fit]), np.vstack([rows[w][1] for w in fit]))
        if order == 1 and not selected:
            yield None, next(fitted_readouts(data, [hyper["ridge"]], order))[1]
            return
        key = (order, data[0].shape, float(data[0].sum()), float(data[0][:, 0] @ data[0][:, -1]))
        if _generalized.get("key") != key:  # (a single fit for all the coefficients of the same rows)
            _generalized.update(key=key, readout=GeneralizedRidge(order=order).fit(*data))
        yield None, _generalized["readout"].set_ridge(hyper["ridge"])
        return
    mean_norm = np.mean(np.concatenate([np.sum(rows[w][0] ** 2, axis=1) for w in fit])) + 1  # (with the bias)
    # (JAX rules: the rows of all the songs fit, the songs in the order of rng at each epoch, as one pass over them)
    X, Y = (np.vstack([rows[w][i] for w in fit]) for i in (0, 1))
    if order > 1:  # (the dual form, on the kernel of the monomials of the rows, always normalized)
        if rule == "nlms":
            node = KernelLMS(X, order, hyper["rate"])
        else:
            options = {k: v for k, v in version_options(rule).items() if k != "normalized"}
            node = KernelEH(X, order, hyper["rate"], noise=hyper["noise"], seed=rng, **options)
    elif rule == "rls":
        node = RLS(alpha=hyper["alpha"])
    elif rule in ("lms", "nlms"):
        node = LMS(learning_rate=hyper["rate"] / mean_norm) if rule == "lms" else LMS(hyper["rate"], normalized=True)
    else:  # (the versions of EH; the rate of the unnormalized ones divided by the mean squared norm, as lms)
        options = version_options(rule)
        rate = hyper["rate"] if options["normalized"] else hyper["rate"] / mean_norm
        node = EH(learning_rate=rate, noise=hyper["noise"], seed=rng, **options)
    X, Y = jnp.asarray(X), jnp.asarray(Y)
    starts = np.cumsum([0] + [len(rows[w][0]) for w in fit])
    with np.errstate(all="ignore"):  # (diverging rates: their scores are NaN, never chosen)
        for epoch in range(1, max(epochs) + 1):
            if rule == "rls":
                for w in rng.permutation(fit):
                    node.partial_fit(*rows[w])
            else:
                order_rows = np.concatenate([np.arange(starts[s], starts[s + 1]) for s in rng.permutation(len(fit))])
                if order > 1:
                    node.partial_fit(Y, order_rows)
                else:
                    node.partial_fit(X, Y, order_rows)
            if epoch in epochs:
                yield epoch, node
                if not np.all(np.isfinite(node.Wout)):
                    return


def accuracy_counts(scores, true, classes):
    with np.errstate(all="ignore"):
        return int((classes[np.nan_to_num(scores, nan=-np.inf).argmax(axis=1)] == true).sum())


def configuration_counts(tasks, all_rows, all_states, fit, val, true, classes, fold, moments=None):
    """{(label, configuration, epochs): correct predictions on the songs val} of the configurations tasks [(label,
    configuration index, hyperparameters, epochs)] trained on the songs fit (one process: the ridge configurations of
    a label together, their generalized readout fitted once)."""
    correct = {}
    for label, c, hyper, epochs in tasks:
        rule, order, selection = rule_inputs(label)
        rng = np.random.default_rng([CV_SEED + fold, c])
        if selection:
            rows, inputs = selected_rows(selection, all_rows, fit, moments, CV_SEED + fold)
            X_val = inputs(np.vstack([all_states[w] for w in val]))
        else:
            rows, states = normalized(hyper, all_rows, all_states, fit)
            X_val = np.vstack([states[w] for w in val])
        for epoch, node in trained(rule, hyper, rows, fit, rng, epochs, order, bool(selection)):
            correct[(label, c, epoch)] = accuracy_counts(node.run(X_val), true, classes)
    return correct


def choose(model, params, size, songs, train_songs, frame, classes, readout, rules, rate=None, max_epochs=None,
           noise=None, alpha=None):
    """({rule: (hyperparameters with "epochs", out-of-fold accuracy)} of the best configuration of each rule (labels of
    HAG_CANARY_READOUT_RULES), out-of-fold accuracy of each configuration and number of epochs), in the 3-fold
    cross-validation by song of the studies on the train songs (the configurations of configurations(rule, rate,
    max_epochs, noise, alpha); all the rules on the same reservoirs, drawn with the seed CV_SEED + k for fold k)."""
    correct, total = {}, 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        all_states, all_rows, moments = song_data(reservoirs, songs, fit + val, readout, classes,
                                                  frame_songs_of(rules, fit))
        true = np.concatenate([songs[w]["next"] for w in val])
        groups = []  # (the configurations of a ridge rule together, one group per configuration of the online rules)
        for label in rules:
            tasks = [(label, c, hyper, epochs) for c, (hyper, epochs)
                     in enumerate(label_configurations(label, rate, max_epochs, noise, alpha))]
            groups += [tasks] if rule_order(label)[0] == "ridge" else [[task] for task in tasks]
        for counts in Parallel(n_jobs=min(JOBS, len(groups)))(
                delayed(configuration_counts)(tasks, all_rows, all_states, fit, val, true, classes, k, moments)
                for tasks in groups):
            for key, count in counts.items():
                correct[key] = correct.get(key, 0) + count
        total += len(true)
    chosen = {}
    for rule in rules:
        (_, c, epoch), count = max(((key, count) for key, count in correct.items() if key[0] == rule),
                                   key=lambda item: item[1])
        configs = label_configurations(rule, rate, max_epochs, noise, alpha)
        chosen[rule] = (dict(configs[c][0], epochs=epoch), count / total)
    curves = pd.DataFrame([dict(rule=rule, **label_configurations(rule, rate, max_epochs, noise, alpha)[c][0],
                                epochs=epoch, cv_accuracy=count / total)
                           for (rule, c, epoch), count in correct.items()])
    return chosen, curves


def seed_scores(label, hyper, all_data, all_states, train_songs, test_songs, true, classes, seed, moments=None):
    """Metrics on the test songs of the rule label with its chosen hyperparameters hyper, trained on the train songs
    (the reservoirs of the seed)."""
    rule, order, selection = rule_inputs(label)
    config = {key: value for key, value in hyper.items() if key != "epochs"}
    if selection:
        data, inputs = selected_rows(selection, all_data, train_songs, moments, seed)
        X_test = inputs(np.vstack([all_states[w] for w in test_songs]))
    else:
        data, states = normalized(config, all_data, all_states, train_songs)
        X_test = np.vstack([states[w] for w in test_songs])
    epochs = [hyper["epochs"]] if hyper["epochs"] is not None else [None]
    _, node = next(trained(rule, config, data, train_songs, np.random.default_rng([seed, 1]), epochs, order,
                           bool(selection)))
    with np.errstate(all="ignore"):
        return metrics(np.nan_to_num(node.run(X_test), nan=0.0), true, classes)


def run(todo, nb_seeds):
    """(results of the rules (RULES) for each condition and seed, their learning curves in the cross-validation)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    studies = pd.read_csv(OUTPUTS / f"{NAME}.csv") if TAIL_HPO else None
    rows, curves = [], []
    for condition, model, params, _, size in todo:
        if TAIL_HPO:  # (the readout of the study of the readout of the condition)
            study = studies[(studies.condition == condition) & (studies.network_size == size) & studies.hpo.astype(bool)]
            if study.empty:
                raise ValueError(f"{condition} ({size}): no readout in {OUTPUTS / f'{NAME}.csv'}")
            points, decision, tail = study.iloc[0][["points", "decision", "tail"]]
            readout = readout_of(points, decision, None if pd.isna(tail) else int(tail))
        else:  # (the studies of the reservoirs with the linear ridge readout)
            readout = BASE_READOUT
            points, decision, tail = readout[:3]
        print(f"{condition} ({size}): readout {readout}", flush=True)
        # (the studies of a rule, HPO_RULE: its rate, noise and alpha of the study, the hybrids those of the grid)
        chosen, curve = choose(model, params, size, songs, train_songs, frame, classes, readout, RULES,
                               params.get("readout:rate"), MAX_EPOCHS, params.get("readout:noise"),
                               params.get("readout:alpha"))
        curves.append(curve.assign(condition=condition, network_size=size))
        for rule, (hyper, cv_accuracy) in chosen.items():
            print(f"  {rule}: {hyper}, cross-validation accuracy {100 * cv_accuracy:.2f} %", flush=True)
        true = np.concatenate([songs[w]["next"] for w in test_songs])
        seed_data = {}  # (all the rules on the same reservoirs, drawn with the seed)
        for seed in range(nb_seeds):
            np.random.seed(seed)
            reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
            all_states, all_data, moments = song_data(reservoirs, songs, train_songs + test_songs, readout, classes,
                                                      frame_songs_of(RULES, train_songs))
            seed_data[seed] = (all_data, all_states, moments)
        tasks = [(seed, rule) for seed in range(nb_seeds) for rule in chosen]
        results = Parallel(n_jobs=min(JOBS, len(tasks)))(
            delayed(seed_scores)(rule, chosen[rule][0], *seed_data[seed][:2], train_songs, test_songs, true, classes,
                                 seed, seed_data[seed][2])
            for seed, rule in tasks)
        for (seed, rule), result in zip(tasks, results):
            hyper, cv_accuracy = chosen[rule]
            rows.append(dict(condition=condition, network_size=size, seed=seed, rule=rule, order=rule_order(rule)[1],
                             selection=rule_inputs(rule)[2] or "", points=points, decision=decision, tail=tail, **hyper, cv_accuracy=cv_accuracy, **result))
        for seed in range(nb_seeds):
            print(f"  seed {seed}: " + ", ".join(f"{r['rule']} {100 * r['accuracy']:.1f} %" for r in rows
                                                 if r["seed"] == seed and r["condition"] == condition
                                                 and r["network_size"] == size), flush=True)
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
    if TAIL_SEPARATE:
        raise RuntimeError("HAG_CANARY_TAIL_HPO=0 (studies of the reservoirs) or 1 (studies of the readout) is needed")
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
