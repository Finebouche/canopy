"""
Shared parts of the studies of canopy.studies (phrase_boundaries, next_phrase, syllables_classification):

  - storage of the Optuna studies of the reservoirs, as in hag and bachue: all those of a study and of a dataset in one
    SQLite database, outputs/<study>/hpo/marron1_db.sqlite3 (canopy.settings.database), their names
    canopy.settings.study_name. Several jobs writing the same SQLite file would corrupt it: a job (slurm/canary.sbatch)
    optimizes its Optuna study in a database of its own (CANOPY_JOB_STUDIES_DIR, in /dev/shm during the job, copied back
    to its part, outputs/<study>/hpo/parts/<dataset>/<Optuna study>/: python -m canopy.studies.common part <study>
    <model> <size>), started from the Optuna study of the database of the dataset if there is one; the parts are merged
    into the databases of the datasets afterwards (python -m canopy.studies.common merge [studies]);
  - search space of the reservoirs (reservoir_params) and run of a study by several workers (run_study);
  - conditions of the final evaluations, for each size (number of neurons in all), from the best trials of the
    studies: the pure models (canopy.models.reservoirs.PURE_MODELS) with the best parameters of their study,
    "<hag>_proj" (the parameters of the study of "<hag>" without the HAG learning) and "<hag>+<esn>" (hybrid of the
    best parts optimized alone with size / 2 neurons, not optimized), missing studies skipped ("mfcc", without
    reservoir, is added by each evaluation);
  - paired comparisons by seed of the results of the evaluations (compare): mean difference of a metric (condition A -
    condition B), its bootstrap 95 % confidence interval and the p-value of the Wilcoxon signed-rank test, for each
    size: each condition with the reference condition, the HAG models with their projections and the hybrids with
    their parts;
  - certainty of a readout along the phrases of the test songs (curve, summarize, evaluate_conditions,
    save_certainty), for the next phrase and the class of the syllable: on the frames of every phrase, true_output
    (output of the readout unit of the true class), margin (true_output minus the largest output of the other classes,
    > 0: frame correctly classified) and correct, averaged per phrase in bins of the relative position in the phrase
    (0 = onset, 1 = end of its frames) and of the time since the onset, then over the phrases; for each phrase, the
    mean true_output of its last third minus that of its first third (Wilcoxon signed-rank test over the phrases).
"""
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from scipy.stats import wilcoxon

from canopy.models.reservoirs import ESN_MODELS, HAG_MODELS, PURE_MODELS, parts
from canopy.settings import (DATASET_NAME, JOB_STUDIES, N_TRIALS, PREFIXES, WORKERS, database, outputs_dir,
                             studies_database, study_name)


# ---- studies of the reservoirs
def storage(path):
    """Optuna storage of the SQLite database path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def study_names(path):
    """Names of the Optuna studies of a database (none if it does not exist: opening a missing SQLite database creates
    it, empty)."""
    return set(optuna.get_all_study_names(storage(path))) if Path(path).exists() else set()


def best_trial(study, name):
    """Best trial of the Optuna study name of the study (None without it), in canopy.settings.studies_database."""
    if name not in study_names(studies_database(study)):
        return None
    try:
        return optuna.load_study(study_name=name, storage=storage(studies_database(study))).best_trial
    except (KeyError, ValueError):
        return None


def stop_after_failures(max_failures=10):
    """Callback stopping the optimization after max_failures failed trials in a row (systematic error)."""
    def callback(study, trial):
        last = [t.state for t in study.get_trials(deepcopy=False)][-max_failures:]
        if len(last) == max_failures and all(state == TrialState.FAIL for state in last):
            study.stop()
    return callback


def reservoir_params(trial, model, max_length, leak=False, input_gains=False, input_scaling_max=0.2):
    """
    Parameters of each part of the model (prefix "<part>:" in a hybrid), the ranges of hag.hpo.hpo_esn. With
    input_gains, a gain for each group of input features: the MFCC (input_scaling) and, relative to them, their deltas
    (delta_gain) and delta-deltas (delta2_gain), as canapy (input scaling per group of its ESN). With leak, the leak
    rate of each reservoir (1 otherwise; canapy: 0.1), used by the HAG learning too. max_length: the number of frames of
    the longest phrase of the train songs (bounds of the increments of HAG). input_scaling_max: the largest input
    scaling of the search (0.2 for the MFCC, in (0, 1)).
    """
    params = {}
    for part, p in parts(model):
        params[p + "input_scaling"] = trial.suggest_float(p + "input_scaling", 0.01, input_scaling_max, step=0.005)
        if input_gains:
            params[p + "delta_gain"] = trial.suggest_float(p + "delta_gain", 0.01, 100, log=True)
            params[p + "delta2_gain"] = trial.suggest_float(p + "delta2_gain", 0.01, 100, log=True)
        if leak:
            params[p + "leaky_rate"] = trial.suggest_float(p + "leaky_rate", 0.01, 1, log=True)
        params[p + "bias_scaling"] = trial.suggest_float(p + "bias_scaling", 0, 0.2, step=0.005)
        if part in HAG_MODELS:
            params[p + "weight_increment"] = trial.suggest_float(p + "weight_increment", 0.001, 0.1, step=0.001)
            params[p + "use_full_instance"] = trial.suggest_categorical(p + "use_full_instance", [True, False])
            params[p + "min_increment"] = trial.suggest_int(p + "min_increment", 3, max_length)
            params[p + "max_increment"] = trial.suggest_int(p + "max_increment", params[p + "min_increment"],
                                                            max_length * 5)
            if part == "mean_hag":
                params[p + "target_rate"] = trial.suggest_float(p + "target_rate", 0.5, 1, step=0.01)
                params[p + "rate_spread"] = trial.suggest_float(p + "rate_spread", 0.01, 0.4, step=0.005)
            else:
                params[p + "variance_target"] = trial.suggest_float(p + "variance_target", 0.001, 0.02, step=0.001)
                params[p + "variance_spread"] = trial.suggest_float(p + "variance_spread", 0.001, 0.05, step=0.002)
                params[p + "intrinsic_saturation"] = trial.suggest_float(p + "intrinsic_saturation", 0.8, 0.98,
                                                                         step=0.02)
                params[p + "intrinsic_coef"] = trial.suggest_float(p + "intrinsic_coef", 0.8, 0.98, step=0.02)
        elif part in ESN_MODELS:
            params[p + "connectivity"] = trial.suggest_float(p + "connectivity", 0, 1)
            params[p + "spectral_radius"] = trial.suggest_float(p + "spectral_radius", 0.4, 1.6, step=0.01)
        else:
            raise ValueError(f"Unknown model {part}")
    return params


def longest_phrase(songs, train_songs, frame):
    """Number of frames of the longest phrase of the train songs."""
    return max(int(end / frame) + 1 - int(start / frame) for w in train_songs for start, end, _ in songs[w]["phrases"])


def _worker(study, name, make_objective, model, size, worker_index):
    optuna_study = optuna.load_study(study_name=name, storage=storage(studies_database(study)),
                                     sampler=TPESampler(constant_liar=True, seed=worker_index))
    optuna_study.optimize(make_objective(model, size), gc_after_trial=True, catch=(Exception,),
                          callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,)), stop_after_failures()])


def run_study(study, model, size, make_objective):
    """Optimizes the Optuna study of the model and size of the study (N_TRIALS complete trials, WORKERS processes),
    resumed where it stopped; make_objective(model, size): the objective of a worker (its data loaded in the worker).
    In a job (CANOPY_JOB_STUDIES_DIR), it starts from the Optuna study of the database of the dataset if it is there and
    not in the database of the job."""
    if model not in PURE_MODELS:
        raise ValueError(f"Unknown model {model}, expected one of {PURE_MODELS} (the hybrids <hag>+<esn> are not "
                         "optimized: they combine the parts optimized alone, see conditions)")
    name = study_name(study, model, size)
    studies = studies_database(study)
    if JOB_STUDIES and name not in study_names(studies) and name in study_names(database(study)):
        optuna.copy_study(from_study_name=name, from_storage=storage(database(study)), to_storage=storage(studies))
        print(f"{name}: resumed from {database(study)}", flush=True)
    optuna_study = optuna.create_study(study_name=name, storage=storage(studies), direction="maximize",
                                       load_if_exists=True)
    done = completed(optuna_study)
    print(f"{name}: {done} / {N_TRIALS} trials done, {WORKERS} workers, {studies}", flush=True)
    if done < N_TRIALS:
        Parallel(n_jobs=WORKERS)(delayed(_worker)(study, name, make_objective, model, size, i) for i in range(WORKERS))
    best = optuna.load_study(study_name=name, storage=storage(studies)).best_trial
    print(f"best trial {best.number}: {best.value:.4f}, {best.user_attrs}, {best.params}")
    return best


def completed(optuna_study):
    return sum(t.state == TrialState.COMPLETE for t in optuna_study.trials)


def copy_into(name, source, target):
    """Copies the Optuna study name of the database source into the database target, replacing it if source has more
    completed trials. Returns a description."""
    source_study = optuna.load_study(study_name=name, storage=storage(source))
    if name in study_names(target):
        current = optuna.load_study(study_name=name, storage=storage(target))
        if completed(current) >= completed(source_study):
            return f"{name}: kept ({completed(current)} completed trials >= {completed(source_study)})"
        optuna.delete_study(study_name=name, storage=storage(target))
    optuna.copy_study(from_study_name=name, from_storage=storage(source), to_storage=storage(target))
    return f"{name}: {completed(source_study)} completed trials copied into {target}"


def part_dir(study, model, size):
    """Folder of the database of a job optimizing the Optuna study of a model and size (slurm/canary.sbatch)."""
    return outputs_dir(study, "hpo") / "parts" / DATASET_NAME / study_name(study, model, size)


def merge(studies=None):
    """Merges the parts of the jobs (outputs/<study>/hpo/parts/<dataset>/<Optuna study>/<dataset>_db.sqlite3) into the
    databases of the datasets (outputs/<study>/hpo/<dataset>_db.sqlite3)."""
    for study in studies or PREFIXES:
        for part in sorted((outputs_dir(study, "hpo") / "parts").glob("*/*/*.sqlite3")):
            for name in sorted(study_names(part)):
                print(copy_into(name, part, outputs_dir(study, "hpo") / part.name))


def conditions(size, study):
    """[(condition, model, params, ridge, threshold)] of size neurons from the Optuna studies of the study (study_name):
    ridge and threshold None when chosen by cross-validation, threshold None without threshold."""
    result = []
    for model in PURE_MODELS:
        trial = best_trial(study, study_name(study, model, size))
        if trial is None:
            print(f"  (no study {study_name(study, model, size)}: {model} skipped)")
        else:
            result.append((model, model, trial.params, trial.user_attrs.get("ridge"),
                           trial.user_attrs.get("threshold")))
    for hag in HAG_MODELS:
        trial = best_trial(study, study_name(study, hag, size))
        if trial is not None:
            result.append((f"{hag}_proj", f"{hag}_proj", trial.params, None, None))
    for hag in HAG_MODELS:
        for esn in ESN_MODELS:
            hybrid = f"{hag}+{esn}"
            halves = [best_trial(study, study_name(study, part, size // 2)) for part, _ in parts(hybrid)]
            if all(trial is not None for trial in halves):
                params = {prefix + key: value for (_, prefix), trial in zip(parts(hybrid), halves)
                          for key, value in trial.params.items()}
                result.append((hybrid, hybrid, params, None, None))
    return result


# ---- paired comparisons of the results
def compare(results, metric, reference, nb_bootstrap=10000):
    """Paired comparisons by seed of the metric (condition A - condition B), for each size: each condition with the
    reference condition, the HAG models with their projections and the hybrids with their parts."""
    rows = []
    rng = np.random.default_rng(0)
    mfcc = results[results.condition == "mfcc"]
    for size, by_size in results[results.condition != "mfcc"].groupby("network_size"):
        by_size = pd.concat([by_size, mfcc])
        f1 = by_size.pivot_table(index="seed", columns="condition", values=metric)
        pairs = [(c, reference) for c in f1.columns if c != reference]
        pairs += [(hag, f"{hag}_proj") for hag in HAG_MODELS]
        for hag in HAG_MODELS:
            for esn in ESN_MODELS:
                hybrid = f"{hag}+{esn}"
                pairs += [(hybrid, hag), (hybrid, esn)]
        for a, b in dict.fromkeys(pairs):
            if a not in f1.columns or b not in f1.columns:
                continue
            diff = (f1[a] - f1[b]).dropna().to_numpy()
            if len(diff) == 0:
                continue
            means = rng.choice(diff, size=(nb_bootstrap, len(diff))).mean(axis=1)
            p_value = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            rows.append(dict(network_size=size, a=a, b=b, nb_seeds=len(diff), mean_diff=diff.mean(),
                             ci_low=np.percentile(means, 2.5), ci_high=np.percentile(means, 97.5),
                             wilcoxon_p=p_value, a_better=int((diff > 0).sum())))
    return pd.DataFrame(rows)


# ---- certainty along the phrases
RELATIVE_BINS = 10
ONSET_BIN = 0.05  # s
ONSET_MAX = 1.0  # s
MIN_FRAMES = 6  # phrases shorter than this (in frames) are skipped
COLORS = {"mean_hag": "#2a78d6", "var_hag": "#eb6834", "random_ee": "#1baf7a", "random_ee_dense": "#eda100",
          "random_ei": "#4a3aa7"}


def curve(out, c, label, frame):
    """Curves of the true output, margin and correctness of the frames out (frames, classes) of a phrase, true class c,
    with the relative position and onset time bins of the frames."""
    true = out[:, c]
    margin = true - np.delete(out, c, axis=1).max(axis=1)
    relative = (np.arange(len(out)) + 0.5) / len(out)
    return dict(label=label, true=true, margin=margin, correct=margin > 0,
                relative=np.minimum((relative * RELATIVE_BINS).astype(int), RELATIVE_BINS - 1),
                onset=(np.arange(len(out)) * frame / ONSET_BIN).astype(int))


def summarize(curves, condition, seed):
    """Rows of the bins (mean over the phrases of the per-phrase bin means) and the first / last third per phrase."""
    rows = []
    for axis, n_bins in (("relative", RELATIVE_BINS), ("onset", int(ONSET_MAX / ONSET_BIN))):
        for b in range(n_bins):
            per_phrase = [(c["true"][c[axis] == b].mean(), c["margin"][c[axis] == b].mean(),
                           c["correct"][c[axis] == b].mean()) for c in curves if np.any(c[axis] == b)]
            if not per_phrase:
                continue
            true, margin, correct = np.array(per_phrase).T
            position = (b + 0.5) / n_bins if axis == "relative" else (b + 0.5) * ONSET_BIN
            rows.append(dict(condition=condition, seed=seed, axis=axis, position=position, true_output=true.mean(),
                             margin=margin.mean(), accuracy=correct.mean(), n_phrases=len(per_phrase)))
    first = np.array([c["true"][c["relative"] < RELATIVE_BINS / 3].mean() for c in curves])
    last = np.array([c["true"][c["relative"] >= 2 * RELATIVE_BINS / 3].mean() for c in curves])
    first_correct = np.array([c["correct"][c["relative"] < RELATIVE_BINS / 3].mean() for c in curves])
    last_correct = np.array([c["correct"][c["relative"] >= 2 * RELATIVE_BINS / 3].mean() for c in curves])
    phrases = dict(condition=condition, seed=seed, n_phrases=len(curves), first_third=first.mean(),
                   last_third=last.mean(), last_minus_first=(last - first).mean(),
                   share_rising=float(np.mean(last > first)), wilcoxon_p=wilcoxon(last - first).pvalue,
                   accuracy_first_third=first_correct.mean(), accuracy_last_third=last_correct.mean())
    return rows, phrases


def evaluate_conditions(study, size, wanted, nb_seeds, songs, train_songs, test_songs, frame, classes, select_ridge,
                        run_seed):
    """(bins, phrases) of the conditions wanted of size neurons (conditions of the study), for nb_seeds seeds:
    run_seed(condition, model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed) ->
    summarize(...) of the task; the ridge of the conditions without study is chosen by select_ridge(model, params, size,
    songs, train_songs, frame, classes)."""
    available = {c[0]: c for c in conditions(size, study)}
    bins, phrases = [], []
    for condition in wanted:
        if condition not in available:
            print(f"{condition} ({size}): no study, skipped")
            continue
        _, model, params, ridge, _ = available[condition]
        if ridge is None:
            ridge, _ = select_ridge(model, params, size, songs, train_songs, frame, classes)
        for seed in range(nb_seeds):
            rows, summary = run_seed(condition, model, params, size, ridge, songs, train_songs, test_songs, frame,
                                     classes, seed)
            bins += rows
            phrases.append(summary)
            print(f"{condition} ({size}) seed {seed}: true-class output first third {summary['first_third']:.3f} -> "
                  f"last third {summary['last_third']:.3f} (rising in {100 * summary['share_rising']:.0f} % of "
                  f"{summary['n_phrases']} phrases, Wilcoxon p {summary['wilcoxon_p']:.1e}); accuracy "
                  f"{100 * summary['accuracy_first_third']:.1f} -> {100 * summary['accuracy_last_third']:.1f} %",
                  flush=True)
    return bins, phrases


def save_certainty(bins, phrases, study, name, size, title, labels):
    """outputs/<study>/analysis/<name>_<size>.csv, ..._phrases.csv and .png (nothing if no condition was
    evaluated)."""
    if not phrases:
        sys.exit("No condition evaluated: nothing saved")
    bins, phrases = pd.DataFrame(bins), pd.DataFrame(phrases)
    stem = f"{name}_{size}"
    folder = outputs_dir(study, "analysis")
    folder.mkdir(parents=True, exist_ok=True)
    bins.to_csv(folder / f"{stem}.csv", index=False)
    phrases.to_csv(folder / f"{stem}_phrases.csv", index=False)
    # mean true output (labels[0]) and accuracy (labels[1]) along the phrases (relative position), per condition
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = bins[bins.axis == "relative"].groupby(["condition", "position"])[["true_output", "accuracy"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
    for ax, column, title_ax in ((axes[0], "true_output", labels[0]), (axes[1], "accuracy", labels[1])):
        ax.set_facecolor("#fcfcfb")
        for condition in data.index.get_level_values(0).unique():
            values = data.loc[condition, column]
            color = COLORS.get(condition, "#4a3aa7")
            ax.plot(values.index, values.values, color=color, linewidth=2, marker="o", markersize=4, label=condition)
            ax.annotate(condition, (values.index[-1], values.values[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#0b0b0b")
        ax.set_title(title_ax, fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xlabel("Relative position in the phrase (0 = onset, 1 = end)", fontsize=8, color="#52514a")
        ax.set_xlim(0, 1.25)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors="#52514a", labelsize=8)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"{title}, test songs ({size} neurons, mean over seeds)", fontsize=11,
                 color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(folder / f"{stem}.png", dpi=150)
    plt.close(fig)
    print("Saved to", folder / f"{stem}.csv", folder / f"{stem}_phrases.csv", folder / f"{stem}.png")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "part":  # (the folder of the part of a job, slurm/canary.sbatch)
        print(part_dir(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    elif command == "merge":
        merge(sys.argv[2:] or None)
    else:
        raise ValueError(f"Unknown command {command!r}: part <study> <model> <size>, merge [studies]")
