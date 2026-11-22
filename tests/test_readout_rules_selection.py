import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from canopy.next_phrase_prediction.hpo import rule_tag
from canopy.next_phrase_prediction.monomial_selection import INNER_FOLDS, RIDGES_STD, Selection
from canopy.next_phrase_prediction.readout_rules import (configuration_counts, configurations, frame_songs_of,
                                                         label_configurations, rule_inputs, rule_order, seed_scores,
                                                         selected_rows)
from test_monomial_selection import data, moments


def test_labels():
    assert rule_inputs("nlms") == ("nlms", 1, None) and rule_inputs("ridge:2") == ("ridge", 2, None)
    assert rule_inputs("nlms:pca:64") == ("nlms", 1, "pca:64")
    assert rule_inputs("eh_np_local:scores") == ("eh_np_local", 1, "scores")
    assert rule_order("ridge:random_pairs:2080") == ("ridge", 1)
    with pytest.raises(ValueError):
        rule_inputs("lms:2")
    assert [rule_tag(label) for label in ("nlms", "nlms:2", "nlms:pca:64", "eh_np_local:scores")] == [
        "nlms", "nlms-order2", "nlms-pca-64", "eh_np_local-scores"]
    assert frame_songs_of(["nlms:pca:64", "ridge"], ["a"]) == () and frame_songs_of(["nlms:pca_frames:8"], ["a"]) == ["a"]


def test_configurations_of_a_selection():
    assert [hyper["ridge"] for hyper, _ in label_configurations("ridge:pca:4")] == list(RIDGES_STD)
    assert label_configurations("nlms:pca:4", rate=0.2, max_epochs=10) == [(dict(rate=0.2), [1, 3, 10])]
    assert label_configurations("eh_np_local:scores", rate=0.2, noise=1.0, max_epochs=3) == [
        (dict(rate=0.2, noise=1.0), [1, 3])]
    rls = label_configurations("rls:pca:4")  # (the two normalizations of the grid of RLS: once)
    assert len(rls) == len(configurations("rls")) // 2 and all(set(hyper) == {"alpha"} for hyper, _ in rls)
    assert label_configurations("nlms") == configurations("nlms")


def song_rows(rng):
    """({wave: (states, labels)}, {wave: states}, songs, moments): classes given by the sign of x_0 x_1 and of x_2."""
    X, Y, groups, _ = data(rng)
    waves = [f"song{i}" for i in range(groups.max() + 1)]
    return ({w: (X[groups == i], Y[groups == i]) for i, w in enumerate(waves)},
            {w: X[groups == i] for i, w in enumerate(waves)}, waves, moments(X))


@pytest.mark.parametrize("rule, hyper, epochs", [("nlms", dict(rate=0.3), [30]), ("ridge", dict(ridge=1.0), [None]),
                                                 ("eh_np_local", dict(rate=0.3, noise=0.5), [30]),
                                                 ("rls", dict(alpha=1e-2), [1])])
def test_rules_learn_the_selected_product(rule, hyper, epochs):
    rng = np.random.default_rng(0)
    rows, states, waves, frame_moments = song_rows(rng)
    fit, val = waves[:24], waves[24:]
    true = np.concatenate([rows[w][1].argmax(axis=1) for w in val])
    accuracy = {}
    for label, config in ((rule, dict(hyper, **({} if rule == "ridge" else dict(normalization="standardize")))),
                          (f"{rule}:residual:1", hyper), (f"{rule}:pca_frames:12", hyper)):
        counts = configuration_counts([(label, 0, config, epochs)], rows, states, fit, val, true, np.arange(4), 0,
                                      frame_moments)
        accuracy[label] = counts[(label, 0, epochs[-1])] / len(true)
    assert accuracy[rule] < 0.6 < accuracy[f"{rule}:residual:1"]
    # (all the products of all the components, 78 for 720 rows: learned more slowly than the single product)
    assert accuracy[f"{rule}:pca_frames:12"] > accuracy[rule] + 0.03


def test_selected_rows_and_seed_scores():
    rng = np.random.default_rng(0)
    rows, states, waves, _ = song_rows(rng)
    train, test = waves[:24], waves[24:]
    selected, inputs = selected_rows("scores", rows, train, None, 0)
    X, Y = (np.vstack([rows[w][i] for w in train]) for i in (0, 1))
    selection = Selection(X, Y, np.repeat(np.arange(24), 30), None, None, seed=0)
    assert selection.linear_ridge in RIDGES_STD
    np.testing.assert_allclose(np.vstack([selected[w][0] for w in train]), selection.features("scores")[0])
    assert selected[train[0]][0].shape == (30, 12 + 10) and inputs(states[test[0]]).shape == (30, 22)
    true = np.concatenate([rows[w][1].argmax(axis=1) for w in test])
    result = seed_scores("nlms:residual:1", dict(rate=0.3, epochs=30), rows, states, train, test, true, np.arange(4), 0)
    assert result["accuracy"] > 0.6


def test_out_of_fold_scores_ignore_the_labels_of_their_fold():
    # (nested choice of the coefficient of the linear readout: the labels of a fold change neither the weights nor the
    # coefficient of the readout giving its scores)
    rng = np.random.default_rng(0)
    X, Y, groups, _ = data(rng)
    _, held = next(GroupKFold(INNER_FOLDS).split(X, groups=groups))
    shuffled = Y.copy()
    shuffled[held] = Y[rng.permutation(held)]
    scores = Selection(X, Y, groups, None, None).cross_scores
    np.testing.assert_array_equal(Selection(X, shuffled, groups, None, None).cross_scores[held], scores[held])
    fixed = Selection(X, Y, groups, None, 1.0)  # (a given coefficient: used for all the folds)
    assert fixed.linear_ridge == 1.0 and fixed.cross_scores.shape == Y.shape
