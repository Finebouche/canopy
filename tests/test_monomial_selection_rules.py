import numpy as np
import pytest

from canopy.next_phrase_prediction.monomial_selection import Selection
from canopy.next_phrase_prediction.monomial_selection_rules import (configuration_counts, pairs, rule_configurations,
                                                                    seed_metrics, trained_readouts)
from test_monomial_selection import data, moments


def song_data(rng):
    """(data (X, Y, moments) by song, songs, labels of all the rows)."""
    X, Y, groups, labels = data(rng)
    waves = [f"song{i}" for i in range(groups.max() + 1)]
    return ({w: X[groups == i] for i, w in enumerate(waves)}, {w: Y[groups == i] for i, w in enumerate(waves)},
            moments(X)), waves, labels


def test_pairs_and_configurations():
    assert pairs(["linear", "full_std"], ["nlms", "eh_np_local", "rls"]) == [
        ("linear", "nlms"), ("linear", "eh_np_local"), ("linear", "rls"), ("full_std", "nlms")]
    assert len(rule_configurations("eh_np_local")) == 8 and len(rule_configurations("neh")) == 6
    assert all("normalization" not in hyper for rule in ("nlms", "neh", "rls") for hyper, _ in rule_configurations(rule))
    assert rule_configurations("rls")[0][1] == [1]


@pytest.mark.parametrize("rule, hyper", [("nlms", dict(rate=0.3)), ("neh", dict(rate=0.1, noise=0.5)),
                                         ("eh_np_local", dict(rate=0.3, noise=0.5)),
                                         ("rls", dict(alpha=1e-2))])
def test_rules_learn_the_selected_product(rule, hyper):
    # the classes depend on the product x_0 x_1: the rule on z alone stays near 0.5, with the product it learns it
    rng = np.random.default_rng(0)
    (X, Y, frame_moments), waves, _ = song_data(rng)
    fit, val = waves[:24], waves[24:]
    true = np.concatenate([Y[w].argmax(axis=1) for w in val])
    epochs = [1] if rule == "rls" else [30]
    accuracy = {}
    for method in ("linear", "residual:1"):
        counts = configuration_counts((X, Y, frame_moments), fit, val, true, np.arange(4), 1.0, method, rule, 0, hyper,
                                      epochs, 0)
        accuracy[method] = counts[epochs[-1]] / len(true)
    assert accuracy["linear"] < 0.6 < accuracy["residual:1"]


def test_dual_form_of_all_the_products_and_seed_metrics():
    rng = np.random.default_rng(0)
    (X, Y, frame_moments), waves, _ = song_data(rng)
    train, test = waves[:24], waves[24:]
    selection = Selection(np.vstack([X[w] for w in train]), np.vstack([Y[w] for w in train]),
                          np.repeat(np.arange(24), 30), frame_moments, 1.0)
    _, readout, inputs, n_inputs = next(trained_readouts(selection, train, [30] * 24, "full_std", "nlms",
                                                         dict(rate=0.3), [2], np.random.default_rng(0)))
    assert n_inputs == 12 + 78 and readout.run(inputs(X[test[0]])).shape == (30, 4)
    true = np.concatenate([Y[w].argmax(axis=1) for w in test])
    result = seed_metrics((X, Y, frame_moments), train, test, true, np.arange(4), 1.0, "scores", "nlms",
                          dict(rate=0.3, epochs=3), 0)
    assert result["inputs"] == 12 + 10 and 0 <= result["accuracy"] <= 1
