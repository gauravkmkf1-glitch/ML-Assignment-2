"""Unit and contract tests for the submission's core utilities."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.common import (
    N_CLASSES,
    RESULTS_DIR,
    ROLL_NUMBER,
    SEED,
    BaggingEnsemble,
    SAMME,
    bias_variance_from_probabilities,
    central_feature_indices,
    derive_seed,
    load_dataset,
    load_split,
    make_split,
    mask_centered_block,
    mean_pairwise_disagreement,
    validate_split,
    vote_from_predictions,
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_seed_rule():
    assert ROLL_NUMBER == "P26AI0006"
    assert "".join(c for c in ROLL_NUMBER if c.isdigit()) == "260006"
    assert SEED == 260006


def test_split_is_saved_and_valid(dataset):
    train, validation, test = load_split(RESULTS_DIR / "split.npz", dataset.target)
    assert (len(train), len(validation), len(test)) == (240, 80, 80)
    validate_split(train, validation, test, dataset.target)
    expected = make_split(dataset.target)
    assert all(np.array_equal(a, b) for a, b in zip(expected, (train, validation, test)))


def test_masking_is_exact(dataset):
    X = dataset.data[:5]
    masked = mask_centered_block(X)
    original = X.reshape(-1, 64, 64)
    changed = masked.reshape(-1, 64, 64)
    assert np.all(changed[:, 24:40, 24:40] == 0.0)
    outside = np.ones((64, 64), dtype=bool)
    outside[24:40, 24:40] = False
    assert np.array_equal(changed[:, outside], original[:, outside])
    assert np.array_equal(changed[:, 24:40, 24:40], np.zeros((5, 16, 16)))


def test_central_indices():
    idx = central_feature_indices()
    assert len(idx) == 256
    assert idx.min() == 24 * 64 + 24
    assert idx.max() == 39 * 64 + 39


def test_vote_and_disagreement():
    preds = np.asarray([[0, 1, 2], [0, 1, 1], [1, 1, 2]])
    counts, frac, winners = vote_from_predictions(preds, n_classes=3)
    assert counts.tolist() == [[2, 1, 0], [0, 3, 0], [0, 1, 2]]
    assert winners.tolist() == [0, 1, 2]
    assert np.allclose(frac.sum(axis=1), 1.0)
    assert mean_pairwise_disagreement(preds[:1], n_classes=3) == 0.0
    # Pair (0,1) differs on sample 2; (0,2) on sample 0; (1,2) on samples
    # 0 and 2.  Average over 3 samples and 3 pairs.
    expected = (1 / 3 + 1 / 3 + 2 / 3) / 3
    assert np.isclose(mean_pairwise_disagreement(preds, n_classes=3), expected)
    assert np.isclose(expected, 4 / 9)


def test_bagging_is_deterministic(dataset):
    train, _, _ = load_split(RESULTS_DIR / "split.npz", dataset.target)
    X, y = dataset.data[train][:40], dataset.target[train][:40]
    params = {"max_depth": 4, "min_samples_leaf": 1}
    a = BaggingEnsemble(params, 7, seed=derive_seed("test", "bag"), n_jobs=1).fit(X, y)
    b = BaggingEnsemble(params, 7, seed=derive_seed("test", "bag"), n_jobs=1).fit(X, y)
    assert np.array_equal(a.predict_matrix(X), b.predict_matrix(X))
    c = BaggingEnsemble(params, 7, seed=derive_seed("test", "other"), n_jobs=1).fit(X, y)
    assert not np.array_equal(a.predict_matrix(X), c.predict_matrix(X))
    assert a.serial_training_time(3) <= a.serial_training_time(7)


def test_samme_toy_round():
    X = np.asarray([[0.0], [0.1], [0.2], [0.4], [0.5], [0.6], [1.0], [1.1], [1.2]], dtype=np.float32)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
    model = SAMME(max_depth=1, n_estimators=1, seed=derive_seed("test", "toy")).fit(X, y)
    assert not model.failed_
    assert model.n_estimators_fitted_ == 1
    assert np.isclose(model.history[0]["weighted_error"], 1 / 3)
    assert np.isclose(model.history[0]["alpha"], np.log(4.0))
    assert np.isclose(np.sum(model.weight_history[1]), 1.0)
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)
    assert set(model.predict(X)).issubset({0, 1, 2})


def test_samme_rejects_random_learner():
    # A single constant feature cannot beat the 1 - 1/K random threshold for
    # three balanced classes.
    X = np.zeros((9, 1), dtype=np.float32)
    y = np.repeat([0, 1, 2], 3)
    model = SAMME(max_depth=1, n_estimators=3, seed=derive_seed("test", "reject")).fit(X, y)
    assert model.failed_
    assert model.n_estimators_fitted_ == 0
    assert "rejected" in model.stop_reason


def test_bias_variance_formula():
    # Two repetitions and one sample; hand-computable vectors.
    p = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=float)
    p = np.concatenate([p, np.asarray([[[0.0, 1.0], [1.0, 0.0]]], dtype=float)], axis=0)
    bias, var = bias_variance_from_probabilities(p, np.asarray([0, 0]), n_classes=2)
    # pbar = [[0.5,0.5],[0.5,0.5]]; e = [[1,0],[1,0]]
    # bias^2 = mean(0.5, 0.5) = 0.5; variance = mean(0.5, 0.5) = 0.5.
    assert np.isclose(bias, 0.5)
    assert np.isclose(var, 0.5)


def test_no_dimensionality_reduction():
    """Guard against an accidental PCA import in the experiment code."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "PCA(" not in text
        assert "TruncatedSVD(" not in text
        assert "KernelPCA(" not in text
