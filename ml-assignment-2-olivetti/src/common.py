"""Shared, deterministic utilities for the Olivetti Faces assignment.

The module intentionally contains no dimensionality reduction.  Every model
consumes the raw 64 x 64 = 4096 pixel vector.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Identity / seed
# ---------------------------------------------------------------------------
ROLL_NUMBER = "P26AI0006"
STUDENT_NAME = "Kumar Gaurav"
# All numeric characters, in their original order: 2,6,0,0,0,6.
SEED = int("".join(ch for ch in ROLL_NUMBER if ch.isdigit()))
N_IDENTITIES = 40
N_CLASSES = 40
IMAGE_SHAPE = (64, 64)
N_FEATURES = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
MASK_ROW_START, MASK_ROW_END = 24, 40
MASK_COL_START, MASK_COL_END = 24, 40

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"
CONFIGS_DIR = ROOT / "configs"
PAPER_DIR = ROOT / "paper"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CONFIGS_DIR, PAPER_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Serialisation / logging
# ---------------------------------------------------------------------------
def json_default(obj: Any) -> Any:
    """Convert numpy and pathlib objects into JSON-safe values."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def json_dump(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=json_default, sort_keys=False)
        fh.write("\n")
    tmp.replace(path)


def json_load(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def setup_logging(log_file: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure one console handler and one UTF-8 file handler."""
    if log_file is None:
        log_file = LOGS_DIR / "run.log"
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ml_assignment_2")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Logging to %s", log_file)
    return logger


def dataframe_to_csv(rows: Sequence[dict[str, Any]] | pd.DataFrame, path: Path | str) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def derive_seed(*parts: Any) -> int:
    """Stable integer seed derived from arbitrary parts (independent of hash())."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    images: np.ndarray  # (400, 64, 64) float32 in [0, 1]
    target: np.ndarray  # (400,) int16, identities 0..39
    data: np.ndarray  # (400, 4096) float32 raw flattened pixels

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.data.shape[1])


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_dataset(path: Path | str | None = None) -> Dataset:
    """Load the cached Olivetti arrays, or fetch and cache them on first use.

    The raw pixels are kept as float32 in [0, 1], exactly as returned by
    scikit-learn's ``fetch_olivetti_faces(shuffle=False)``.  No scaling,
    centering, PCA, or other transformation is applied.
    """
    path = Path(path) if path is not None else DATA_DIR / "olivetti_faces.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            images = np.asarray(z["images"], dtype=np.float32)
            target = np.asarray(z["target"], dtype=np.int16)
    else:
        # Keep the scikit-learn download cache inside the repository so that
        # repeated runs on the same machine do not re-download the data.
        bunch = fetch_olivetti_faces(shuffle=False, data_home=str(DATA_DIR / "sklearn_cache"))
        images = np.asarray(bunch.images, dtype=np.float32)
        target = np.asarray(bunch.target, dtype=np.int16)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, images=images, target=target)

    if images.shape != (400, 64, 64):
        raise ValueError(f"Unexpected Olivetti image shape: {images.shape}")
    if target.shape != (400,):
        raise ValueError(f"Unexpected Olivetti target shape: {target.shape}")
    if not np.array_equal(np.unique(target), np.arange(40)):
        raise ValueError("Expected exactly 40 identities labelled 0..39")
    if not (0.0 <= float(images.min()) and float(images.max()) <= 1.0):
        raise ValueError("Expected raw grayscale pixels in [0, 1]")

    data = np.ascontiguousarray(images.reshape(images.shape[0], -1), dtype=np.float32)
    return Dataset(images=images, target=target, data=data)


def dataset_checksum(dataset: Dataset) -> dict[str, Any]:
    return {
        "images_sha256": _sha256_array(dataset.images),
        "target_sha256": _sha256_array(dataset.target),
        "images_shape": list(dataset.images.shape),
        "target_shape": list(dataset.target.shape),
        "pixel_min": float(dataset.images.min()),
        "pixel_max": float(dataset.images.max()),
        "n_identities": int(len(np.unique(dataset.target))),
        "images_per_identity": sorted({int(v) for v in np.bincount(dataset.target)}),
    }


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def make_split(target: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within each identity, randomly assign 6 train / 2 val / 2 test images.

    The single ``numpy.random.default_rng(seed)`` stream is consumed identity
    by identity (identities in ascending order).  For each identity the ten
    indices are permuted once; the first six are train, the next two are
    validation, and the last two are test.  This exact procedure is saved and
    reused by every experiment.
    """
    target = np.asarray(target)
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for identity in np.unique(target):
        idx = np.flatnonzero(target == identity)
        if len(idx) != 10:
            raise ValueError(f"Identity {identity} does not have exactly 10 images")
        perm = rng.permutation(idx)
        train.extend(int(i) for i in perm[:6])
        validation.extend(int(i) for i in perm[6:8])
        test.extend(int(i) for i in perm[8:10])
    return (
        np.asarray(train, dtype=np.int16),
        np.asarray(validation, dtype=np.int16),
        np.asarray(test, dtype=np.int16),
    )


def validate_split(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    target: np.ndarray,
) -> None:
    train = np.asarray(train, dtype=int)
    validation = np.asarray(validation, dtype=int)
    test = np.asarray(test, dtype=int)
    for name, arr in (("train", train), ("validation", validation), ("test", test)):
        if arr.ndim != 1:
            raise ValueError(f"{name} indices must be one-dimensional")
        if len(np.unique(arr)) != len(arr):
            raise ValueError(f"{name} contains duplicate indices")
        if arr.min() < 0 or arr.max() >= len(target):
            raise ValueError(f"{name} contains out-of-range indices")
    all_idx = np.concatenate([train, validation, test])
    if len(np.unique(all_idx)) != len(target):
        raise ValueError("Train/validation/test do not form a partition of all 400 images")
    if sorted(all_idx.tolist()) != list(range(len(target))):
        raise ValueError("Train/validation/test do not cover every image exactly once")
    for identity in range(40):
        counts = (
            int(np.sum(target[train] == identity)),
            int(np.sum(target[validation] == identity)),
            int(np.sum(target[test] == identity)),
        )
        if counts != (6, 2, 2):
            raise ValueError(f"Identity {identity} has split counts {counts}, expected (6, 2, 2)")


def save_split(
    path: Path | str,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    target: np.ndarray,
    seed: int = SEED,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_split(train, validation, test, target)
    np.savez_compressed(
        path,
        train=np.asarray(train, dtype=np.int16),
        validation=np.asarray(validation, dtype=np.int16),
        test=np.asarray(test, dtype=np.int16),
        seed=np.int64(seed),
        roll_number=np.array(ROLL_NUMBER),
    )
    df = pd.DataFrame(
        {
            "dataset_index": np.concatenate([train, validation, test]).astype(int),
            "identity": target[np.concatenate([train, validation, test])].astype(int),
            "split": np.concatenate(
                [
                    np.full(len(train), "train"),
                    np.full(len(validation), "validation"),
                    np.full(len(test), "test"),
                ]
            ),
        }
    ).sort_values("dataset_index", ignore_index=True)
    dataframe_to_csv(df, path.with_suffix(".csv"))


def load_split(path: Path | str | None = None, target: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path) if path is not None else RESULTS_DIR / "split.npz"
    with np.load(path, allow_pickle=False) as z:
        train = np.asarray(z["train"], dtype=int)
        validation = np.asarray(z["validation"], dtype=int)
        test = np.asarray(z["test"], dtype=int)
        stored_seed = int(z["seed"])
        stored_roll = str(z["roll_number"])
    if stored_seed != SEED:
        raise ValueError(f"Saved split seed {stored_seed} does not match roll-number seed {SEED}")
    if stored_roll != ROLL_NUMBER:
        raise ValueError(f"Saved split roll number {stored_roll} does not match {ROLL_NUMBER}")
    if target is not None:
        validate_split(train, validation, test, target)
    return train, validation, test


def split_checksum(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in (train, validation, test):
        h.update(np.ascontiguousarray(arr, dtype=np.int32).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def mask_centered_block(
    X: np.ndarray,
    row_start: int = MASK_ROW_START,
    row_end: int = MASK_ROW_END,
    col_start: int = MASK_COL_START,
    col_end: int = MASK_COL_END,
) -> np.ndarray:
    """Return a copy of flattened images with the centred block set to zero.

    Rows and columns are zero-based; the assignment's rows/columns 24 to 39
    therefore correspond to Python slices ``24:40``.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"Expected (n_samples, {N_FEATURES}) flattened images, got {X.shape}")
    images = X.reshape(-1, *IMAGE_SHAPE).copy()
    images[:, row_start:row_end, col_start:col_end] = 0.0
    return np.ascontiguousarray(images.reshape(X.shape[0], -1), dtype=np.float32)


def central_feature_indices(
    row_start: int = MASK_ROW_START,
    row_end: int = MASK_ROW_END,
    col_start: int = MASK_COL_START,
    col_end: int = MASK_COL_END,
) -> np.ndarray:
    cols = np.arange(IMAGE_SHAPE[1])
    rows = np.arange(IMAGE_SHAPE[0])
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    mask = (rr >= row_start) & (rr < row_end) & (cc >= col_start) & (cc < col_end)
    return np.flatnonzero(mask.reshape(-1))


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def align_proba(proba: np.ndarray, model_classes: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Expand a predict_proba matrix to the fixed 40-column class order."""
    proba = np.asarray(proba, dtype=np.float64)
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    model_classes = np.asarray(model_classes, dtype=int)
    for j, c in enumerate(model_classes):
        if 0 <= int(c) < n_classes:
            out[:, int(c)] = proba[:, j]
    return out


def vote_from_predictions(preds: np.ndarray, n_classes: int = N_CLASSES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plurality vote.

    Parameters
    ----------
    preds : (n_models, n_samples) integer class predictions.

    Returns
    -------
    counts : (n_samples, n_classes) integer vote counts
    fractions : (n_samples, n_classes) vote fractions
    winners : (n_samples,) winning classes
    """
    preds = np.asarray(preds)
    if preds.ndim != 2:
        raise ValueError("preds must be (n_models, n_samples)")
    n_models, n_samples = preds.shape
    counts = np.zeros((n_samples, n_classes), dtype=np.int32)
    for k in range(n_classes):
        counts[:, k] = np.sum(preds == k, axis=0)
    fractions = counts.astype(np.float64) / float(n_models)
    winners = np.argmax(counts, axis=1).astype(int)
    return counts, fractions, winners


def winning_confidence(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba)
    return np.max(proba, axis=1)


def vote_margin(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba)
    if proba.shape[1] < 2:
        return np.ones(proba.shape[0], dtype=float)
    s = np.sort(proba, axis=1)
    return s[:, -1] - s[:, -2]


def mean_pairwise_disagreement(preds: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """Exact average pairwise disagreement for a set of hard predictions."""
    preds = np.asarray(preds)
    n_models = preds.shape[0]
    if n_models < 2:
        return 0.0
    counts = np.stack([np.sum(preds == k, axis=0) for k in range(n_classes)], axis=1)
    same_pairs = np.sum(counts * (counts - 1) // 2, axis=1)
    total_pairs = n_models * (n_models - 1) // 2
    return float(1.0 - np.mean(same_pairs / float(total_pairs)))


def leaf_stats(tree: DecisionTreeClassifier) -> dict[str, float]:
    """Number of leaves, mean training samples per leaf, singleton fraction."""
    leaf = tree.tree_.children_left == tree.tree_.children_right
    counts = np.asarray(tree.tree_.n_node_samples[leaf], dtype=np.float64)
    return {
        "leaves": int(leaf.sum()),
        "mean_samples_per_leaf": float(np.mean(counts)),
        "singleton_leaf_fraction": float(np.mean(counts == 1.0)),
    }


def feature_mass(importances: np.ndarray, indices: np.ndarray) -> float:
    importances = np.asarray(importances, dtype=np.float64)
    return float(np.sum(importances[np.asarray(indices, dtype=int)]))


# ---------------------------------------------------------------------------
# Bagging (implemented here, not imported from sklearn.ensemble)
# ---------------------------------------------------------------------------
def _fit_one_tree(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    params: dict[str, Any],
    random_state: int,
) -> tuple[DecisionTreeClassifier, float]:
    t0 = time.perf_counter()
    tree = DecisionTreeClassifier(random_state=int(random_state), **params)
    tree.fit(X[indices], y[indices])
    return tree, time.perf_counter() - t0


class BaggingEnsemble:
    """Bootstrap-aggregated decision trees.

    A single sequence of up to ``n_estimators`` bootstrap replicates is drawn
    from one deterministic RNG.  Consequently, an ensemble of size B is
    exactly the first B trees of a larger ensemble, which makes the B-curves
    nested and paired.  Tree random states are drawn from the same stream
    immediately after each bootstrap replicate.
    """

    def __init__(
        self,
        base_params: dict[str, Any],
        n_estimators: int,
        seed: int,
        n_jobs: int = 1,
    ) -> None:
        self.base_params = dict(base_params)
        if "random_state" in self.base_params:
            raise ValueError("base_params must not contain random_state")
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.trees: list[DecisionTreeClassifier] = []
        self.bootstrap_indices = np.empty((0, 0), dtype=np.int32)
        self.tree_seeds = np.empty(0, dtype=np.int64)
        self.fit_times = np.empty(0, dtype=np.float64)
        self.n_train_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingEnsemble":
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        self.n_train_ = n
        rng = np.random.default_rng(self.seed)
        boot = np.empty((self.n_estimators, n), dtype=np.int32)
        seeds = np.empty(self.n_estimators, dtype=np.int64)
        for b in range(self.n_estimators):
            boot[b] = rng.integers(0, n, size=n).astype(np.int32)
            seeds[b] = int(rng.integers(0, 2**31 - 1))
        self.bootstrap_indices = boot
        self.tree_seeds = seeds

        params = dict(self.base_params)
        if self.n_jobs == 1:
            results = [
                _fit_one_tree(X, y, boot[b], params, int(seeds[b]))
                for b in range(self.n_estimators)
            ]
        else:
            results = Parallel(n_jobs=self.n_jobs, prefer="processes", batch_size=1)(
                delayed(_fit_one_tree)(X, y, boot[b], params, int(seeds[b]))
                for b in range(self.n_estimators)
            )
        self.trees = [r[0] for r in results]
        self.fit_times = np.asarray([r[1] for r in results], dtype=np.float64)
        return self

    def _check_fitted(self) -> None:
        if len(self.trees) != self.n_estimators:
            raise RuntimeError("BaggingEnsemble is not fitted")

    def predict_matrix(self, X: np.ndarray, n_estimators: int | None = None) -> np.ndarray:
        self._check_fitted()
        B = self.n_estimators if n_estimators is None else int(n_estimators)
        if not 1 <= B <= self.n_estimators:
            raise ValueError(f"n_estimators must be in [1, {self.n_estimators}]")
        return np.stack([tree.predict(X) for tree in self.trees[:B]], axis=0)

    def vote(self, X: np.ndarray, n_estimators: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = self.predict_matrix(X, n_estimators)
        return vote_from_predictions(preds)

    def oob_stats(self, X: np.ndarray, y: np.ndarray, n_estimators: int | None = None) -> dict[str, float]:
        self._check_fitted()
        B = self.n_estimators if n_estimators is None else int(n_estimators)
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        votes = np.zeros((n, N_CLASSES), dtype=np.int32)
        counts = np.zeros(n, dtype=np.int32)
        for b in range(B):
            oob = np.ones(n, dtype=bool)
            oob[np.unique(self.bootstrap_indices[b])] = False
            if not np.any(oob):
                continue
            idx = np.flatnonzero(oob)
            pred = self.trees[b].predict(X[idx])
            votes[idx, pred] += 1
            counts[idx] += 1
        covered = counts > 0
        if not np.any(covered):
            return {"oob_accuracy": float("nan"), "oob_coverage": 0.0}
        pred = np.argmax(votes[covered], axis=1)
        return {
            "oob_accuracy": float(np.mean(pred == y[covered])),
            "oob_coverage": float(np.mean(covered)),
        }

    def distinct_counts(self, n_estimators: int | None = None) -> np.ndarray:
        B = self.n_estimators if n_estimators is None else int(n_estimators)
        return np.asarray(
            [len(np.unique(self.bootstrap_indices[b])) for b in range(B)], dtype=np.int32
        )

    def feature_importances(self, n_estimators: int | None = None) -> np.ndarray:
        B = self.n_estimators if n_estimators is None else int(n_estimators)
        return np.mean(
            np.stack([t.feature_importances_ for t in self.trees[:B]], axis=0), axis=0
        )

    def serial_training_time(self, n_estimators: int | None = None) -> float:
        B = self.n_estimators if n_estimators is None else int(n_estimators)
        return float(np.sum(self.fit_times[:B]))


# ---------------------------------------------------------------------------
# SAMME (multiclass AdaBoost), implemented here
# ---------------------------------------------------------------------------
class SAMME:
    """SAMME multiclass AdaBoost with decision-tree weak learners.

    The update rule is the one in the assignment:

        alpha_t = log((1 - err_t) / err_t) + log(K - 1)
        w_i <- w_i * exp(alpha_t * 1[h_t(x_i) != y_i]),  then normalise.

    A weak learner is accepted only if its weighted error is strictly below
    the random-guess threshold ``1 - 1/K``.  If it is not, boosting stops and
    the reason is recorded; this is explicit handling rather than silently
    assigning a meaningless alpha.
    """

    def __init__(
        self,
        max_depth: int | None,
        n_estimators: int,
        seed: int,
        min_samples_leaf: int = 1,
        epsilon: float = 1e-12,
    ) -> None:
        self.max_depth = max_depth
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.min_samples_leaf = int(min_samples_leaf)
        self.epsilon = float(epsilon)
        self.models: list[DecisionTreeClassifier] = []
        self.alphas_: list[float] = []
        self.classes_ = np.arange(N_CLASSES, dtype=int)
        self.n_classes_ = N_CLASSES
        self.weight_history: list[np.ndarray] = []
        self.round_predictions: list[np.ndarray] = []
        self.history: list[dict[str, float]] = []
        self.stop_reason = ""
        self.failed_ = False
        self.n_estimators_fitted_ = 0
        self.total_alpha_ = 0.0
        self.train_seconds_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAMME":
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("SAMME requires at least two classes")
        n = len(y)
        K = self.n_classes_
        rng = np.random.default_rng(self.seed)
        w = np.full(n, 1.0 / n, dtype=np.float64)
        self.weight_history = [w.copy()]
        self.round_predictions = []
        self.models = []
        self.alphas_ = []
        self.history = []
        self.stop_reason = ""
        self.failed_ = False
        t_start = time.perf_counter()

        for t in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X, y, sample_weight=w)
            pred = tree.predict(X)
            err = float(np.sum(w[pred != y]))
            threshold = 1.0 - 1.0 / K
            if err >= threshold:
                self.stop_reason = (
                    f"rejected round {t + 1}: weighted error {err:.6f} >= "
                    f"random threshold {threshold:.6f}"
                )
                break
            if err <= self.epsilon:
                alpha = float(np.log((1.0 - self.epsilon) / self.epsilon) + np.log(K - 1))
                perfect = True
            else:
                alpha = float(np.log((1.0 - err) / err) + np.log(K - 1))
                perfect = False
            self.models.append(tree)
            self.alphas_.append(alpha)
            self.round_predictions.append(pred.astype(np.int16))
            w = w * np.exp(np.clip(alpha * (pred != y), -50.0, 50.0))
            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0:
                raise FloatingPointError("SAMME weight normalisation failed")
            w /= total
            self.weight_history.append(w.copy())
            self.history.append(
                {
                    "round": t + 1,
                    "weighted_error": err,
                    "alpha": alpha,
                    "max_weight": float(np.max(w)),
                    "top10_weight": float(np.sum(np.sort(w)[-10:])),
                    "ensemble_train_accuracy": float(
                        np.mean(self.predict(X) == y)
                    ) if len(self.models) <= 10 else float("nan"),
                }
            )
            if perfect:
                self.stop_reason = f"perfect weak learner at round {t + 1}"
                break
        else:
            self.stop_reason = "requested number of rounds completed"

        self.n_estimators_fitted_ = len(self.models)
        self.total_alpha_ = float(np.sum(self.alphas_))
        self.train_seconds_ = time.perf_counter() - t_start
        if not self.models:
            self.failed_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.models:
            raise RuntimeError(f"SAMME has no accepted weak learner: {self.stop_reason}")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X)
        scores = np.zeros((len(X), self.n_classes_), dtype=np.float64)
        rows = np.arange(len(X))
        for tree, alpha in zip(self.models, self.alphas_):
            pred = tree.predict(X)
            cols = np.searchsorted(self.classes_, pred)
            np.add.at(scores, (rows, cols), alpha)
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Normalised SAMME weighted vote q_k(x) = S_k(x) / sum_t alpha_t."""
        self._check_fitted()
        scores = self.decision_function(X)
        total = float(np.sum(self.alphas_))
        if total <= 0:
            return np.full_like(scores, 1.0 / self.n_classes_)
        return scores / total

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def feature_importances(self) -> np.ndarray:
        self._check_fitted()
        alphas = np.asarray(self.alphas_, dtype=np.float64)
        imps = np.stack([t.feature_importances_ for t in self.models], axis=0)
        return np.average(imps, axis=0, weights=alphas)

    def history_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ---------------------------------------------------------------------------
# Bias-variance
# ---------------------------------------------------------------------------
def bias_variance_from_probabilities(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_classes: int = N_CLASSES,
) -> tuple[float, float]:
    """Compute the assignment's measured squared bias and variance.

    probabilities has shape (R, N, K).  With pbar_i the mean over the R
    repetitions and e_i the one-hot target:

        Bias^2   = (1/N)   sum_i ||pbar_i - e_i||^2
        Variance = (1/(NR)) sum_{i,r} ||p_r(x_i) - pbar_i||^2
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must be (n_repeats, n_samples, n_classes)")
    if probabilities.shape[1] != len(y_true):
        raise ValueError("probability sample dimension does not match y_true")
    if probabilities.shape[2] != n_classes:
        raise ValueError("unexpected number of classes")
    y_true = np.asarray(y_true, dtype=int)
    pbar = probabilities.mean(axis=0)
    onehot = np.eye(n_classes, dtype=np.float64)[y_true]
    bias_sq = float(np.mean(np.sum((pbar - onehot) ** 2, axis=1)))
    variance = float(np.mean(np.sum((probabilities - pbar[None, :, :]) ** 2, axis=2)))
    return bias_sq, variance


# ---------------------------------------------------------------------------
# Environment / reproducibility helpers
# ---------------------------------------------------------------------------
def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scikit_learn": _version("scikit-learn"),
        "pandas": pd.__version__,
        "matplotlib": _version("matplotlib"),
        "joblib": _version("joblib"),
        "scipy": _version("scipy"),
        "pillow": _version("Pillow"),
    }


def git_commit() -> str:
    candidates = ["git"]
    for candidate in (
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
    ):
        if Path(candidate).exists():
            candidates.append(candidate)
    for exe in candidates:
        try:
            out = subprocess.check_output(
                [exe, "rev-parse", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
            if len(out) == 40:
                return out
        except Exception:
            continue
    return "UNCOMMITTED"


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_manifest(paths: Iterable[Path | str], out_path: Path | str | None = None) -> dict[str, Any]:
    entries = []
    for p in sorted(Path(x) for x in paths):
        if not p.exists() or not p.is_file():
            continue
        entries.append(
            {
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            }
        )
    manifest = {"root": str(ROOT), "files": entries}
    if out_path is not None:
        json_dump(out_path, manifest)
    return manifest
