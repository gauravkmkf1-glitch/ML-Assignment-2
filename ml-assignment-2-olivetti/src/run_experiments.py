"""Run every experiment required by Machine Learning Assignment 2.

Usage (from the repository root):

    python -m src.run_experiments

All selections use the clean validation split only.  The clean test set is
touched only for the final reporting requested by the assignment, and the
masked test set is never used for training or selection.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from .common import (
    CONFIGS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    N_CLASSES,
    RESULTS_DIR,
    ROOT,
    ROLL_NUMBER,
    SEED,
    BaggingEnsemble,
    SAMME,
    accuracy,
    align_proba,
    artifact_manifest,
    bias_variance_from_probabilities,
    central_feature_indices,
    dataframe_to_csv,
    dataset_checksum,
    derive_seed,
    environment_info,
    feature_mass,
    git_commit,
    json_dump,
    json_load,
    leaf_stats,
    load_dataset,
    load_split,
    make_split,
    mask_centered_block,
    mean_pairwise_disagreement,
    save_split,
    setup_logging,
    sha256_file,
    split_checksum,
    validate_split,
    vote_from_predictions,
    vote_margin,
    winning_confidence,
)

LOG = None


def log(message: str) -> None:
    if LOG is not None:
        LOG.info(message)
    else:
        print(message)


# ---------------------------------------------------------------------------
# Configuration and small helpers
# ---------------------------------------------------------------------------
def load_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else CONFIGS_DIR / "config.json"
    cfg = json_load(path)
    if cfg["student"]["name"] != "Kumar Gaurav" or cfg["student"]["roll_number"] != ROLL_NUMBER:
        raise ValueError("config.json student identity does not match the submission")
    digits = "".join(ch for ch in ROLL_NUMBER if ch.isdigit())
    if int(digits) != SEED:
        raise ValueError("Roll-number seed mismatch")
    if cfg["split"]["train_per_identity"] != 6:
        raise ValueError("The assignment requires 6 training images per identity")
    if cfg["split"]["validation_per_identity"] != 2 or cfg["split"]["test_per_identity"] != 2:
        raise ValueError("The assignment requires 2 validation and 2 test images per identity")
    return cfg


def n_jobs_from_config(cfg: dict[str, Any]) -> int:
    requested = int(cfg.get("compute", {}).get("n_jobs", 1))
    cpus = os.cpu_count() or 2
    return max(1, min(requested, cpus - 1))


def selected_setting(rows: Sequence[dict[str, Any]], key: str = "validation_accuracy") -> dict[str, Any]:
    """First maximum in grid order (ties therefore keep the earlier setting)."""
    best = None
    best_value = -np.inf
    for row in rows:
        value = row.get(key, float("nan"))
        if np.isfinite(value) and value > best_value:
            best = row
            best_value = value
    if best is None:
        raise RuntimeError("No valid model to select")
    return best


def probability_topk(proba: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
    proba = np.asarray(proba)
    order = np.argsort(proba)[::-1][:k]
    return [{"class": int(c), "score": float(proba[c])} for c in order]


def vote_topk(counts: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
    counts = np.asarray(counts)
    order = np.argsort(counts)[::-1][:k]
    return [{"class": int(c), "votes": int(counts[c])} for c in order if counts[c] > 0]


def serialise_tree_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_depth": None if params["max_depth"] is None else int(params["max_depth"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
    }


def fit_single_tree(X: np.ndarray, y: np.ndarray, params: dict[str, Any], seed: int = SEED) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(random_state=int(seed), **params)
    model.fit(X, y)
    return model


def model_metrics(
    name: str,
    clean_pred: np.ndarray,
    clean_proba: np.ndarray,
    masked_pred: np.ndarray,
    masked_proba: np.ndarray,
    y_clean: np.ndarray,
    y_masked: np.ndarray,
    central_mass: float,
) -> dict[str, Any]:
    clean_conf = winning_confidence(clean_proba)
    masked_conf = winning_confidence(masked_proba)
    clean_correct = clean_pred == y_clean
    masked_correct = masked_pred == y_masked
    return {
        "model": name,
        "clean_test_accuracy": accuracy(y_clean, clean_pred),
        "masked_test_accuracy": accuracy(y_masked, masked_pred),
        "clean_accuracy_drop": float(accuracy(y_clean, clean_pred) - accuracy(y_masked, masked_pred)),
        "clean_mean_confidence": float(np.mean(clean_conf)),
        "masked_mean_confidence": float(np.mean(masked_conf)),
        "clean_mean_confidence_correct": float(np.mean(clean_conf[clean_correct])) if np.any(clean_correct) else float("nan"),
        "masked_mean_confidence_correct": float(np.mean(masked_conf[masked_correct])) if np.any(masked_correct) else float("nan"),
        "clean_mean_margin": float(np.mean(vote_margin(clean_proba))),
        "masked_mean_margin": float(np.mean(vote_margin(masked_proba))),
        "clean_n_correct": int(np.sum(clean_correct)),
        "masked_n_correct": int(np.sum(masked_correct)),
        "central_feature_importance_mass": float(central_mass),
    }


# ---------------------------------------------------------------------------
# Q1(a): single-tree grid
# ---------------------------------------------------------------------------
def q1a_tree_grid(
    cfg: dict[str, Any],
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
) -> tuple[dict[str, Any], DecisionTreeClassifier, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for depth in cfg["tree_grid"]["max_depth"]:
        for leaf in cfg["tree_grid"]["min_samples_leaf"]:
            t0 = time.perf_counter()
            tree = fit_single_tree(Xtr, ytr, {"max_depth": depth, "min_samples_leaf": leaf})
            elapsed = time.perf_counter() - t0
            stats = leaf_stats(tree)
            rows.append(
                {
                    "max_depth": "None" if depth is None else int(depth),
                    "max_depth_value": None if depth is None else int(depth),
                    "min_samples_leaf": int(leaf),
                    "train_accuracy": accuracy(ytr, tree.predict(Xtr)),
                    "validation_accuracy": accuracy(yva, tree.predict(Xva)),
                    "test_accuracy": accuracy(yte, tree.predict(Xte)),
                    "leaves": stats["leaves"],
                    "mean_samples_per_leaf": stats["mean_samples_per_leaf"],
                    "singleton_leaf_fraction": stats["singleton_leaf_fraction"],
                    "fit_seconds": float(elapsed),
                }
            )
            log(
                f"Q1a depth={rows[-1]['max_depth']:>4} leaf={leaf} "
                f"train={rows[-1]['train_accuracy']:.4f} val={rows[-1]['validation_accuracy']:.4f} "
                f"test={rows[-1]['test_accuracy']:.4f} leaves={stats['leaves']}"
            )
    df = dataframe_to_csv(rows, RESULTS_DIR / "tree_grid.csv")
    best = selected_setting(rows, "validation_accuracy")
    params = {
        "max_depth": None if best["max_depth_value"] is None else int(best["max_depth_value"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
    }
    tree = fit_single_tree(Xtr, ytr, params)
    json_dump(
        RESULTS_DIR / "selected_tree.json",
        {
            "selected_by": "clean validation accuracy only",
            "tie_break": "first row in grid order (shallower depth, then smaller min_samples_leaf)",
            "params": serialise_tree_params(params),
            "row": best,
        },
    )
    log(f"Q1a selected single tree: {params} (validation={best['validation_accuracy']:.4f})")
    return best, tree, df


# ---------------------------------------------------------------------------
# Q1(b,c): bagging
# ---------------------------------------------------------------------------
def q1bc_bagging(
    cfg: dict[str, Any],
    tree_params: dict[str, Any],
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    Xmask: np.ndarray,
    n_jobs: int,
) -> tuple[BaggingEnsemble, int, pd.DataFrame, dict[str, np.ndarray]]:
    max_b = int(cfg["bagging"]["n_estimators_max"])
    bag_seed = derive_seed(SEED, "bagging")
    t0 = time.perf_counter()
    bag = BaggingEnsemble(tree_params, max_b, seed=bag_seed, n_jobs=n_jobs).fit(Xtr, ytr)
    wall = time.perf_counter() - t0
    log(f"Q1b fitted {max_b} bootstrap trees in {wall:.2f} s wall clock")

    np.savez_compressed(
        RESULTS_DIR / "bagging_bootstrap.npz",
        bootstrap_indices=bag.bootstrap_indices,
        tree_seeds=bag.tree_seeds,
        fit_times=bag.fit_times,
        distinct_counts=bag.distinct_counts(max_b),
        n_train=np.int64(len(ytr)),
        seed=np.int64(bag_seed),
    )

    Ptr = bag.predict_matrix(Xtr, max_b)
    Pva = bag.predict_matrix(Xva, max_b)
    Pte = bag.predict_matrix(Xte, max_b)
    Pmask = bag.predict_matrix(Xmask, max_b)

    n = len(ytr)
    theoretical = float(n * (1.0 - (1.0 - 1.0 / n) ** n))
    rows: list[dict[str, Any]] = []
    for B in cfg["bagging"]["B_grid"]:
        B = int(B)
        ctr, ftr, ptr = vote_from_predictions(Ptr[:B])
        cva, fva, pva = vote_from_predictions(Pva[:B])
        cte, fte, pte = vote_from_predictions(Pte[:B])
        cmask, fmask, pmask = vote_from_predictions(Pmask[:B])
        oob = bag.oob_stats(Xtr, ytr, B)
        distinct = bag.distinct_counts(B)
        row = {
            "B": B,
            "train_accuracy": accuracy(ytr, ptr),
            "validation_accuracy": accuracy(yva, pva),
            "test_accuracy": accuracy(yte, pte),
            "masked_test_accuracy": accuracy(yte, pmask),
            "oob_accuracy": oob["oob_accuracy"],
            "oob_coverage": oob["oob_coverage"],
            "mean_distinct": float(np.mean(distinct)),
            "std_distinct": float(np.std(distinct, ddof=1)) if len(distinct) > 1 else 0.0,
            "theoretical_distinct": theoretical,
            "pairwise_disagreement_validation": mean_pairwise_disagreement(Pva[:B]),
            "pairwise_disagreement_test": mean_pairwise_disagreement(Pte[:B]),
            "mean_vote_margin_validation": float(np.mean(vote_margin(fva))),
            "mean_vote_margin_test": float(np.mean(vote_margin(fte))),
            "mean_confidence_validation": float(np.mean(winning_confidence(fva))),
            "mean_confidence_test": float(np.mean(winning_confidence(fte))),
            "mean_confidence_correct_validation": (
                float(np.mean(winning_confidence(fva)[pva == yva])) if np.any(pva == yva) else float("nan")
            ),
            "train_seconds_serial": bag.serial_training_time(B),
            "train_seconds_wall_full_200": wall,
        }
        rows.append(row)
        log(
            f"Q1b/c B={B:3d} val={row['validation_accuracy']:.4f} test={row['test_accuracy']:.4f} "
            f"oob={row['oob_accuracy']:.4f} dis={row['pairwise_disagreement_validation']:.4f} "
            f"margin={row['mean_vote_margin_validation']:.4f}"
        )

    df = dataframe_to_csv(rows, RESULTS_DIR / "bagging_grid.csv")
    # Validation only.  Ties are broken toward the smaller ensemble, which is
    # the parsimonious choice and is fixed before looking at test results.
    best = max(rows, key=lambda r: (r["validation_accuracy"], -r["B"]))
    json_dump(
        RESULTS_DIR / "selected_bagging.json",
        {
            "selected_by": "clean validation accuracy only",
            "tie_break": "smallest B among validation ties",
            "selected_B": int(best["B"]),
            "row": best,
            "B_grid": [int(b) for b in cfg["bagging"]["B_grid"]],
            "n_estimators_max_fitted": max_b,
        },
    )
    log(
        f"Q1c selected bagging B={best['B']} "
        f"(validation={best['validation_accuracy']:.4f}, test={best['test_accuracy']:.4f})"
    )
    return bag, int(best["B"]), df, {
        "Ptr": Ptr,
        "Pva": Pva,
        "Pte": Pte,
        "Pmask": Pmask,
    }


# ---------------------------------------------------------------------------
# Q1(d): case studies
# ---------------------------------------------------------------------------
def q1d_cases(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    tree: DecisionTreeClassifier,
    bag: BaggingEnsemble,
    selected_B: int,
    Xte: np.ndarray,
    Xmask: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    tree_proba = align_proba(tree.predict_proba(Xte), tree.classes_)
    tree_pred = tree.predict(Xte)
    tree_proba_mask = align_proba(tree.predict_proba(Xmask), tree.classes_)
    tree_pred_mask = tree.predict(Xmask)

    counts, frac, bag_pred = vote_from_predictions(predictions["Pte"][:selected_B])
    counts_mask, frac_mask, bag_pred_mask = vote_from_predictions(predictions["Pmask"][:selected_B])
    bag_conf = winning_confidence(frac)
    bag_conf_mask = winning_confidence(frac_mask)

    clean_correct_bag = bag_pred == yte
    clean_correct_tree = tree_pred == yte
    corrections = [i for i in range(len(yte)) if (not clean_correct_tree[i]) and clean_correct_bag[i]]
    both_wrong = [i for i in range(len(yte)) if (not clean_correct_tree[i]) and (not clean_correct_bag[i])]

    def pick(indices: list[int], used: set[int]) -> list[int]:
        ordered = sorted(indices, key=lambda i: float(bag_conf[i]), reverse=True)
        picks: list[int] = []
        for i in ordered:
            if int(yte[i]) not in used:
                picks.append(i)
                used.add(int(yte[i]))
                if len(picks) == 2:
                    return picks
        for i in ordered:
            if i not in picks:
                picks.append(i)
                if len(picks) == 2:
                    return picks
        return picks

    used: set[int] = set()
    correction_positions = pick(corrections, used)
    used = set()
    both_wrong_positions = pick(both_wrong, used)

    def describe(i: int, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "test_position": int(i),
            "dataset_index": int(test_idx[i]),
            "true_identity": int(yte[i]),
            "tree_prediction": int(tree_pred[i]),
            "bag_prediction": int(bag_pred[i]),
            "tree_confidence": float(np.max(tree_proba[i])),
            "bag_confidence": float(bag_conf[i]),
            "bag_vote_counts": counts[i].astype(int).tolist(),
            "bag_vote_fractions": frac[i].astype(float).tolist(),
            "tree_probabilities": tree_proba[i].astype(float).tolist(),
            "tree_top5": probability_topk(tree_proba[i], 5),
            "bag_top5": vote_topk(counts[i], 5),
            "masked_tree_prediction": int(tree_pred_mask[i]),
            "masked_bag_prediction": int(bag_pred_mask[i]),
            "masked_bag_confidence": float(bag_conf_mask[i]),
        }

    result = {
        "test_set_size": int(len(yte)),
        "n_corrections_bag_over_tree": int(len(corrections)),
        "n_both_wrong": int(len(both_wrong)),
        "corrections": [describe(i, "bagging_corrects_tree") for i in correction_positions],
        "both_wrong": [describe(i, "both_wrong") for i in both_wrong_positions],
    }
    json_dump(RESULTS_DIR / "q1_cases.json", result)
    np.savez_compressed(
        RESULTS_DIR / "q1_test_predictions.npz",
        y_test=yte,
        test_indices=test_idx,
        tree_pred=tree_pred,
        tree_proba=tree_proba,
        tree_pred_masked=tree_pred_mask,
        tree_proba_masked=tree_proba_mask,
        bag_votes=counts,
        bag_fractions=frac,
        bag_pred=bag_pred,
        bag_votes_masked=counts_mask,
        bag_fractions_masked=frac_mask,
        bag_pred_masked=bag_pred_mask,
    )
    log(
        f"Q1d cases: {len(corrections)} corrections, {len(both_wrong)} both-wrong; "
        f"wrote q1_cases.json"
    )
    return result


# ---------------------------------------------------------------------------
# Q2(a): toy SAMME round
# ---------------------------------------------------------------------------
def q2a_toy() -> dict[str, Any]:
    toy_X = np.asarray([[0.0], [0.1], [0.2], [0.4], [0.5], [0.6], [1.0], [1.1], [1.2]], dtype=np.float32)
    toy_y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
    model = SAMME(max_depth=1, n_estimators=1, seed=derive_seed(SEED, "toy")).fit(toy_X, toy_y)
    if model.failed_ or len(model.models) != 1:
        raise RuntimeError(f"Toy SAMME round was not accepted: {model.stop_reason}")
    pred = model.round_predictions[0].astype(int)
    initial = model.weight_history[0]
    updated = model.weight_history[1]
    row = model.history[0]
    payload = {
        "description": "Three classes, three one-dimensional samples per class, one depth-1 stump.",
        "X": toy_X.ravel().astype(float).tolist(),
        "y": toy_y.astype(int).tolist(),
        "classes": model.classes_.astype(int).tolist(),
        "weak_learner": "DecisionTreeClassifier(max_depth=1)",
        "initial_weights": initial.astype(float).tolist(),
        "weak_predictions": pred.tolist(),
        "misclassified_positions": np.flatnonzero(pred != toy_y).astype(int).tolist(),
        "weighted_error": float(row["weighted_error"]),
        "alpha": float(row["alpha"]),
        "updated_normalised_weights": updated.astype(float).tolist(),
        "weight_sum_after_update": float(np.sum(updated)),
        "stop_reason": model.stop_reason,
    }
    json_dump(RESULTS_DIR / "toy_samme.json", payload)

    lines = [
        "| i | x_i | y_i | initial w_i | h_1(x_i) | correct? | updated w_i |",
        "|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for i in range(len(toy_y)):
        lines.append(
            f"| {i} | {toy_X[i, 0]:.1f} | {toy_y[i]} | {initial[i]:.6f} | {pred[i]} | "
            f"{'yes' if pred[i] == toy_y[i] else 'no'} | {updated[i]:.6f} |"
        )
    lines += [
        "",
        f"Weighted error: **{row['weighted_error']:.6f}**; "
        f"alpha_1 = log((1-e)/e) + log(K-1) = **{row['alpha']:.6f}**.",
        f"Updated weights sum to {np.sum(updated):.6f}.",
    ]
    (RESULTS_DIR / "toy_samme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Q2a toy round: error={row['weighted_error']:.4f} alpha={row['alpha']:.4f}")
    return payload


# ---------------------------------------------------------------------------
# Q2(b): boosting grid and selected diagnostics
# ---------------------------------------------------------------------------
def q2b_boosting(
    cfg: dict[str, Any],
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[SAMME, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    models: dict[tuple[int, int], SAMME] = {}
    for depth in cfg["boosting"]["max_depth"]:
        for rounds in cfg["boosting"]["n_estimators"]:
            depth = int(depth)
            rounds = int(rounds)
            model = SAMME(
                max_depth=depth,
                n_estimators=rounds,
                seed=derive_seed(SEED, "samme", depth, rounds),
            ).fit(Xtr, ytr)
            models[(depth, rounds)] = model
            if model.failed_:
                train_acc = val_acc = test_acc = float("nan")
            else:
                train_acc = accuracy(ytr, model.predict(Xtr))
                val_acc = accuracy(yva, model.predict(Xva))
                test_acc = accuracy(yte, model.predict(Xte))
            last = model.history[-1] if model.history else {}
            first = model.history[0] if model.history else {}
            rows.append(
                {
                    "depth": depth,
                    "requested_rounds": rounds,
                    "actual_rounds": int(model.n_estimators_fitted_),
                    "train_accuracy": float(train_acc),
                    "validation_accuracy": float(val_acc),
                    "test_accuracy": float(test_acc),
                    "train_seconds": float(model.train_seconds_),
                    "first_weighted_error": float(first.get("weighted_error", float("nan"))),
                    "last_weighted_error": float(last.get("weighted_error", float("nan"))),
                    "last_alpha": float(last.get("alpha", float("nan"))),
                    "last_max_weight": float(last.get("max_weight", float("nan"))),
                    "last_top10_weight": float(last.get("top10_weight", float("nan"))),
                    "central_feature_importance_mass": (
                        float(feature_mass(model.feature_importances(), central_feature_indices()))
                        if not model.failed_
                        else float("nan")
                    ),
                    "stop_reason": model.stop_reason,
                }
            )
            log(
                f"Q2b depth={depth} rounds={rounds:3d} actual={model.n_estimators_fitted_:3d} "
                f"val={val_acc:.4f} test={test_acc:.4f} "
                f"last_err={rows[-1]['last_weighted_error']:.4f} time={model.train_seconds_:.1f}s"
            )
    df = dataframe_to_csv(rows, RESULTS_DIR / "boosting_grid.csv")
    eligible = [r for r in rows if r["actual_rounds"] > 0 and np.isfinite(r["validation_accuracy"])]
    best = selected_setting(eligible, "validation_accuracy")
    selected_key = (int(best["depth"]), int(best["requested_rounds"]))
    model = models[selected_key]
    if model.failed_:
        raise RuntimeError("Selected boosting model unexpectedly has no weak learner")
    history_df = dataframe_to_csv(model.history, RESULTS_DIR / "boosting_history.csv")
    np.savez_compressed(
        RESULTS_DIR / "boosting_weight_history.npz",
        weight_history=np.stack(model.weight_history, axis=0),
        round_predictions=np.stack(model.round_predictions, axis=0),
        alphas=np.asarray(model.alphas_, dtype=float),
        weighted_errors=np.asarray([h["weighted_error"] for h in model.history], dtype=float),
        train_indices=train_idx,
        y_train=ytr,
        selected_depth=np.int64(model.max_depth if model.max_depth is not None else -1),
        selected_rounds=np.int64(model.n_estimators),
    )
    json_dump(
        RESULTS_DIR / "selected_boosting.json",
        {
            "selected_by": "clean validation accuracy only",
            "tie_break": "first row in grid order (shallower depth, then fewer requested rounds)",
            "selected": {
                "max_depth": int(model.max_depth) if model.max_depth is not None else None,
                "n_estimators_requested": int(model.n_estimators),
                "n_estimators_fitted": int(model.n_estimators_fitted_),
                "stop_reason": model.stop_reason,
            },
            "row": best,
            "grid": cfg["boosting"],
        },
    )

    # Inspect the training face with the largest final boosting weight.
    final_w = model.weight_history[-1]
    position = int(np.argmax(final_w))
    T = len(model.models)
    scores = np.zeros(N_CLASSES, dtype=float)
    hist_rows = []
    n_misclassified = 0
    for t in range(T):
        pred = int(model.round_predictions[t][position])
        if pred != int(ytr[position]):
            n_misclassified += 1
        scores[pred] += float(model.alphas_[t])
        cumulative_pred = int(np.argmax(scores))
        hist_rows.append(
            {
                "round": t + 1,
                "weight_before": float(model.weight_history[t][position]),
                "weak_prediction": pred,
                "weak_correct": bool(pred == int(ytr[position])),
                "alpha": float(model.alphas_[t]),
                "weight_after": float(model.weight_history[t + 1][position]),
                "cumulative_ensemble_prediction": cumulative_pred,
                "cumulative_ensemble_correct": bool(cumulative_pred == int(ytr[position])),
            }
        )
    weight_case = {
        "selection": "training position with the largest final SAMME sample weight",
        "train_position": position,
        "dataset_index": int(train_idx[position]),
        "true_identity": int(ytr[position]),
        "final_weight": float(final_w[position]),
        "max_weight_over_rounds": float(np.max(np.stack(model.weight_history, axis=0)[:, position])),
        "mean_weight_over_rounds": float(np.mean(np.stack(model.weight_history, axis=0)[:, position])),
        "uniform_weight": float(1.0 / len(ytr)),
        "n_misclassified": int(n_misclassified),
        "n_rounds": int(T),
        "first_correct_cumulative_round": next(
            (r["round"] for r in hist_rows if r["cumulative_ensemble_correct"]), None
        ),
        "history": hist_rows,
    }
    json_dump(RESULTS_DIR / "q2_weight_case.json", weight_case)
    log(
        f"Q2b selected depth={model.max_depth} rounds={model.n_estimators} "
        f"(val={best['validation_accuracy']:.4f}, test={best['test_accuracy']:.4f}); "
        f"max-weight train face {weight_case['dataset_index']} (identity {weight_case['true_identity']})"
    )
    return model, best, df, history_df


# ---------------------------------------------------------------------------
# Q2(c): boosting vs bagging at 25/50/100 learners
# ---------------------------------------------------------------------------
def q2c_comparison(
    cfg: dict[str, Any],
    tree_params: dict[str, Any],
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    bag: BaggingEnsemble,
    bag_rows: pd.DataFrame,
    boost_models: dict[tuple[int, int], SAMME],
    selected_boost: dict[str, Any],
    n_jobs: int,
) -> pd.DataFrame:
    bag_seed = derive_seed(SEED, "bagging")
    rows: list[dict[str, Any]] = []
    for L in (25, 50, 100):
        # Refit the prefix (identical bootstrap replicates and tree seeds) to
        # measure wall-clock training time rather than estimating it.
        t0 = time.perf_counter()
        prefix = BaggingEnsemble(tree_params, L, seed=bag_seed, n_jobs=n_jobs).fit(Xtr, ytr)
        bag_time = time.perf_counter() - t0
        _, _, pred_tr = prefix.vote(Xtr, L)
        _, _, pred_va = prefix.vote(Xva, L)
        _, _, pred_te = prefix.vote(Xte, L)
        # Verify that the prefix really matches the larger fitted ensemble.
        full_prefix_pred = bag.predict_matrix(Xte, L)
        if not np.array_equal(full_prefix_pred, prefix.predict_matrix(Xte, L)):
            raise RuntimeError("Bagging prefix is not reproducible")
        rows.append(
            {
                "family": "Bagging",
                "n_learners": int(L),
                "train_accuracy": accuracy(ytr, pred_tr),
                "validation_accuracy": accuracy(yva, pred_va),
                "test_accuracy": accuracy(yte, pred_te),
                # Sum of per-tree fit times: a stable serial-equivalent measure
                # that is not distorted by parallel scheduling on this machine.
                "train_seconds": float(bag.serial_training_time(L)),
                "wall_seconds": float(bag_time),
                "actual_rounds": int(L),
                "note": "bootstrap trees; frozen tree setting; serial-equivalent time",
            }
        )
        depth = int(selected_boost["depth"])
        boost = boost_models[(depth, L)]
        if boost.failed_:
            train_acc = val_acc = test_acc = float("nan")
        else:
            train_acc = accuracy(ytr, boost.predict(Xtr))
            val_acc = accuracy(yva, boost.predict(Xva))
            test_acc = accuracy(yte, boost.predict(Xte))
        rows.append(
            {
                "family": "Boosting",
                "n_learners": int(L),
                "train_accuracy": float(train_acc),
                "validation_accuracy": float(val_acc),
                "test_accuracy": float(test_acc),
                "train_seconds": float(boost.train_seconds_),
                "wall_seconds": float(boost.train_seconds_),
                "actual_rounds": int(boost.n_estimators_fitted_),
                "note": f"SAMME depth {depth}; selected depth from Q2b",
            }
        )
        log(
            f"Q2c L={L}: bag test={rows[-2]['test_accuracy']:.4f} time={bag_time:.1f}s; "
            f"boost test={rows[-1]['test_accuracy']:.4f} actual={boost.n_estimators_fitted_} "
            f"time={boost.train_seconds_:.1f}s"
        )
    df = dataframe_to_csv(rows, RESULTS_DIR / "boosting_vs_bagging.csv")
    return df


# ---------------------------------------------------------------------------
# Q2(d): final comparison and masked-response case
# ---------------------------------------------------------------------------
def q2d_comparison(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    tree: DecisionTreeClassifier,
    bag: BaggingEnsemble,
    boost: SAMME,
    selected_B: int,
    Xte: np.ndarray,
    Xmask: np.ndarray,
    bag_predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tree_proba = align_proba(tree.predict_proba(Xte), tree.classes_)
    tree_proba_mask = align_proba(tree.predict_proba(Xmask), tree.classes_)
    tree_pred = tree.predict(Xte)
    tree_pred_mask = tree.predict(Xmask)

    counts, frac, bag_pred = vote_from_predictions(bag_predictions["Pte"][:selected_B])
    counts_mask, frac_mask, bag_pred_mask = vote_from_predictions(bag_predictions["Pmask"][:selected_B])

    boost_proba = align_proba(boost.predict_proba(Xte), boost.classes_)
    boost_proba_mask = align_proba(boost.predict_proba(Xmask), boost.classes_)
    boost_pred = np.argmax(boost_proba, axis=1)
    boost_pred_mask = np.argmax(boost_proba_mask, axis=1)

    rows = [
        model_metrics(
            "Single tree", tree_pred, tree_proba, tree_pred_mask, tree_proba_mask,
            yte, yte, feature_mass(tree.feature_importances_, central_feature_indices()),
        ),
        model_metrics(
            "Bagging", bag_pred, frac, bag_pred_mask, frac_mask,
            yte, yte, feature_mass(bag.feature_importances(selected_B), central_feature_indices()),
        ),
        model_metrics(
            "Boosting", boost_pred, boost_proba, boost_pred_mask, boost_proba_mask,
            yte, yte, feature_mass(boost.feature_importances(), central_feature_indices()),
        ),
    ]
    df = dataframe_to_csv(rows, RESULTS_DIR / "final_comparison.csv")

    preferred = [
        i for i in range(len(yte))
        if bag_pred_mask[i] == yte[i] and boost_pred_mask[i] != yte[i]
    ]
    direction = "bagging_correct_boosting_wrong"
    if not preferred:
        preferred = [
            i for i in range(len(yte))
            if boost_pred_mask[i] == yte[i] and bag_pred_mask[i] != yte[i]
        ]
        direction = "boosting_correct_bagging_wrong"
    if not preferred:
        preferred = [i for i in range(len(yte)) if bag_pred_mask[i] != boost_pred_mask[i]]
        direction = "different_prediction"
    if not preferred:
        raise RuntimeError("No test face found where bagging and boosting respond differently to the mask")
    # Prefer a case with a large boosting confidence on its masked prediction,
    # so that the contrast is visible and the explanation is well posed.
    case = max(preferred, key=lambda i: float(np.max(boost_proba_mask[i])))

    w_final = boost.weight_history[-1]
    top_positions = np.argsort(w_final)[::-1][:5]
    true_id = int(yte[case])
    pred_id = int(boost_pred_mask[case])
    payload = {
        "selection": (
            "test face where the clean-trained bagging and boosting models "
            "disagree on the masked test set; prefer bagging correct / boosting wrong, "
            "then the largest boosting confidence"
        ),
        "direction": direction,
        "test_position": int(case),
        "dataset_index": int(test_idx[case]),
        "true_identity": true_id,
        "clean": {
            "tree_prediction": int(tree_pred[case]),
            "bag_prediction": int(bag_pred[case]),
            "boost_prediction": int(boost_pred[case]),
            "tree_confidence": float(np.max(tree_proba[case])),
            "bag_confidence": float(np.max(frac[case])),
            "boost_confidence": float(np.max(boost_proba[case])),
        },
        "masked": {
            "tree_prediction": int(tree_pred_mask[case]),
            "bag_prediction": int(bag_pred_mask[case]),
            "boost_prediction": int(boost_pred_mask[case]),
            "tree_confidence": float(np.max(tree_proba_mask[case])),
            "bag_confidence": float(np.max(frac_mask[case])),
            "boost_confidence": float(np.max(boost_proba_mask[case])),
            "bag_vote_counts": counts_mask[case].astype(int).tolist(),
            "bag_vote_fractions": frac_mask[case].astype(float).tolist(),
            "bag_top5": vote_topk(counts_mask[case], 5),
            "boost_scores": boost_proba_mask[case].astype(float).tolist(),
            "boost_top5": probability_topk(boost_proba_mask[case], 5),
            "bag_disagreement": mean_pairwise_disagreement(
                bag_predictions["Pmask"][:selected_B, case].reshape(selected_B, 1)
            ),
            "bag_trees_voting_true": int(counts_mask[case, true_id]),
            "bag_trees_voting_pred": int(counts_mask[case, pred_id]),
            "bag_margin": float(vote_margin(frac_mask[case : case + 1])[0]),
            "boost_margin": float(vote_margin(boost_proba_mask[case : case + 1])[0]),
            "boost_true_score": float(boost_proba_mask[case, true_id]),
        },
        "clean_bag_disagreement": mean_pairwise_disagreement(
            bag_predictions["Pte"][:selected_B, case].reshape(selected_B, 1)
        ),
        "weighting": {
            "final_weight_max": float(np.max(w_final)),
            "final_weight_mean": float(np.mean(w_final)),
            "true_identity_weight_fraction": float(np.sum(w_final[ytr == true_id]) / np.sum(w_final)),
            "predicted_identity_weight_fraction": float(np.sum(w_final[ytr == pred_id]) / np.sum(w_final)),
            "top_weight_training_faces": [
                {
                    "train_position": int(p),
                    "dataset_index": int(train_idx[p]),
                    "true_identity": int(ytr[p]),
                    "final_weight": float(w_final[p]),
                }
                for p in top_positions
            ],
        },
        "feature_importance": {
            "central_mass_tree": float(feature_mass(tree.feature_importances_, central_feature_indices())),
            "central_mass_bagging": float(feature_mass(bag.feature_importances(selected_B), central_feature_indices())),
            "central_mass_boosting": float(feature_mass(boost.feature_importances(), central_feature_indices())),
        },
    }
    json_dump(RESULTS_DIR / "q2_mask_case.json", payload)
    np.savez_compressed(
        RESULTS_DIR / "final_test_predictions.npz",
        y_test=yte,
        test_indices=test_idx,
        tree_proba=tree_proba,
        tree_pred=tree_pred,
        tree_proba_masked=tree_proba_mask,
        tree_pred_masked=tree_pred_mask,
        bag_fractions=frac,
        bag_pred=bag_pred,
        bag_fractions_masked=frac_mask,
        bag_pred_masked=bag_pred_mask,
        boost_proba=boost_proba,
        boost_pred=boost_pred,
        boost_proba_masked=boost_proba_mask,
        boost_pred_masked=boost_pred_mask,
    )
    log(
        f"Q2d mask case: test position {case} (identity {true_id}), direction={direction}, "
        f"bag masked={bag_pred_mask[case]} boost masked={boost_pred_mask[case]}"
    )
    return df, payload


# ---------------------------------------------------------------------------
# Q3: measured bias/variance
# ---------------------------------------------------------------------------
def make_resamples(train_idx: np.ndarray, y: np.ndarray, n_repeats: int) -> np.ndarray:
    rng = np.random.default_rng(derive_seed(SEED, "q3", "resamples"))
    y_train = y[train_idx]
    positions_by_identity = [np.flatnonzero(y_train == identity) for identity in range(40)]
    resamples = np.empty((n_repeats, len(train_idx)), dtype=np.int32)
    for r in range(n_repeats):
        positions: list[int] = []
        for identity in range(40):
            positions.extend(
                int(p) for p in rng.choice(positions_by_identity[identity], size=6, replace=True)
            )
        resamples[r] = np.asarray(positions, dtype=np.int32)
    return resamples


def _refit_selected_worker(
    rep: int,
    positions: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    tree_params: dict[str, Any],
    bag_B: int,
    boost_depth: int,
    boost_rounds: int,
) -> dict[str, Any]:
    Xtr = X[train_idx[positions]]
    ytr = y[train_idx[positions]]
    Xte = X[test_idx]
    yte = y[test_idx]
    classes = np.arange(N_CLASSES, dtype=int)

    tree = DecisionTreeClassifier(random_state=derive_seed(SEED, "q3", "tree", rep), **tree_params)
    tree.fit(Xtr, ytr)
    tree_proba = align_proba(tree.predict_proba(Xte), tree.classes_)

    bag = BaggingEnsemble(
        tree_params, bag_B, seed=derive_seed(SEED, "q3", "bag", rep), n_jobs=1
    ).fit(Xtr, ytr)
    _, bag_proba, _ = bag.vote(Xte, bag_B)

    try:
        boost = SAMME(
            max_depth=boost_depth,
            n_estimators=boost_rounds,
            seed=derive_seed(SEED, "q3", "boost", rep),
        ).fit(Xtr, ytr)
        boost_proba = align_proba(boost.predict_proba(Xte), boost.classes_)
        boost_failed = 0
        boost_actual = int(boost.n_estimators_fitted_)
    except Exception:
        boost_proba = np.full((len(Xte), N_CLASSES), 1.0 / N_CLASSES, dtype=float)
        boost_failed = 1
        boost_actual = 0

    return {
        "rep": rep,
        "tree_proba": tree_proba,
        "bag_proba": bag_proba,
        "boost_proba": boost_proba,
        "tree_acc": accuracy(yte, np.argmax(tree_proba, axis=1)),
        "bag_acc": accuracy(yte, np.argmax(bag_proba, axis=1)),
        "boost_acc": accuracy(yte, np.argmax(boost_proba, axis=1)),
        "boost_failed": boost_failed,
        "boost_actual": boost_actual,
    }


def q3a_bias_variance(
    cfg: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    tree_params: dict[str, Any],
    selected_B: int,
    selected_boost: dict[str, Any],
    n_jobs: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_repeats = int(cfg["bias_variance"]["n_repeats"])
    resamples = make_resamples(train_idx, y, n_repeats)
    np.savez_compressed(
        RESULTS_DIR / "q3_resamples.npz",
        resamples=resamples,
        train_indices=train_idx,
        n_repeats=np.int64(n_repeats),
        seed=np.int64(derive_seed(SEED, "q3", "resamples")),
    )
    log(f"Q3a drawing {n_repeats} within-identity bootstrap training sets; refitting selected models")
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=1)(
        delayed(_refit_selected_worker)(
            r,
            resamples[r],
            X,
            y,
            train_idx,
            test_idx,
            tree_params,
            selected_B,
            int(selected_boost["depth"]),
            int(selected_boost["rounds"]),
        )
        for r in range(n_repeats)
    )
    log(f"Q3a refits finished in {time.perf_counter() - t0:.1f} s")

    tree_probs = np.stack([r["tree_proba"] for r in results], axis=0)
    bag_probs = np.stack([r["bag_proba"] for r in results], axis=0)
    boost_probs = np.stack([r["boost_proba"] for r in results], axis=0)
    yte = y[test_idx]
    np.savez_compressed(
        RESULTS_DIR / "q3_predictions.npz",
        tree_proba=tree_probs,
        bag_proba=bag_probs,
        boost_proba=boost_probs,
        y_test=yte,
        test_indices=test_idx,
        boost_failed=np.asarray([r["boost_failed"] for r in results], dtype=np.int8),
        boost_actual_rounds=np.asarray([r["boost_actual"] for r in results], dtype=np.int16),
    )

    rows = []
    for name, probs, accs in (
        ("Single tree", tree_probs, [r["tree_acc"] for r in results]),
        ("Bagging", bag_probs, [r["bag_acc"] for r in results]),
        ("Boosting", boost_probs, [r["boost_acc"] for r in results]),
    ):
        bias_sq, variance = bias_variance_from_probabilities(probs, yte)
        rows.append(
            {
                "model": name,
                "bias_squared": bias_sq,
                "variance": variance,
                "refit_accuracy_mean": float(np.mean(accs)),
                "refit_accuracy_std": float(np.std(accs, ddof=1)),
                "refit_accuracy_min": float(np.min(accs)),
                "refit_accuracy_max": float(np.max(accs)),
                "n_repeats": n_repeats,
            }
        )
        log(
            f"Q3a {name}: bias^2={bias_sq:.6f} variance={variance:.6f} "
            f"acc={np.mean(accs):.4f} +/- {np.std(accs, ddof=1):.4f}"
        )
    df = dataframe_to_csv(rows, RESULTS_DIR / "bias_variance.csv")
    return df, resamples


def _depth_worker(
    rep: int,
    positions: np.ndarray,
    depth: int | None,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    min_samples_leaf: int,
    bag_B: int,
) -> dict[str, Any]:
    Xtr = X[train_idx[positions]]
    ytr = y[train_idx[positions]]
    Xte = X[test_idx]
    yte = y[test_idx]
    tree = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=min_samples_leaf,
        random_state=derive_seed(SEED, "q3", "depth", depth, rep),
    ).fit(Xtr, ytr)
    tree_proba = align_proba(tree.predict_proba(Xte), tree.classes_)
    bag = BaggingEnsemble(
        {"max_depth": depth, "min_samples_leaf": min_samples_leaf},
        bag_B,
        seed=derive_seed(SEED, "q3", "depthbag", depth, rep),
        n_jobs=1,
    ).fit(Xtr, ytr)
    bag_preds = bag.predict_matrix(Xte, bag_B)
    _, bag_proba, _ = vote_from_predictions(bag_preds)
    stats = leaf_stats(tree)
    return {
        "rep": rep,
        "tree_proba": tree_proba,
        "bag_proba": bag_proba,
        "tree_acc": accuracy(yte, np.argmax(tree_proba, axis=1)),
        "bag_acc": accuracy(yte, np.argmax(bag_proba, axis=1)),
        "bag_disagreement": mean_pairwise_disagreement(bag_preds),
        "leaves": stats["leaves"],
        "mean_samples_per_leaf": stats["mean_samples_per_leaf"],
        "singleton_leaf_fraction": stats["singleton_leaf_fraction"],
    }


def q3b_depth_study(
    cfg: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    selected_depth: int | None,
    min_samples_leaf: int,
    resamples: np.ndarray,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_repeats = resamples.shape[0]
    bag_B = int(cfg["bias_variance"]["n_bag_trees_depth_study"])

    if selected_depth is None:
        depths: list[int | None] = [2, 16, None]
        labels = ["shallow (2)", "deep finite (16)", "selected / unrestricted (None)"]
    else:
        selected_depth = int(selected_depth)
        if selected_depth <= 2:
            shallow, deep = 1, 16
        elif selected_depth < 16:
            shallow, deep = 2, 32
        else:
            shallow, deep = 2, None
        depths = [shallow, selected_depth, deep]
        labels = [f"shallow ({shallow})", f"selected ({selected_depth})", f"deep ({deep})"]
    # Guard against duplicate depths in unusual configurations.
    unique: list[int | None] = []
    unique_labels: list[str] = []
    for d, lab in zip(depths, labels):
        if d not in unique:
            unique.append(d)
            unique_labels.append(lab)
    depths, labels = unique, unique_labels

    log(f"Q3b depth study at {depths} with {n_repeats} refits and {bag_B}-tree bags")
    per_depth: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for depth, label in zip(depths, labels):
        t0 = time.perf_counter()
        results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=1)(
            delayed(_depth_worker)(
                r,
                resamples[r],
                depth,
                X,
                y,
                train_idx,
                test_idx,
                int(min_samples_leaf),
                bag_B,
            )
            for r in range(n_repeats)
        )
        tree_probs = np.stack([r["tree_proba"] for r in results], axis=0)
        bag_probs = np.stack([r["bag_proba"] for r in results], axis=0)
        yte = y[test_idx]
        tree_bias, tree_var = bias_variance_from_probabilities(tree_probs, yte)
        bag_bias, bag_var = bias_variance_from_probabilities(bag_probs, yte)
        tree_accs = [r["tree_acc"] for r in results]
        bag_accs = [r["bag_acc"] for r in results]
        row = {
            "depth_label": label,
            "max_depth": "None" if depth is None else int(depth),
            "single_bias_squared": tree_bias,
            "single_variance": tree_var,
            "single_accuracy_mean": float(np.mean(tree_accs)),
            "single_accuracy_std": float(np.std(tree_accs, ddof=1)),
            "bag_bias_squared": bag_bias,
            "bag_variance": bag_var,
            "bag_accuracy_mean": float(np.mean(bag_accs)),
            "bag_accuracy_std": float(np.std(bag_accs, ddof=1)),
            "variance_reduction_absolute": float(tree_var - bag_var),
            "variance_reduction_fraction": float((tree_var - bag_var) / tree_var) if tree_var > 0 else float("nan"),
            "accuracy_gain_bagging": float(np.mean(bag_accs) - np.mean(tree_accs)),
            "leaves_mean": float(np.mean([r["leaves"] for r in results])),
            "mean_samples_per_leaf": float(np.mean([r["mean_samples_per_leaf"] for r in results])),
            "singleton_leaf_fraction": float(np.mean([r["singleton_leaf_fraction"] for r in results])),
            "bag_pairwise_disagreement_test": float(np.mean([r["bag_disagreement"] for r in results])),
            "n_repeats": n_repeats,
            "bag_trees": bag_B,
            "seconds": float(time.perf_counter() - t0),
        }
        per_depth.append(row)
        depth_key = f"depth_{len(per_depth) - 1}"
        arrays[f"tree_proba_{depth_key}"] = tree_probs
        arrays[f"bag_proba_{depth_key}"] = bag_probs
        arrays[f"tree_acc_{depth_key}"] = np.asarray(tree_accs)
        arrays[f"bag_acc_{depth_key}"] = np.asarray(bag_accs)
        log(
            f"Q3b {label}: single bias^2={tree_bias:.6f} var={tree_var:.6f} "
            f"acc={np.mean(tree_accs):.4f}; bag bias^2={bag_bias:.6f} var={bag_var:.6f} "
            f"acc={np.mean(bag_accs):.4f}; var reduction={row['variance_reduction_fraction']:.3f}"
        )
    df = dataframe_to_csv(per_depth, RESULTS_DIR / "depth_bias_variance.csv")
    np.savez_compressed(RESULTS_DIR / "depth_study_predictions.npz", **arrays)

    improvements = [row["accuracy_gain_bagging"] for row in per_depth]
    best_idx = int(np.argmax(improvements))
    deepest_idx = len(per_depth) - 1 if depths[-1] is None else max(
        range(len(depths)), key=lambda j: (depths[j] is None, depths[j])
    )
    conclusion = {
        "depths": [("None" if d is None else int(d)) for d in depths],
        "labels": labels,
        "accuracy_gain_bagging": improvements,
        "largest_improvement_depth": per_depth[best_idx]["max_depth"],
        "deepest_depth": per_depth[deepest_idx]["max_depth"],
        "deepest_has_largest_improvement": bool(best_idx == deepest_idx),
        "variance_reduction_fraction": [row["variance_reduction_fraction"] for row in per_depth],
        "single_variance": [row["single_variance"] for row in per_depth],
        "bag_variance": [row["bag_variance"] for row in per_depth],
        "bag_pairwise_disagreement_test": [row["bag_pairwise_disagreement_test"] for row in per_depth],
        "leaves_mean": [row["leaves_mean"] for row in per_depth],
        "singleton_leaf_fraction": [row["singleton_leaf_fraction"] for row in per_depth],
    }
    json_dump(RESULTS_DIR / "q3_depth_conclusion.json", conclusion)
    return df, conclusion


# ---------------------------------------------------------------------------
# Final summary and manifest
# ---------------------------------------------------------------------------
def write_final_summary(
    cfg: dict[str, Any],
    dataset,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
    selected_tree: dict[str, Any],
    selected_bagging: dict[str, Any],
    selected_boosting: dict[str, Any],
    final_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    depth_conclusion: dict[str, Any],
    runtime_seconds: float,
    log_path: Path,
) -> dict[str, Any]:
    summary = {
        "student": {"name": "Kumar Gaurav", "roll_number": ROLL_NUMBER},
        "seed": SEED,
        "seed_rule": "all numeric characters of P26AI0006, in order",
        "dataset": dataset_checksum(dataset),
        "split": {
            "train": len(train_idx),
            "validation": len(validation_idx),
            "test": len(test_idx),
            "sha256": split_checksum(train_idx, validation_idx, test_idx),
            "per_identity": {"train": 6, "validation": 2, "test": 2},
        },
        "selected_tree": selected_tree,
        "selected_bagging": selected_bagging,
        "selected_boosting": selected_boosting,
        "final_comparison": final_df.to_dict(orient="records"),
        "bias_variance": bias_df.to_dict(orient="records"),
        "depth_study_conclusion": depth_conclusion,
        "runtime_seconds": float(runtime_seconds),
        "runtime_human": f"{runtime_seconds / 3600:.2f} h",
        "environment": environment_info(),
        "git_commit": git_commit(),
        "log_file": str(log_path.resolve().relative_to(ROOT)).replace("\\", "/")
        if log_path.resolve().is_relative_to(ROOT) else str(log_path),
        "config": cfg,
        "notes": [
            "All model selections use clean validation accuracy only.",
            "The masked test set is never used for training or selection.",
            "Raw flattened 4096-dimensional pixels only; no PCA or other dimensionality reduction.",
            "Bagging and SAMME are implemented in src/common.py.",
        ],
    }
    json_dump(RESULTS_DIR / "run_summary.json", summary)
    json_dump(RESULTS_DIR / "config_resolved.json", cfg)
    json_dump(RESULTS_DIR / "environment.json", environment_info())

    # Final Q3(d) table.
    mapping = {"Single tree": "Single tree", "Bagging": "Bagging", "Boosting": "Boosting"}
    final_rows = []
    for _, r in final_df.iterrows():
        b = bias_df[bias_df["model"] == mapping[r["model"]]].iloc[0]
        final_rows.append(
            {
                "model": r["model"],
                "clean_test_accuracy": float(r["clean_test_accuracy"]),
                "masked_test_accuracy": float(r["masked_test_accuracy"]),
                "clean_mean_confidence": float(r["clean_mean_confidence"]),
                "masked_mean_confidence": float(r["masked_mean_confidence"]),
                "bias_squared": float(b["bias_squared"]),
                "variance": float(b["variance"]),
                "refit_accuracy_mean": float(b["refit_accuracy_mean"]),
                "refit_accuracy_std": float(b["refit_accuracy_std"]),
            }
        )
    dataframe_to_csv(final_rows, RESULTS_DIR / "final_table.csv")

    # Keep the exact Python environment used for the run.
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, timeout=120
        )
        (RESULTS_DIR / "requirements_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - environment dependent
        log(f"Could not write pip freeze: {exc}")

    log("Wrote run_summary.json, final_table.csv, environment.json")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    global LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIGS_DIR / "config.json"))
    parser.add_argument("--log", default=str(LOGS_DIR / "run.log"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log_path = Path(args.log)
    LOG = setup_logging(log_path)
    t_start = time.perf_counter()
    log("=" * 78)
    log("Machine Learning Assignment 2 | Kumar Gaurav | P26AI0006")
    log(f"Seed (all digits of roll number): {SEED}")
    log(f"Config: {args.config}")
    log("=" * 78)

    dataset = load_dataset()
    log(f"Dataset: {dataset.data.shape}, identities={len(np.unique(dataset.target))}, "
        f"pixel range=[{dataset.images.min():.3f}, {dataset.images.max():.3f}]")

    split_path = RESULTS_DIR / "split.npz"
    if split_path.exists():
        train_idx, validation_idx, test_idx = load_split(split_path, dataset.target)
    else:
        train_idx, validation_idx, test_idx = make_split(dataset.target)
        save_split(split_path, train_idx, validation_idx, test_idx, dataset.target)
    validate_split(train_idx, validation_idx, test_idx, dataset.target)
    log(f"Split: train={len(train_idx)} validation={len(validation_idx)} test={len(test_idx)}; "
        f"sha256={split_checksum(train_idx, validation_idx, test_idx)[:16]}...")

    X, y = dataset.data, dataset.target.astype(int)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[validation_idx], y[validation_idx]
    Xte, yte = X[test_idx], y[test_idx]
    Xmask = mask_centered_block(Xte)
    if not np.allclose(Xmask.reshape(-1, 64, 64)[:, 24:40, 24:40], 0.0):
        raise RuntimeError("Masked block is not zero")
    outside = np.ones((64, 64), dtype=bool)
    outside[24:40, 24:40] = False
    if not np.array_equal(
        Xmask.reshape(-1, 64, 64)[:, outside],
        Xte.reshape(-1, 64, 64)[:, outside],
    ):
        raise RuntimeError("Masking changed pixels outside the centred block")
    log("Mask check passed: only rows/columns 24..39 of the test images are zeroed")

    n_jobs = n_jobs_from_config(cfg)
    log(f"Parallel tree jobs: {n_jobs}")

    # Q1
    tree_best, tree, tree_df = q1a_tree_grid(cfg, Xtr, ytr, Xva, yva, Xte, yte)
    tree_params = {
        "max_depth": None if tree_best["max_depth_value"] is None else int(tree_best["max_depth_value"]),
        "min_samples_leaf": int(tree_best["min_samples_leaf"]),
    }
    bag, selected_B, bag_df, bag_preds = q1bc_bagging(
        cfg, tree_params, Xtr, ytr, Xva, yva, Xte, yte, Xmask, n_jobs
    )
    q1d_cases(
        train_idx, test_idx, ytr, yte, tree, bag, selected_B, Xte, Xmask, bag_preds
    )

    # Q2
    q2a_toy()
    boost, boost_best, boost_df, boost_hist = q2b_boosting(
        cfg, Xtr, ytr, Xva, yva, Xte, yte, train_idx
    )
    boost_models = {}
    selected_depth = int(boost_best["depth"])
    # Refit the three required round counts at the selected depth.  These are
    # deterministic and are used only for the controlled Q2(c) comparison.
    for L in (25, 50, 100):
        boost_models[(selected_depth, L)] = SAMME(
            max_depth=selected_depth,
            n_estimators=L,
            seed=derive_seed(SEED, "samme", selected_depth, L),
        ).fit(Xtr, ytr)
    q2c_comparison(
        cfg, tree_params, Xtr, ytr, Xva, yva, Xte, yte, bag, bag_df,
        boost_models, boost_best, n_jobs,
    )
    final_df, mask_case = q2d_comparison(
        train_idx, test_idx, ytr, yte, tree, bag, boost, selected_B, Xte, Xmask, bag_preds
    )

    # Q3
    selected_boost_conclusion = {
        "depth": int(boost.max_depth),
        "rounds": int(boost.n_estimators),
        "fitted": int(boost.n_estimators_fitted_),
    }
    bias_df, resamples = q3a_bias_variance(
        cfg, X, y, train_idx, test_idx, tree_params, selected_B,
        selected_boost_conclusion, n_jobs
    )
    depth_df, depth_conclusion = q3b_depth_study(
        cfg, X, y, train_idx, test_idx,
        None if tree_params["max_depth"] is None else int(tree_params["max_depth"]),
        int(tree_params["min_samples_leaf"]), resamples, n_jobs,
    )

    runtime = time.perf_counter() - t_start
    selected_bagging = json_load(RESULTS_DIR / "selected_bagging.json")
    selected_boosting = json_load(RESULTS_DIR / "selected_boosting.json")
    summary = write_final_summary(
        cfg, dataset, train_idx, validation_idx, test_idx,
        json_load(RESULTS_DIR / "selected_tree.json"),
        selected_bagging, selected_boosting,
        final_df, bias_df, depth_conclusion, runtime, log_path,
    )

    # Manifest of result artifacts (figures are added by make_figures.py).
    result_files = [
        p for p in RESULTS_DIR.glob("*")
        if p.is_file() and p.name not in {"artifact_manifest.json"}
    ]
    artifact_manifest(result_files, RESULTS_DIR / "artifact_manifest_results.json")
    log("=" * 78)
    log(f"DONE in {runtime / 3600:.2f} h. Results in {RESULTS_DIR}")
    log(f"Final table:\n{final_df.to_string(index=False)}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
