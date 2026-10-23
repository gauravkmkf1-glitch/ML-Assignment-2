"""Generate every figure in the report from saved result files.

Run after ``python -m src.run_experiments``:

    python -m src.make_figures

No model is refitted here, so the figures cannot silently change the
experimental results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .common import (
    FIGURES_DIR,
    N_CLASSES,
    RESULTS_DIR,
    SEED,
    load_dataset,
    json_load,
    mask_centered_block,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    }
)

BLUE = "#1f5fa8"
ORANGE = "#d9791e"
GREEN = "#2e8b57"
RED = "#b83a3a"
GRAY = "#666666"
LIGHT = "#d9d9d9"


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf and .png")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name, keep_default_na=False)


def depth_labels(values: list) -> list[str]:
    return ["None" if str(v) in {"None", "nan", ""} else str(v) for v in values]


# ---------------------------------------------------------------------------
# Figure 1: tree grid
# ---------------------------------------------------------------------------
def fig_tree_grid() -> None:
    df = read_csv("tree_grid.csv")
    selected = json_load(RESULTS_DIR / "selected_tree.json")
    sel_depth = selected["params"]["max_depth"]
    depths = ["2", "4", "8", "16", "None"]
    x = np.arange(len(depths))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    ax = axes[0]
    for leaf, color, marker in [(1, BLUE, "o"), (2, ORANGE, "s"), (4, GREEN, "^")]:
        sub = df[df["min_samples_leaf"].astype(int) == leaf].copy()
        sub["d"] = sub["max_depth"].astype(str)
        vals = [float(sub[sub["d"] == d]["validation_accuracy"].iloc[0]) for d in depths]
        ax.plot(x, vals, marker=marker, ms=4, lw=1.2, color=color, label=f"min leaf={leaf}")
    ax.set_xticks(x, depths)
    ax.set_xlabel("maximum depth (None = unrestricted)")
    ax.set_ylabel("validation accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) validation accuracy")

    ax = axes[1]
    sub = df[df["min_samples_leaf"].astype(int) == 1].copy()
    sub["d"] = sub["max_depth"].astype(str)
    for col, color, marker, label in [
        ("train_accuracy", GRAY, "o", "train"),
        ("validation_accuracy", BLUE, "s", "validation"),
        ("test_accuracy", ORANGE, "^", "test"),
    ]:
        vals = [float(sub[sub["d"] == d][col].iloc[0]) for d in depths]
        ax.plot(x, vals, marker=marker, ms=4, lw=1.2, color=color, label=label)
    sel_x = depths.index("None" if sel_depth is None else str(sel_depth))
    ax.axvline(sel_x, color=RED, ls="--", lw=0.9, alpha=0.8)
    ax.set_xticks(x, depths)
    ax.set_xlabel("maximum depth (None = unrestricted)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="center left")
    ax.set_title("(b) selected leaf size = 1")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig1_tree_grid")


# ---------------------------------------------------------------------------
# Figure 2: bagging diagnostics
# ---------------------------------------------------------------------------
def fig_bagging() -> None:
    df = read_csv("bagging_grid.csv")
    selected = json_load(RESULTS_DIR / "selected_bagging.json")
    sel_B = int(selected["selected_B"])
    B = df["B"].astype(int).values

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7))
    ax = axes[0, 0]
    for col, color, marker, label in [
        ("train_accuracy", GRAY, "o", "train"),
        ("validation_accuracy", BLUE, "s", "validation"),
        ("test_accuracy", ORANGE, "^", "clean test"),
        ("masked_test_accuracy", RED, "D", "masked test"),
    ]:
        ax.plot(B, df[col].astype(float), marker=marker, ms=3.5, lw=1.1, color=color, label=label)
    ax.axvline(sel_B, color=GREEN, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(B, [str(b) for b in B])
    ax.set_xlabel("B (bootstrap trees)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("(a) accuracy vs ensemble size")

    ax = axes[0, 1]
    ax.plot(B, df["oob_accuracy"].astype(float), marker="o", ms=4, lw=1.2, color=BLUE)
    ax.axvline(sel_B, color=GREEN, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(B, [str(b) for b in B])
    ax.set_xlabel("B")
    ax.set_ylabel("OOB accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("(b) out-of-bag accuracy")
    ax2 = ax.twinx()
    ax2.plot(B, df["oob_coverage"].astype(float), marker="s", ms=3, lw=0.9, color=GRAY, alpha=0.8)
    ax2.set_ylabel("OOB coverage", color=GRAY)
    ax2.tick_params(axis="y", colors=GRAY)
    ax2.set_ylim(0, 1.05)
    ax2.spines["right"].set_visible(True)

    ax = axes[1, 0]
    ax.plot(B, df["pairwise_disagreement_validation"].astype(float), marker="o", ms=4, lw=1.2, color=BLUE, label="validation")
    ax.plot(B, df["pairwise_disagreement_test"].astype(float), marker="s", ms=3, lw=0.9, color=ORANGE, alpha=0.8, label="test")
    ax.axvline(sel_B, color=GREEN, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(B, [str(b) for b in B])
    ax.set_xlabel("B")
    ax.set_ylabel("mean pairwise disagreement")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("(c) tree disagreement")

    ax = axes[1, 1]
    ax.plot(B, df["mean_vote_margin_validation"].astype(float), marker="o", ms=4, lw=1.2, color=BLUE, label="validation")
    ax.plot(B, df["mean_vote_margin_test"].astype(float), marker="s", ms=3, lw=0.9, color=ORANGE, alpha=0.8, label="test")
    ax.axvline(sel_B, color=GREEN, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(B, [str(b) for b in B])
    ax.set_xlabel("B")
    ax.set_ylabel("mean vote margin")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("(d) winning minus runner-up")
    fig.tight_layout(w_pad=1.2, h_pad=1.2)
    save(fig, "fig2_bagging")


# ---------------------------------------------------------------------------
# Figure 3: Q1 face cases
# ---------------------------------------------------------------------------
def fig_q1_cases() -> None:
    dataset = load_dataset()
    cases = json_load(RESULTS_DIR / "q1_cases.json")
    items = cases["corrections"] + cases["both_wrong"]
    if len(items) < 4:
        raise RuntimeError("Need four Q1 case studies")
    fig = plt.figure(figsize=(7.0, 4.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.25, 1.0, 1.25], hspace=0.55, wspace=0.35)
    for idx, item in enumerate(items[:4]):
        r, c = divmod(idx, 2)
        ax_img = fig.add_subplot(gs[r, 2 * c])
        img = dataset.images[item["dataset_index"]]
        ax_img.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        prefix = "(a)" if idx == 0 else ("(b)" if idx == 1 else ("(c)" if idx == 2 else "(d)"))
        ax_img.set_title(
            f"{prefix} true {item['true_identity']}\n"
            f"tree {item['tree_prediction']}, bag {item['bag_prediction']}",
            fontsize=7.5,
            pad=2,
        )
        ax_bar = fig.add_subplot(gs[r, 2 * c + 1])
        counts = np.asarray(item["bag_vote_counts"], dtype=int)
        top = np.argsort(counts)[::-1][:5]
        vals = counts[top]
        labels = [str(int(c)) for c in top]
        colors = [GREEN if int(c) == int(item["true_identity"]) else BLUE for c in top]
        ax_bar.bar(np.arange(len(top)), vals, color=colors, width=0.72)
        ax_bar.set_xticks(np.arange(len(top)), labels)
        ax_bar.set_xlabel("identity (top bag votes)", fontsize=7)
        ax_bar.set_ylabel("votes", fontsize=7)
        ax_bar.set_ylim(0, max(1, counts.max() * 1.18))
        ax_bar.tick_params(labelsize=6.5)
        ax_bar.grid(axis="y", alpha=0.25, lw=0.4)
        ax_bar.set_title(
            "green = true class" if idx < 2 else "confident shared error",
            fontsize=6.8,
            pad=1,
        )
    fig.suptitle(
        "Q1(d) cases: bagging corrections (top) and errors shared by both models (bottom)",
        fontsize=8.8,
        y=1.00,
    )
    save(fig, "fig3_q1_cases")


# ---------------------------------------------------------------------------
# Figure 4: selected SAMME diagnostics
# ---------------------------------------------------------------------------
def fig_samme_history() -> None:
    hist = read_csv("boosting_history.csv")
    selected = json_load(RESULTS_DIR / "selected_boosting.json")
    depth = selected["selected"]["max_depth"]
    rounds = selected["selected"]["n_estimators_requested"]
    r = hist["round"].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    panels = [
        ("weighted_error", "(a) weighted training error", BLUE),
        ("alpha", "(b) learner weight $\\alpha_t$", ORANGE),
        ("max_weight", "(c) maximum sample weight", GREEN),
        ("top10_weight", "(d) total weight of 10 heaviest samples", RED),
    ]
    for ax, (col, title, color) in zip(axes.ravel(), panels):
        ax.plot(r, hist[col].astype(float), lw=1.2, color=color)
        ax.set_xlabel("boosting round")
        ax.set_ylabel(col.replace("_", " "))
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(1, max(2, int(r.max())))
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    fig.suptitle(f"SAMME diagnostics, depth {depth}, {rounds} requested rounds", fontsize=8.8, y=1.00)
    fig.tight_layout(h_pad=1.1, w_pad=1.2)
    save(fig, "fig4_samme_history")


# ---------------------------------------------------------------------------
# Figure 5: heavily weighted training face
# ---------------------------------------------------------------------------
def fig_weight_case() -> None:
    dataset = load_dataset()
    case = json_load(RESULTS_DIR / "q2_weight_case.json")
    hist = pd.DataFrame(case["history"])
    fig = plt.figure(figsize=(7.0, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.55, 1.55], wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(dataset.images[case["dataset_index"]], cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"(a) identity {case['true_identity']}\nfinal weight {case['final_weight']:.3f}",
        fontsize=7.5,
    )

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(hist["round"], hist["weight_before"], color=BLUE, lw=1.2)
    ax.scatter(
        hist.loc[~hist["weak_correct"], "round"],
        hist.loc[~hist["weak_correct"], "weight_before"],
        s=10,
        color=RED,
        zorder=3,
        label="weak learner wrong",
    )
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("sample weight before round")
    ax.set_title("(b) weight trajectory")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=6.5)

    ax = fig.add_subplot(gs[0, 2])
    jitter = (np.arange(len(hist)) % 5 - 2) * 0.08
    colors = np.where(hist["weak_correct"], GREEN, RED)
    ax.scatter(hist["round"], hist["weak_prediction"] + jitter, s=8, c=colors, alpha=0.85)
    ax.axhline(case["true_identity"], color="black", lw=0.8, ls="--")
    ax.set_xlabel("round")
    ax.set_ylabel("weak prediction")
    ax.set_title("(c) prediction history")
    ax.set_ylim(-1, 40)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.text(
        0.02,
        0.98,
        f"misclassified {case['n_misclassified']}/{case['n_rounds']} rounds\n"
        f"green = correct, red = wrong",
        transform=ax.transAxes,
        va="top",
        fontsize=6.5,
    )
    fig.suptitle("Q2(c) training face receiving the largest final SAMME weight", fontsize=8.8, y=1.02)
    save(fig, "fig5_weight_case")


# ---------------------------------------------------------------------------
# Figure 6: masked-response case
# ---------------------------------------------------------------------------
def fig_mask_case() -> None:
    dataset = load_dataset()
    case = json_load(RESULTS_DIR / "q2_mask_case.json")
    img = dataset.images[case["dataset_index"]]
    masked = mask_centered_block(img.reshape(1, -1)).reshape(64, 64)

    fig = plt.figure(figsize=(7.0, 3.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], wspace=0.35, hspace=0.55)

    for j, (title, arr) in enumerate([("(a) clean", img), ("(b) masked", masked)]):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        if j == 1:
            ax.add_patch(Rectangle((23.5, 23.5), 16, 16, fill=False, edgecolor=RED, lw=1.0))
            ax.text(41, 30, "zeroed", color=RED, fontsize=6.5, va="center")

    ax = fig.add_subplot(gs[1, 0:2])
    counts = np.asarray(case["masked"]["bag_vote_counts"], dtype=int)
    top = np.argsort(counts)[::-1][:5]
    vals = counts[top]
    colors = [GREEN if int(c) == int(case["true_identity"]) else BLUE for c in top]
    ax.bar(np.arange(len(top)) - 0.18, vals, width=0.36, color=colors, label="bag votes")
    boost = np.asarray(case["masked"]["boost_scores"], dtype=float)
    ax.bar(np.arange(len(top)) + 0.18, boost[top] * 50.0, width=0.36, color=ORANGE, alpha=0.8, label="boost q (scaled x50)")
    ax.set_xticks(np.arange(len(top)), [str(int(c)) for c in top])
    ax.set_xlabel("identity (top bag votes)")
    ax.set_ylabel("bag votes / scaled score")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_title(
        f"masked: bag predicts {case['masked']['bag_prediction']}, "
        f"boost predicts {case['masked']['boost_prediction']} (true {case['true_identity']})",
        fontsize=8,
    )

    ax = fig.add_subplot(gs[1, 2])
    parts = [
        ("bag disag.", case["masked"]["bag_disagreement"]),
        ("bag margin", case["masked"]["bag_margin"]),
        ("boost margin", case["masked"]["boost_margin"]),
        ("boost true q", case["masked"]["boost_true_score"]),
    ]
    ax.barh([p[0] for p in parts], [p[1] for p in parts], color=[BLUE, BLUE, ORANGE, ORANGE])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("value")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_title("(d) diagnostics", fontsize=8)
    ax.tick_params(labelsize=6.5)

    fig.suptitle("Q2(d) face where bagging and boosting respond differently to the centred mask", fontsize=8.8, y=1.01)
    save(fig, "fig6_mask_case")


# ---------------------------------------------------------------------------
# Figure 7: bias-variance and accuracy summary
# ---------------------------------------------------------------------------
def fig_bias_variance() -> None:
    bv = read_csv("bias_variance.csv")
    final = read_csv("final_comparison.csv")
    depth = read_csv("depth_bias_variance.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    ax = axes[0, 0]
    x = np.arange(len(bv))
    ax.bar(x - 0.18, bv["bias_squared"].astype(float), width=0.36, color=BLUE, label="bias$^2$")
    ax.bar(x + 0.18, bv["variance"].astype(float), width=0.36, color=ORANGE, label="variance")
    ax.set_xticks(x, bv["model"])
    ax.set_ylabel("measured quantity")
    ax.set_ylim(0, max(1.0, float(bv[["bias_squared", "variance"]].astype(float).to_numpy().max()) * 1.18))
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("(a) selected models: bias and variance")

    ax = axes[0, 1]
    x = np.arange(len(final))
    ax.bar(x - 0.18, final["clean_test_accuracy"].astype(float), width=0.36, color=BLUE, label="clean")
    ax.bar(x + 0.18, final["masked_test_accuracy"].astype(float), width=0.36, color=RED, label="masked")
    ax.set_xticks(x, final["model"])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("(b) clean vs masked test accuracy")

    ax = axes[1, 0]
    labels = depth["depth_label"].tolist()
    x = np.arange(len(depth))
    ax.plot(x, depth["single_variance"].astype(float), marker="o", ms=4, color=ORANGE, label="single tree")
    ax.plot(x, depth["bag_variance"].astype(float), marker="s", ms=4, color=BLUE, label="100-tree bag")
    for i, row in depth.iterrows():
        ax.annotate(
            f"{100 * float(row['variance_reduction_fraction']):.0f}%",
            (i, max(float(row["single_variance"]), float(row["bag_variance"]))),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=6.5,
            color=GREEN,
        )
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("variance")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("(c) depth study: variance (labels = reduction)")

    ax = axes[1, 1]
    ax.plot(x, depth["single_accuracy_mean"].astype(float), marker="o", ms=4, color=ORANGE, label="single tree")
    ax.plot(x, depth["bag_accuracy_mean"].astype(float), marker="s", ms=4, color=BLUE, label="100-tree bag")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("mean test accuracy over 30 refits")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("(d) depth study: accuracy")
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, "fig7_bias_variance")


def main() -> None:
    fig_tree_grid()
    fig_bagging()
    fig_q1_cases()
    fig_samme_history()
    fig_weight_case()
    fig_mask_case()
    fig_bias_variance()
    print("All figures generated from saved results.")


if __name__ == "__main__":
    main()
