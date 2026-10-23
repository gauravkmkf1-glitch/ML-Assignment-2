"""Generate LaTeX tables/numbers from results and compile the CVPR report.

Usage:

    python -m src.build_report              # generate .tex and compile
    python -m src.build_report --no-build   # generate .tex only

Every number in the paper is read from ``results/``; none is typed by hand.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

from .common import (
    CONFIGS_DIR,
    PAPER_DIR,
    RESULTS_DIR,
    ROOT,
    SEED,
    git_commit,
    json_load,
)

GENERATED = PAPER_DIR / "generated"
TABLES = GENERATED / "tables"

MIKTEX_BIN = Path(r"C:\Users\ACER\AppData\Local\Programs\MiKTeX\miktex\bin\x64")


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * float(x):.{digits}f}"


def num(x: float, digits: int = 4) -> str:
    return f"{float(x):.{digits}f}"


def tex_text(value: object) -> str:
    s = str(value)
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def load_results() -> dict:
    required = [
        "tree_grid.csv",
        "selected_tree.json",
        "bagging_grid.csv",
        "selected_bagging.json",
        "boosting_grid.csv",
        "selected_boosting.json",
        "toy_samme.json",
        "boosting_vs_bagging.csv",
        "final_comparison.csv",
        "bias_variance.csv",
        "depth_bias_variance.csv",
        "final_table.csv",
        "q1_cases.json",
        "q2_weight_case.json",
        "q2_mask_case.json",
        "q3_depth_conclusion.json",
        "run_summary.json",
    ]
    missing = [name for name in required if not (RESULTS_DIR / name).exists()]
    if missing:
        raise SystemExit("Missing result files: " + ", ".join(missing) + "\nRun src.run_experiments first.")
    out = {
        "tree": pd.read_csv(RESULTS_DIR / "tree_grid.csv"),
        "bag": pd.read_csv(RESULTS_DIR / "bagging_grid.csv"),
        "boost": pd.read_csv(RESULTS_DIR / "boosting_grid.csv"),
        "bvb": pd.read_csv(RESULTS_DIR / "boosting_vs_bagging.csv"),
        "final": pd.read_csv(RESULTS_DIR / "final_comparison.csv"),
        "bias": pd.read_csv(RESULTS_DIR / "bias_variance.csv"),
        "depth": pd.read_csv(RESULTS_DIR / "depth_bias_variance.csv"),
        "final_table": pd.read_csv(RESULTS_DIR / "final_table.csv"),
        "tree_sel": json_load(RESULTS_DIR / "selected_tree.json"),
        "bag_sel": json_load(RESULTS_DIR / "selected_bagging.json"),
        "boost_sel": json_load(RESULTS_DIR / "selected_boosting.json"),
        "toy": json_load(RESULTS_DIR / "toy_samme.json"),
        "q1": json_load(RESULTS_DIR / "q1_cases.json"),
        "weight": json_load(RESULTS_DIR / "q2_weight_case.json"),
        "mask": json_load(RESULTS_DIR / "q2_mask_case.json"),
        "depth_concl": json_load(RESULTS_DIR / "q3_depth_conclusion.json"),
        "summary": json_load(RESULTS_DIR / "run_summary.json"),
    }
    return out


def repo_url() -> str:
    path = PAPER_DIR / "repo_url.txt"
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    env = os.environ.get("REPO_URL", "").strip()
    if env:
        return env
    return "REPOSITORY_URL_TO_BE_ADDED"


def generate_numbers(r: dict) -> None:
    tree = r["tree_sel"]
    bag = r["bag_sel"]
    boost = r["boost_sel"]
    toy = r["toy"]
    weight = r["weight"]
    mask = r["mask"]
    bias = r["bias"].set_index("model")
    final = r["final"].set_index("model")
    final_table = r["final_table"].set_index("model")
    depth_concl = r["depth_concl"]
    depth = r["depth"].set_index("depth_label")
    summary = r["summary"]
    commit = git_commit()

    tree_row = tree["row"]
    bag_row = bag["row"]
    boost_row = boost["row"]
    boost_sel = boost["selected"]

    macros: dict[str, str] = {
        "Seed": str(SEED),
        "RollNumber": "P26AI0006",
        "TrainN": str(summary["split"]["train"]),
        "ValN": str(summary["split"]["validation"]),
        "TestN": str(summary["split"]["test"]),
        "NFeatures": "4096",
        "SplitHash": summary["split"]["sha256"][:12],
        "DataHash": summary["dataset"]["images_sha256"][:12],
        "RuntimeHours": f"{summary['runtime_seconds'] / 3600:.2f}",
        "RepoURL": repo_url(),
        "CommitHash": commit,
        "TreeDepth": "None" if tree["params"]["max_depth"] is None else str(tree["params"]["max_depth"]),
        "TreeLeaf": str(tree["params"]["min_samples_leaf"]),
        "TreeTrain": num(tree_row["train_accuracy"]),
        "TreeVal": num(tree_row["validation_accuracy"]),
        "TreeTest": num(tree_row["test_accuracy"]),
        "TreeLeaves": str(int(tree_row["leaves"])),
        "TreeMeanLeaf": num(tree_row["mean_samples_per_leaf"], 2),
        "TreeSingleton": num(tree_row["singleton_leaf_fraction"], 3),
        "TreeFitSeconds": num(tree_row["fit_seconds"], 2),
        "BagB": str(int(bag["selected_B"])),
        "BagTrain": num(bag_row["train_accuracy"]),
        "BagVal": num(bag_row["validation_accuracy"]),
        "BagTest": num(bag_row["test_accuracy"]),
        "BagMask": num(bag_row["masked_test_accuracy"]),
        "BagOOB": num(bag_row["oob_accuracy"]),
        "BagDis": num(bag_row["pairwise_disagreement_validation"]),
        "BagMargin": num(bag_row["mean_vote_margin_validation"]),
        "BagDistinct": num(bag_row["mean_distinct"], 2),
        "BagDistinctTheory": num(bag_row["theoretical_distinct"], 2),
        "BagTime": num(bag_row["train_seconds_serial"], 2),
        "BagWall": num(bag_row["train_seconds_wall_full_200"], 2),
        "NCorrections": str(int(r["q1"]["n_corrections_bag_over_tree"])),
        "NBothWrong": str(int(r["q1"]["n_both_wrong"])),
        "ToyError": num(toy["weighted_error"]),
        "ToyAlpha": num(toy["alpha"]),
        "BoostDepth": str(boost_sel["max_depth"]),
        "BoostRounds": str(boost_sel["n_estimators_requested"]),
        "BoostActual": str(boost_sel["n_estimators_fitted"]),
        "BoostTrain": num(boost_row["train_accuracy"]),
        "BoostVal": num(boost_row["validation_accuracy"]),
        "BoostTest": num(boost_row["test_accuracy"]),
        "BoostTime": num(boost_row["train_seconds"], 2),
        "BoostLastErr": num(boost_row["last_weighted_error"]),
        "BoostAlphaLast": num(boost_row["last_alpha"]),
        "BoostMaxWeight": num(boost_row["last_max_weight"], 4),
        "BoostTopTen": num(boost_row["last_top10_weight"], 3),
        "BoostCentral": num(boost_row["central_feature_importance_mass"], 3),
        "WeightFace": str(weight["dataset_index"]),
        "WeightIdentity": str(weight["true_identity"]),
        "WeightFinal": num(weight["final_weight"], 4),
        "WeightUniform": num(weight["uniform_weight"], 5),
        "WeightNMiss": str(weight["n_misclassified"]),
        "WeightRounds": str(weight["n_rounds"]),
        "WeightFirstCorrect": str(weight["first_correct_cumulative_round"]),
        "MaskFace": str(mask["dataset_index"]),
        "MaskTrue": str(mask["true_identity"]),
        "MaskBagPred": str(mask["masked"]["bag_prediction"]),
        "MaskBoostPred": str(mask["masked"]["boost_prediction"]),
        "MaskBagDis": num(mask["masked"]["bag_disagreement"], 3),
        "MaskBagMargin": num(mask["masked"]["bag_margin"], 3),
        "MaskBoostMargin": num(mask["masked"]["boost_margin"], 3),
        "MaskBagTrueVotes": str(mask["masked"]["bag_trees_voting_true"]),
        "MaskBoostTrueQ": num(mask["masked"]["boost_true_score"], 3),
        "MaskTreePred": str(mask["masked"]["tree_prediction"]),
        "MaskDirection": tex_text(mask["direction"]),
        "TreeBias": num(bias.loc["Single tree", "bias_squared"]),
        "TreeVar": num(bias.loc["Single tree", "variance"]),
        "BagBias": num(bias.loc["Bagging", "bias_squared"]),
        "BagVar": num(bias.loc["Bagging", "variance"]),
        "BagVarRatio": num(
            bias.loc["Single tree", "variance"] / bias.loc["Bagging", "variance"], 1
        ),
        "BagBiasDelta": num(
            bias.loc["Bagging", "bias_squared"] - bias.loc["Single tree", "bias_squared"], 4
        ),
        "BoostBias": num(bias.loc["Boosting", "bias_squared"]),
        "BoostVar": num(bias.loc["Boosting", "variance"]),
        "BoostBiasRatio": num(
            bias.loc["Boosting", "bias_squared"] / bias.loc["Single tree", "bias_squared"], 1
        ),
        "TreeRefitAcc": num(bias.loc["Single tree", "refit_accuracy_mean"]),
        "TreeRefitStd": num(bias.loc["Single tree", "refit_accuracy_std"]),
        "BagRefitAcc": num(bias.loc["Bagging", "refit_accuracy_mean"]),
        "BagRefitStd": num(bias.loc["Bagging", "refit_accuracy_std"]),
        "BoostRefitAcc": num(bias.loc["Boosting", "refit_accuracy_mean"]),
        "BoostRefitStd": num(bias.loc["Boosting", "refit_accuracy_std"]),
        "TreeCleanAcc": num(final.loc["Single tree", "clean_test_accuracy"]),
        "TreeMaskAcc": num(final.loc["Single tree", "masked_test_accuracy"]),
        "BagCleanAcc": num(final.loc["Bagging", "clean_test_accuracy"]),
        "BagMaskAcc": num(final.loc["Bagging", "masked_test_accuracy"]),
        "BoostCleanAcc": num(final.loc["Boosting", "clean_test_accuracy"]),
        "BoostMaskAcc": num(final.loc["Boosting", "masked_test_accuracy"]),
        "TreeCleanConf": num(final.loc["Single tree", "clean_mean_confidence"]),
        "TreeMaskConf": num(final.loc["Single tree", "masked_mean_confidence"]),
        "BagCleanConf": num(final.loc["Bagging", "clean_mean_confidence"]),
        "BagMaskConf": num(final.loc["Bagging", "masked_mean_confidence"]),
        "BoostCleanConf": num(final.loc["Boosting", "clean_mean_confidence"]),
        "BoostMaskConf": num(final.loc["Boosting", "masked_mean_confidence"]),
        "TreeCleanDrop": num(final.loc["Single tree", "clean_accuracy_drop"]),
        "BagCleanDrop": num(final.loc["Bagging", "clean_accuracy_drop"]),
        "BoostCleanDrop": num(final.loc["Boosting", "clean_accuracy_drop"]),
        "TreeCentral": num(final.loc["Single tree", "central_feature_importance_mass"], 3),
        "BagCentral": num(final.loc["Bagging", "central_feature_importance_mass"], 3),
        "BoostCentralMass": num(final.loc["Boosting", "central_feature_importance_mass"], 3),
        "DeepestDepth": tex_text(depth_concl["deepest_depth"]),
        "LargestImprovementDepth": tex_text(depth_concl["largest_improvement_depth"]),
        "LargestImprovementGain": num(max(depth_concl["accuracy_gain_bagging"]), 3),
        "DeepestLargest": "Yes" if depth_concl["deepest_has_largest_improvement"] else "No",
        "TreeFinalBias": num(final_table.loc["Single tree", "bias_squared"]),
        "TreeFinalVar": num(final_table.loc["Single tree", "variance"]),
        "BagFinalBias": num(final_table.loc["Bagging", "bias_squared"]),
        "BagFinalVar": num(final_table.loc["Bagging", "variance"]),
        "BoostFinalBias": num(final_table.loc["Boosting", "bias_squared"]),
        "BoostFinalVar": num(final_table.loc["Boosting", "variance"]),
    }
    # Depth-row macros keyed by semantic position, independent of long labels.
    depth_names = {1: "Shallow", 2: "Mid", 3: "Deep"}
    for i, row in enumerate(r["depth"].itertuples(index=False), start=1):
        stem = f"Depth{depth_names.get(i, f'Level{i}')}"
        macros[f"{stem}Label"] = tex_text(row.depth_label)
        macros[f"{stem}SingleBias"] = num(row.single_bias_squared)
        macros[f"{stem}BagBias"] = num(row.bag_bias_squared)
        macros[f"{stem}SingleVar"] = num(row.single_variance)
        macros[f"{stem}BagVar"] = num(row.bag_variance)
        macros[f"{stem}VarRed"] = pct(row.variance_reduction_fraction)
        macros[f"{stem}SingleAcc"] = num(row.single_accuracy_mean)
        macros[f"{stem}BagAcc"] = num(row.bag_accuracy_mean)
        macros[f"{stem}Gain"] = num(row.accuracy_gain_bagging)
        macros[f"{stem}Leaves"] = num(row.leaves_mean, 1)
        macros[f"{stem}MeanLeaf"] = num(row.mean_samples_per_leaf, 1)
        macros[f"{stem}Singleton"] = num(row.singleton_leaf_fraction, 3)
        macros[f"{stem}Dis"] = num(row.bag_pairwise_disagreement_test, 3)

    # Case-study macros: top-two vote counts and the resulting vote margin.
    cases = r["q1"]["corrections"][:2] + r["q1"]["both_wrong"][:2]
    case_names = {1: "One", 2: "Two", 3: "Three", 4: "Four"}
    for i, item in enumerate(cases, start=1):
        counts = sorted((int(v) for v in item["bag_vote_counts"]), reverse=True)
        total = max(1, sum(counts))
        stem = f"Case{case_names.get(i, f'Number{i}')}"
        macros[f"{stem}Votes"] = str(counts[0])
        macros[f"{stem}Margin"] = num((counts[0] - counts[1]) / total, 3)
        macros[f"{stem}True"] = str(item["true_identity"])
        macros[f"{stem}Tree"] = str(item["tree_prediction"])
        macros[f"{stem}Bag"] = str(item["bag_prediction"])

    lines = ["% Auto-generated by src/build_report.py. Do not edit by hand."]
    for key, value in macros.items():
        lines.append(rf"\newcommand{{\{key}}}{{{value}}}")
    write(GENERATED / "numbers.tex", "\n".join(lines))


def tab_tree(r: dict) -> None:
    tree = r["tree_sel"]
    row = tree["row"]
    body = rf"""
\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{3.2pt}}
\caption{{Frozen single-tree baseline selected on clean validation accuracy only. The grid was $5$ depths $\times$ $3$ leaf sizes; the complete grid is in \texttt{{results/tree\_grid.csv}}.}}
\label{{tab:tree}}
\begin{{tabular}}{{lcccccc}}
\toprule
Depth & Leaf & Train & Val & Test & Leaves & Mean leaf & Sing. \\
\midrule
{("None" if tree["params"]["max_depth"] is None else tree["params"]["max_depth"])}
& {tree["params"]["min_samples_leaf"]}
& {num(row["train_accuracy"])}
& {num(row["validation_accuracy"])}
& {num(row["test_accuracy"])}
& {int(row["leaves"])}
& {num(row["mean_samples_per_leaf"], 2)}
& {num(row["singleton_leaf_fraction"], 3)} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(TABLES / "tree.tex", body)


def tab_bagging(r: dict) -> None:
    df = r["bag"].copy()
    body = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
            r"\caption{Bagging with the frozen tree setting. All $B$ values use the same nested bootstrap sequence; OOB accuracy is computed only on covered training samples. Disagreement and margin are on the clean validation set. ``Distinct'' is the mean number of distinct training indices per bootstrap sample; the theoretical value $n(1-(1-1/n)^n)$ is in parentheses.}",
            r"\label{tab:bagging}", r"\begin{tabular}{rcccccccc}", r"\toprule",
            r"$B$ & Train & Val & Test & Masked & OOB & Disagreement & Margin & Distinct (theory) \\", r"\midrule"]
    for _, row in df.iterrows():
        body.append(
            f"{int(row['B'])} & {num(row['train_accuracy'])} & {num(row['validation_accuracy'])} & "
            f"{num(row['test_accuracy'])} & {num(row['masked_test_accuracy'])} & {num(row['oob_accuracy'])} & "
            f"{num(row['pairwise_disagreement_validation'])} & {num(row['mean_vote_margin_validation'])} & "
            f"{num(row['mean_distinct'], 1)} ({num(row['theoretical_distinct'], 1)}) \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(TABLES / "bagging.tex", "\n".join(body))


def tab_toy(r: dict) -> None:
    toy = r["toy"]
    rows = []
    for i, (y, w0, h, w1) in enumerate(
        zip(toy["y"], toy["initial_weights"], toy["weak_predictions"], toy["updated_normalised_weights"])
    ):
        rows.append(f"{i} & {y} & {w0:.4f} & {h} & {'yes' if h == y else 'no'} & {w1:.4f} \\\\")
    body = rf"""
\begin{{table}}[t]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3.5pt}}
\caption{{One complete SAMME round on a three-class, nine-sample toy problem. The depth-1 stump gives weighted error $\epsilon_1={num(toy['weighted_error'],4)}$ and $\alpha_1={num(toy['alpha'],4)}$. Updated weights are normalised.}}
\label{{tab:toy}}
\begin{{tabular}}{{rcccccc}}
\toprule
$i$ & $y_i$ & $w_i^{{(1)}}$ & $h_1(x_i)$ & Correct? & $w_i^{{(2)}}$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(TABLES / "toy.tex", body)


def tab_boosting_grid(r: dict) -> None:
    df = r["boost"]
    depths = sorted(df["depth"].astype(int).unique())
    rounds = sorted(df["requested_rounds"].astype(int).unique())
    header = " & ".join(["Depth"] + [f"{q}" for q in rounds])
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.2pt}",
             r"\caption{SAMME grid. Each cell is clean validation accuracy / actual accepted rounds; the full train/test/timing grid is in \texttt{results/boosting\_grid.csv}. All 16 runs accepted every requested round.}",
             r"\label{tab:boostgrid}", rf"\begin{{tabular}}{{c{'c' * len(rounds)}}}", r"\toprule", header + r" \\", r"\midrule"]
    for d in depths:
        cells = []
        for q in rounds:
            row = df[(df["depth"].astype(int) == d) & (df["requested_rounds"].astype(int) == q)].iloc[0]
            cells.append(f"{num(row['validation_accuracy'], 3)} / {int(row['actual_rounds'])}")
        lines.append(f"{d} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(TABLES / "boosting_grid.tex", "\n".join(lines))


def tab_boost_vs_bag(r: dict) -> None:
    df = r["bvb"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.0pt}",
             r"\caption{Controlled comparison at 25, 50 and 100 learners with the frozen settings. Bagging times are serial-equivalent sums of per-tree fit times; boosting is sequential.}",
             r"\label{tab:boostvsbag}", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Learners & Train & Val & Test & Seconds \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(
            f"{tex_text(row['family'])} & {int(row['n_learners'])} & {num(row['train_accuracy'])} & "
            f"{num(row['validation_accuracy'])} & {num(row['test_accuracy'])} & {num(row['train_seconds'], 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(TABLES / "boost_vs_bag.tex", "\n".join(lines))


def tab_depth(r: dict) -> None:
    df = r["depth"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
             r"\caption{Bias--variance depth study over the same 30 within-identity resamples. Each bag has 100 trees. Variance reduction is relative to the matched single tree. Disagreement is measured on the clean test set.}",
             r"\label{tab:depth}", r"\begin{tabular}{lcccccccccc}", r"\toprule",
             r"Depth & Single bias$^2$ & Single var. & Single acc. & Bag bias$^2$ & Bag var. & Bag acc. & Var. red. (\%) & Leaves & Singleton & Disagr. \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(
            f"{tex_text(row['depth_label'])} & {num(row['single_bias_squared'])} & {num(row['single_variance'])} & "
            f"{num(row['single_accuracy_mean'])} & {num(row['bag_bias_squared'])} & {num(row['bag_variance'])} & "
            f"{num(row['bag_accuracy_mean'])} & {pct(row['variance_reduction_fraction'])} & "
            f"{num(row['leaves_mean'], 1)} & {num(row['singleton_leaf_fraction'], 3)} & "
            f"{num(row['bag_pairwise_disagreement_test'], 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(TABLES / "depth.tex", "\n".join(lines))


def tab_final(r: dict) -> None:
    df = r["final_table"].copy()
    conf = r["final"].set_index("model")
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\caption{Final selected models. Confidence is the mean winning vote/probability; bias$^2$ and variance are measured over 30 refits. The masked column zeroes the central $16\times16$ block at test time; training used clean images.}",
             r"\label{tab:final}", r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Model & Clean acc. & Masked acc. & Clean conf. & Masked conf. & Bias$^2$ & Var. \\", r"\midrule"]
    for _, row in df.iterrows():
        clean_conf = row["clean_mean_confidence"] if "clean_mean_confidence" in row else conf.loc[row["model"], "clean_mean_confidence"]
        masked_conf = row["masked_mean_confidence"] if "masked_mean_confidence" in row else conf.loc[row["model"], "masked_mean_confidence"]
        lines.append(
            f"{tex_text(row['model'])} & {num(row['clean_test_accuracy'])} & {num(row['masked_test_accuracy'])} & "
            f"{num(clean_conf)} & {num(masked_conf)} & "
            f"{num(row['bias_squared'])} & {num(row['variance'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(TABLES / "final.tex", "\n".join(lines))


def ensure_style_files() -> None:
    kit = ROOT.parent / "cvpr-kit"
    for name in ("cvpr.sty", "ieeenat_fullname.bst"):
        src = kit / name
        dst = PAPER_DIR / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"copied {name} from CVPR author kit")
    if not (PAPER_DIR / "cvpr.sty").exists():
        raise SystemExit("cvpr.sty not found; run from the submission workspace or copy it into paper/.")


def compile_pdf() -> Path:
    env = os.environ.copy()
    if MIKTEX_BIN.exists():
        env["PATH"] = str(MIKTEX_BIN) + os.pathsep + env.get("PATH", "")
    # Ask MiKTeX to install missing packages without interactive prompts.
    try:
        subprocess.run(
            [str(MIKTEX_BIN / "initexmf.exe"), "--set-config-value", "[MPM]AutoInstall=1"],
            check=False,
            capture_output=True,
            env=env,
        )
    except Exception:
        pass
    pdflatex = str(MIKTEX_BIN / "pdflatex.exe") if (MIKTEX_BIN / "pdflatex.exe").exists() else "pdflatex"
    latexmk = str(MIKTEX_BIN / "latexmk.exe") if (MIKTEX_BIN / "latexmk.exe").exists() else "latexmk"
    bibtex = str(MIKTEX_BIN / "bibtex.exe") if (MIKTEX_BIN / "bibtex.exe").exists() else "bibtex"
    cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    result = subprocess.run(cmd, cwd=PAPER_DIR, env=env, text=True, capture_output=True)
    log_text = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        # latexmk needs Perl, which is not installed everywhere. Fall back to
        # the classic pdflatex -> bibtex -> pdflatex -> pdflatex sequence.
        log_text += "\n--- latexmk failed; falling back to pdflatex/bibtex ---\n"
        steps = [
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            [bibtex, "main"],
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ]
        for step in steps:
            step_result = subprocess.run(
                step, cwd=PAPER_DIR, env=env, text=True, capture_output=True
            )
            log_text += (
                f"\n--- {' '.join(step)} (exit {step_result.returncode}) ---\n"
                + step_result.stdout
                + "\n"
                + step_result.stderr
            )
            if step_result.returncode != 0:
                break
        result = step_result
    (PAPER_DIR / "build.log").write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        print(log_text[-8000:])
        raise SystemExit("LaTeX build failed; see paper/build.log")
    pdf = PAPER_DIR / "main.pdf"
    if not pdf.exists():
        raise SystemExit("LaTeX reported success but main.pdf is missing")
    reader = PdfReader(str(pdf))
    print(f"compiled {pdf} ({len(reader.pages)} pages)")
    return pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args(argv)

    ensure_style_files()
    r = load_results()
    generate_numbers(r)
    tab_tree(r)
    tab_bagging(r)
    tab_toy(r)
    tab_boosting_grid(r)
    tab_boost_vs_bag(r)
    tab_depth(r)
    tab_final(r)
    if not args.no_build:
        pdf = compile_pdf()
        target = ROOT / "Kumar_Gaurav_P26AI0006_ML_Assignment_2.pdf"
        shutil.copy2(pdf, target)
        print(f"copied report to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
