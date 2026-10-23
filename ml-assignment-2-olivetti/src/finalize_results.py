"""Recompute derived tables that depend only on already-measured results.

The experiment pipeline writes the raw measurements.  This step exists because
two derived tables have presentation-specific columns:

* ``final_table.csv`` combines the clean/masked accuracy table with the
  30-refit bias--variance table, and carries the mean confidence columns used in
  Table 5 of the report.
* ``boosting_vs_bagging.csv`` reports bagging training cost as the sum of
  per-tree fit times, which is a stable serial-equivalent measure on machines
  where trees are fitted in parallel.  The measured wall time is retained in a
  separate column.

The script reads only files under ``results/`` and never refits a model, so it
cannot change a prediction or an accuracy.  It is idempotent.

Usage::

    python -m src.finalize_results
"""

from __future__ import annotations

import pandas as pd

from .common import LOGS_DIR, RESULTS_DIR, artifact_manifest, dataframe_to_csv, setup_logging

LOG = setup_logging(LOGS_DIR / "finalize.log")


def finalize_final_table() -> None:
    path = RESULTS_DIR / "final_table.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    comparison = pd.read_csv(RESULTS_DIR / "final_comparison.csv").set_index("model")
    for column in ("clean_mean_confidence", "masked_mean_confidence"):
        if column not in table.columns:
            table[column] = [float(comparison.loc[name, column]) for name in table["model"]]
    # Keep the report's column order, then append the extra diagnostic columns.
    order = [
        "model",
        "clean_test_accuracy",
        "masked_test_accuracy",
        "clean_mean_confidence",
        "masked_mean_confidence",
        "bias_squared",
        "variance",
        "refit_accuracy_mean",
        "refit_accuracy_std",
    ]
    table = table[[c for c in order if c in table.columns]]
    dataframe_to_csv(table, path)
    LOG.info("final_table.csv includes confidence columns")


def finalize_boosting_vs_bagging() -> None:
    path = RESULTS_DIR / "boosting_vs_bagging.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    grid = pd.read_csv(RESULTS_DIR / "bagging_grid.csv").set_index("B")
    if "wall_seconds" not in table.columns:
        table["wall_seconds"] = float("nan")
    for i, row in table.iterrows():
        if row["family"] != "Bagging":
            if pd.isna(table.at[i, "wall_seconds"]):
                table.at[i, "wall_seconds"] = float(row["train_seconds"])
            continue
        b = int(row["n_learners"])
        # Preserve the old wall-clock value the first time this runs.
        if pd.isna(table.at[i, "wall_seconds"]):
            table.at[i, "wall_seconds"] = float(row["train_seconds"])
        table.at[i, "train_seconds"] = float(grid.loc[b, "train_seconds_serial"])
        table.at[i, "note"] = "bootstrap trees; frozen tree setting; serial-equivalent time"
    dataframe_to_csv(table, path)
    LOG.info("boosting_vs_bagging.csv uses serial-equivalent bagging times")


def main() -> int:
    finalize_final_table()
    finalize_boosting_vs_bagging()
    files = [p for p in RESULTS_DIR.glob("*") if p.is_file() and p.name != "artifact_manifest.json"]
    artifact_manifest(files, RESULTS_DIR / "artifact_manifest_results.json")
    LOG.info("Refreshed derived tables and artifact manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
