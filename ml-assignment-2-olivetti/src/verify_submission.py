"""Contract checks for the final submission.

Run from the repository root:

    python -m src.verify_submission

Exits non-zero if a hard requirement is missing.  PDF page-limit and page-1
metadata checks are strict when the report has been built.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from .common import (
    MASK_COL_END,
    MASK_COL_START,
    MASK_ROW_END,
    MASK_ROW_START,
    RESULTS_DIR,
    ROLL_NUMBER,
    ROOT,
    SEED,
    load_dataset,
    load_split,
    mask_centered_block,
    validate_split,
)

REQUIRED_RESULTS = [
    "split.npz",
    "split.csv",
    "split_summary.json",
    "tree_grid.csv",
    "selected_tree.json",
    "bagging_grid.csv",
    "selected_bagging.json",
    "boosting_grid.csv",
    "selected_boosting.json",
    "boosting_history.csv",
    "boosting_vs_bagging.csv",
    "final_comparison.csv",
    "bias_variance.csv",
    "depth_bias_variance.csv",
    "final_table.csv",
    "toy_samme.json",
    "q1_cases.json",
    "q2_weight_case.json",
    "q2_mask_case.json",
    "q3_depth_conclusion.json",
    "run_summary.json",
    "environment.json",
    "requirements_freeze.txt",
    "artifact_manifest_results.json",
]

REQUIRED_FIGURES = [
    "fig1_tree_grid.pdf",
    "fig2_bagging.pdf",
    "fig3_q1_cases.pdf",
    "fig4_samme_history.pdf",
    "fig5_weight_case.pdf",
    "fig6_mask_case.pdf",
    "fig7_bias_variance.pdf",
]


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"FAIL: {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-draft", action="store_true", help="downgrade PDF metadata/page checks to warnings")
    args = parser.parse_args(argv)
    errors: list[str] = []
    warnings: list[str] = []

    if ROLL_NUMBER != "P26AI0006" or SEED != 260006:
        fail(errors, "roll number / seed mismatch")

    for name in REQUIRED_RESULTS:
        if not (RESULTS_DIR / name).exists():
            fail(errors, f"missing results/{name}")
    for name in REQUIRED_FIGURES:
        if not (ROOT / "figures" / name).exists():
            fail(errors, f"missing figures/{name}")

    try:
        dataset = load_dataset()
        train, validation, test = load_split(RESULTS_DIR / "split.npz", dataset.target)
        validate_split(train, validation, test, dataset.target)
        assert (len(train), len(validation), len(test)) == (240, 80, 80)
        print("OK: saved split is a 240/80/80 partition with 6/2/2 per identity")
    except Exception as exc:
        fail(errors, f"split check failed: {exc}")

    try:
        dataset = load_dataset()
        X = dataset.data[:3]
        M = mask_centered_block(X)
        assert np.all(M.reshape(-1, 64, 64)[:, MASK_ROW_START:MASK_ROW_END, MASK_COL_START:MASK_COL_END] == 0)
        outside = np.ones((64, 64), dtype=bool)
        outside[MASK_ROW_START:MASK_ROW_END, MASK_COL_START:MASK_COL_END] = False
        assert np.array_equal(M.reshape(-1, 64, 64)[:, outside], X.reshape(-1, 64, 64)[:, outside])
        print("OK: mask zeros exactly rows/columns 24..39 and changes nothing else")
    except Exception as exc:
        fail(errors, f"mask check failed: {exc}")

    # Guard the experiment code against accidental dimensionality reduction.
    # Build the needles from prefixes so this checker does not trip the same
    # source scan when it inspects its own file.
    for path in (ROOT / "src").glob("*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for stem in ("PCA", "TruncatedSVD", "KernelPCA"):
            banned = f"{stem}("
            if banned in text:
                fail(errors, f"dimensionality reduction found in {path.name}: {banned}")

    pdf = ROOT / "Kumar_Gaurav_P26AI0006_ML_Assignment_2.pdf"
    if not pdf.exists():
        pdf = ROOT / "paper" / "main.pdf"
    if not pdf.exists():
        (warnings if args.allow_draft else errors).append("report PDF is missing")
        print("WARN: report PDF is missing")
    else:
        reader = PdfReader(str(pdf))
        page_count = len(reader.pages)
        body_pages = page_count
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if re.search(r"\bReferences\b", text):
                body_pages = i
                break
        if body_pages > 8:
            (warnings if args.allow_draft else errors).append(f"body has {body_pages} pages before references (max 8)")
            print(f"WARN: body has {body_pages} pages before references (max 8)")
        else:
            print(f"OK: {body_pages} pages before references (max 8; total {page_count})")
        first = reader.pages[0].extract_text() or ""
        for needle in ("Kumar Gaurav", "P26AI0006"):
            if needle not in first:
                (warnings if args.allow_draft else errors).append(f"page 1 does not contain {needle!r}")
                print(f"WARN: page 1 does not contain {needle!r}")
        if re.search(r"REPOSITORY_URL_TO_BE_ADDED|TO_BE_ADDED", first):
            (warnings if args.allow_draft else errors).append("page 1 repository URL is still a placeholder")
            print("WARN: page 1 repository URL is still a placeholder")
        if "UNCOMMITTED" in first:
            (warnings if args.allow_draft else errors).append("page 1 still says UNCOMMITTED")
            print("WARN: page 1 still says UNCOMMITTED")

    if errors:
        print(f"\n{len(errors)} hard error(s):")
        for e in errors:
            print(" -", e)
        return 1
    print("\nAll submission contract checks passed.")
    if warnings:
        print("Warnings (draft mode):")
        for w in warnings:
            print(" -", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
