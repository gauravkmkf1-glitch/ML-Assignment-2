"""Create and save the one split used by every experiment.

Run from the repository root:

    python -m src.make_split

The split is determined solely by the roll-number seed and the identity order,
so re-running this command reproduces byte-identical index arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (
    RESULTS_DIR,
    ROLL_NUMBER,
    SEED,
    dataset_checksum,
    json_dump,
    load_dataset,
    make_split,
    save_split,
    split_checksum,
    validate_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args()

    dataset = load_dataset()
    split_path = RESULTS_DIR / "split.npz"
    if split_path.exists() and not args.force:
        with np.load(split_path, allow_pickle=False) as z:
            existing = (
                np.asarray(z["train"], dtype=int),
                np.asarray(z["validation"], dtype=int),
                np.asarray(z["test"], dtype=int),
            )
        expected = make_split(dataset.target)
        if all(np.array_equal(a, b) for a, b in zip(existing, expected)):
            print(f"Existing split is valid and reproducible: {split_path}")
            validate_split(*existing, dataset.target)
            return
        raise SystemExit(
            f"{split_path} exists but does not match the roll-number split; "
            "rerun with --force after checking the file."
        )

    train, validation, test = make_split(dataset.target)
    validate_split(train, validation, test, dataset.target)
    save_split(split_path, train, validation, test, dataset.target, SEED)

    summary = {
        "student": "Kumar Gaurav",
        "roll_number": ROLL_NUMBER,
        "seed": SEED,
        "seed_rule": "all numeric characters of the roll number, in order",
        "split_rule": (
            "identities in ascending order; for each identity one "
            "numpy.random.default_rng(seed).permutation of its 10 indices; "
            "first 6 train, next 2 validation, last 2 test"
        ),
        "counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "per_identity_counts": {"train": 6, "validation": 2, "test": 2},
        "split_sha256": split_checksum(train, validation, test),
        "dataset": dataset_checksum(dataset),
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "test_indices": test.tolist(),
        "files": {
            "split_npz": "results/split.npz",
            "split_csv": "results/split.csv",
        },
    }
    json_dump(RESULTS_DIR / "split_summary.json", summary)
    print(f"Wrote {split_path}")
    print(f"Wrote {split_path.with_suffix('.csv')}")
    print(f"Wrote {RESULTS_DIR / 'split_summary.json'}")
    print(
        f"Train/validation/test sizes: {len(train)}/{len(validation)}/{len(test)}; "
        f"seed={SEED}; split_sha256={summary['split_sha256'][:16]}..."
    )


if __name__ == "__main__":
    main()
