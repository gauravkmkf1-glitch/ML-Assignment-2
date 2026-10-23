"""One-command reproduction of the complete submission.

    python -m src.run_all

This runs, in order:

1. src.make_split          (saved, roll-number-seeded 6/2/2 split)
2. src.run_experiments     (all Q1--Q3 experiments, logs, result files)
3. src.finalize_results    (derived tables; no refitting)
4. src.make_figures        (all report figures from saved results)
5. src.build_report        (LaTeX tables/numbers and the CVPR PDF)

The first run downloads the Olivetti Faces data (about 4 MB) unless the
cached ``data/olivetti_faces.npz`` is present.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT


def run(module: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module] + (extra or [])
    print("\n" + "=" * 78)
    print("$ " + " ".join(cmd))
    print("=" * 78, flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"Step {module} failed with exit code {result.returncode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-report", action="store_true", help="skip LaTeX compilation")
    parser.add_argument("--force-split", action="store_true", help="overwrite the saved split")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    run("src.make_split", ["--force"] if args.force_split else [])
    run("src.run_experiments")
    run("src.finalize_results")
    run("src.make_figures")
    if not args.skip_report:
        run("src.build_report")
    print(f"\nAll steps completed in {(time.perf_counter() - t0) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
