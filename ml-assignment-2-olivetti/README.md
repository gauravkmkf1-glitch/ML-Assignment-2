# Machine Learning Assignment 2 - Olivetti Faces

**Kumar Gaurav** - Roll number **P26AI0006**

Public repository: **REPOSITORY_URL_TO_BE_ADDED**

This repository contains a complete, reproducible implementation of the three
questions in Machine Learning Assignment 2. Every number in the report is
produced from the saved split by the scripts below; no result is copied from a
starter notebook and no test image is used for model selection.

## What is implemented

| Part | Contents |
| --- | --- |
| Split | One `numpy.random.default_rng(260006)` stream, identities in ascending order, one permutation of each identity's ten images, 6 train / 2 validation / 2 test. Saved to `results/split.npz` and `results/split.csv`. |
| Q1 | Depth x leaf-size decision-tree grid; bagging with `B in {1,5,10,25,50,100,200}`; OOB accuracy; pairwise disagreement; vote margin; clean vs masked test; face case studies. |
| Q2 | SAMME implemented from scratch with `alpha = log((1-eps)/eps) + log(K-1)`, explicit random-learner rejection, normalised `q`; toy round; depth x rounds grid; 25/50/100 comparison; weighted-face case; masked-response case. |
| Q3 | 30 within-identity bootstrap resamples; measured squared bias and variance; selected tree/bagging/boosting; depth study; final table. |

Bagging and SAMME are **student implementations** in `src/common.py`
(`BaggingEnsemble` and `SAMME`). `sklearn.tree.DecisionTreeClassifier` is used
only as the permitted base learner. Raw flattened 4096-pixel vectors are used
throughout; there is no PCA or any other dimensionality reduction.

## Data and seed

* Dataset: `sklearn.datasets.fetch_olivetti_faces` - 40 identities x 10 images,
  64 x 64 pixels, pixel values in `[0, 1]`.
* Seed: every numeric character of the roll number `P26AI0006` in order, i.e.
  **260006**, passed to `numpy.random.default_rng`.
* Split SHA-256: `8903d5baad04a89c2d2d5024b87a913185c36390262696154c406da7b0e53408`.
* Mask: test images only, rows **24-39** and columns **24-39** (a centred
  16 x 16 block) are set to zero. Training and validation images are clean.
* Model selection uses **clean validation accuracy only**. The clean test set
  is used only for the final reporting requested by the assignment. The masked
  test set is never used for training or selection.

The compact cache `data/olivetti_faces.npz` (1.8 MB) is tracked so the pipeline
does not depend on the download server being available. If it is absent,
`src/common.py` calls `fetch_olivetti_faces` and writes the cache.

## Reproduce

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m src.run_all
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m src.run_all
```

`src.run_all` executes, in order:

1. `python -m src.make_split` - recreate and save the 240/80/80 split.
2. `python -m src.run_experiments` - Q1-Q3, predictions, logs, CSV/JSON files.
3. `python -m src.finalize_results` - derive report tables from the saved
   measurements (no refitting, no model selection).
4. `python -m src.make_figures` - all seven report figures from saved results.
5. `python -m src.build_report` - LaTeX tables/macros and the CVPR PDF.

Individual steps can be run separately. `python -m src.build_report --no-build`
writes the tables and `paper/generated/numbers.tex` without invoking LaTeX.
LaTeX needs `pdflatex` and `bibtex`; `latexmk` is used when available, with a
`pdflatex -> bibtex -> pdflatex -> pdflatex` fallback. On Windows the script
also looks in the default MiKTeX installation directory.

Expected runtime on the run machine (10 parallel jobs): about 1.3 hours,
dominated by the bagging and bias-variance refits. The exact runtime and
environment are stored in `results/run_summary.json` and
`results/environment.json`.

## Checks

```bash
python -m pytest -q                 # unit tests for split, mask, voting, SAMME, bias-variance
python -m src.verify_submission     # contract checks: split, mask, results, figures, PDF metadata
```

`verify_submission` checks the 240/80/80 split, the exact mask, the absence of
PCA/SVD calls, the presence of every required artifact, the page limit (body at
most 8 pages before references) and the page-1 name, roll number, repository URL
and commit hash.

## Repository layout

```
configs/config.json          all grids, seed, mask, paths, parallel job count
data/olivetti_faces.npz      cached dataset (raw pixels, [0,1])
figures/                     fig1-fig7 as PDF and 300-dpi PNG
logs/run.log                 full experiment log
paper/                       CVPR 2026 source (main.tex, cvpr.sty, bibliography)
results/                     split, predictions, histories, CSV/JSON results
scripts/                     convenience shell/PowerShell runners
src/common.py                seed, data, split, mask, voting, BaggingEnsemble, SAMME
src/make_split.py            split creation and summary
src/run_experiments.py       complete Q1-Q3 pipeline
src/finalize_results.py      derived tables from saved measurements
src/make_figures.py          figures from saved results only
src/build_report.py          LaTeX numbers/tables and PDF build
src/run_all.py               one-command reproduction
src/verify_submission.py     submission contract checks
tests/test_common.py         unit tests
```

## Key results (clean validation selection)

| Model | Selected setting | Validation | Clean test | Masked test |
| --- | --- | ---: | ---: | ---: |
| Single tree | unrestricted depth, leaf 1 | 0.6000 | 0.5000 | 0.3750 |
| Bagging | B = 200 | 0.9250 | 0.8375 | 0.7375 |
| SAMME | depth 4, 50 rounds | 0.4375 | 0.3875 | 0.2750 |

The single tree overfits (training accuracy 1.0000, 41 leaves). Bagging is the
clear winner on both clean and masked test sets. Boosting is stable and its
training error falls, but its hard-vote multiclass decisions remain weak on the
240 clean training images: the selected 50-round model ends with weighted
training error 0.6143, above the 39/40 random-guess threshold, and no round is
rejected. The report discusses these outcomes rather than hiding them.

Over 30 within-identity bootstrap resamples the measured decomposition is:
single tree bias^2 0.4208 / variance 0.6417; bagging bias^2 0.5270 /
variance 0.0208; boosting bias^2 0.8855 / variance 0.0206. Bagging removes
almost all of the tree's variance at the cost of a modest bias increase, which
is why it wins on accuracy; boosting has low variance but a bias more than twice
the tree's.

## Notes on timing

Bagging fits trees in parallel, so its wall-clock time depends on the machine
and scheduler. The report uses the **sum of per-tree fit times** as a stable
serial-equivalent training cost; `results/bagging_grid.csv` contains both
`train_seconds_serial` and `train_seconds_wall_full_200`.
`results/boosting_vs_bagging.csv` stores the serial-equivalent bagging time in
`train_seconds` and the measured wall time in `wall_seconds`.

## Contact

Kumar Gaurav - P26AI0006.
