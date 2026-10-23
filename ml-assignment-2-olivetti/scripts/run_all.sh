#!/usr/bin/env bash
# Exact reproduction (Linux/macOS). Run from the repository root.
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m src.run_all
python -m pytest -q
python -m src.verify_submission
