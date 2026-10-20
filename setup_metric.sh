#!/usr/bin/env bash
# Install the official competition metric (`topometrics`) into the uv env.
# Source: Kaggle dataset sohier/vesuvius-metric-resources (not on PyPI/GitHub).
# Download it from https://www.kaggle.com/datasets/sohier/vesuvius-metric-resources
# and place the zip in work/metric/ before running this script.
set -euo pipefail
cd "$(dirname "$0")"

DST=work/metric
[ -d "$DST/topological-metrics-kaggle" ] || unzip -q "$DST/vesuvius-metric-resources.zip" -d "$DST"

uv sync                                                                   # installs topometrics-3d (editable) + cmake + pybind11
uv run bash "$DST/topological-metrics-kaggle/scripts/build_betti.sh"      # compiles the C++ Betti-matching extension
uv run python -c "import topometrics.leaderboard; print('topometrics OK')"
