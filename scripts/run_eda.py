#!/usr/bin/env python
"""Exploratory data analysis.

    python scripts/run_eda.py                              # main + confident datasets, all available sections
    python scripts/run_eda.py --dataset main --sections duration views

Reads  : data/final/relevant_videos.csv and data/final/relevant_confident_videos.csv
Writes : data/final/relevant_features.csv (+ confident), reports/eda/<dataset>/ (EDA_report.md, tables/, figures/)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.config import load_settings  # noqa: E402
from vidscreen.eda.runner import DEFAULT_SECTIONS, run_eda  # noqa: E402
from vidscreen.eda.sections import SECTIONS  # noqa: E402
from vidscreen.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["main", "confident", "both"], default="both")
    ap.add_argument("--sections", nargs="*", default=DEFAULT_SECTIONS, help="available: " + ", ".join(SECTIONS))
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    s = load_settings()
    setup_logging(s.log_dir)
    ds = ("main", "confident") if a.dataset == "both" else (a.dataset,)
    done = run_eda(s, ds, a.sections, a.out_dir)
    for k, v in done.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
