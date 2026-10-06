#!/usr/bin/env python
"""Revised screening (score-based). Combined vs per-platform models, text vs text+features, creator-grouped CV.

    python scripts/run_screening_v2.py --stage cv     # model comparison only (no screening outputs)
    python scripts/run_screening_v2.py                # comparison + final screening of all videos
    python scripts/run_screening_v2.py --no-embeddings
Previous (v1) outputs in data/final are moved to data/final/v1_initial/ before new outputs are written.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.config import load_settings  # noqa: E402
from vidscreen.logging_setup import setup_logging  # noqa: E402
from vidscreen.screening.models import EMBEDDING_MODEL  # noqa: E402
from vidscreen.screening.v2 import run_v2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cv", "all"], default="all")
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    s = load_settings()
    setup_logging(s.log_dir)
    r = run_v2(s, a.stage, not a.no_embeddings, a.out_dir, repeats=a.repeats, folds=a.folds, embedding_model=a.embedding_model)
    print(json.dumps({"selected": r["selected"], "thresholds": r["thresholds"], **({"funnel": r["funnel"]} if "funnel" in r else {})}, indent=2, default=str))
    print("\nReport: data/final/screening_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())