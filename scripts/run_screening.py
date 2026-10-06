#!/usr/bin/env python
"""Steps 3-4: rules + models -> pick the best on development -> evaluate once on test -> screen all videos.

    python scripts/run_screening.py --stage dev     # development only (cross-validation); test is NOT touched
    python scripts/run_screening.py                 # everything: test evaluation + screening of all 1,929 videos
    python scripts/run_screening.py --no-embeddings # skip the pretrained transformer (rules + TF-IDF only)

Inputs : data/intermediate/master_cleaned.csv, data/labeled/final_labels.csv
Outputs: data/final/relevant_videos.csv, irrelevant_videos.csv, screening_results.csv, screening_report.md/.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

from vidscreen.config import load_settings  # noqa: E402
from vidscreen.logging_setup import setup_logging  # noqa: E402
from vidscreen.screening.models import EMBEDDING_MODEL  # noqa: E402
from vidscreen.screening.pipeline import run_screening  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "all"], default="all")
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    s = load_settings()
    setup_logging(s.log_dir)
    report = run_screening(s, args.stage, not args.no_embeddings, args.out_dir, embedding_model=args.embedding_model)
    print(json.dumps(report.get("funnel", report["dev_cv"]["threshold_optimised"]), indent=2, default=str))
    print("\nReport: data/final/screening_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())