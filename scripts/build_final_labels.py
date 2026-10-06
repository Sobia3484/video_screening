#!/usr/bin/env python
"""Step 2b: human annotation workbooks -> data/labeled/final_labels.csv (binary labels, with dev/test split).

Inputs (data/labeled/):  labeled_sample_300.xlsx, labeled_sample_300_reviewed.xlsx, sample_selection.csv, ai_prelabels.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidscreen.screening.labels import build_final_labels  # noqa: E402

D = Path(__file__).resolve().parents[1] / "data" / "labeled"

if __name__ == "__main__":
    labels, report = build_final_labels(
        D / "labeled_sample_300.xlsx", D / "labeled_sample_300_reviewed.xlsx",
        D / "sample_selection.csv", D / "ai_prelabels.csv")
    labels.to_csv(D / "final_labels.csv", index=False, encoding="utf-8-sig")
    (D / "final_labels_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {D/'final_labels.csv'}")
