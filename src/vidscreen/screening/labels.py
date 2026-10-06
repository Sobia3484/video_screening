"""Build the final binary labels of the 300-video sample from the two human annotation workbooks."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

VALID = {"relevant", "irrelevant"}


def _norm(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def build_final_labels(first_pass_xlsx: Path, review_xlsx: Path, selection_csv: Path, prelabels_csv: Path = None):
    """first pass (relevant/irrelevant/potentially_relevant) + review (resolves potentially) -> binary labels.

    Returns (labels_df, report_dict). Raises if any row is unlabeled or still potentially_relevant.
    """
    first = pd.read_excel(first_pass_xlsx, sheet_name="Labeling", dtype={"sample_id": str})
    first["label_first_pass"] = _norm(first["label"])
    first["first_pass_note"] = first["notes"].fillna("") if "notes" in first else ""
    review = pd.read_excel(review_xlsx, sheet_name="Review", dtype={"sample_id": str})
    review["review_label"] = _norm(review["new_label"])
    review["review_reason"] = review["reason"].fillna("") if "reason" in review else ""
    sel = pd.read_csv(selection_csv, encoding="utf-8-sig")

    df = first[["sample_id", "label_first_pass", "first_pass_note"]].merge(
        review[["sample_id", "review_label", "review_reason"]], on="sample_id", how="left", validate="one_to_one")
    df = df.merge(sel[["sample_id", "video_uid", "platform", "stratum", "split", "sampling_weight"]],
                  on="sample_id", how="left", validate="one_to_one")
    if df["video_uid"].isna().any():
        raise ValueError("Some labeled rows are missing from sample_selection.csv")
    df["resolved_in_review"] = df["review_label"].notna()
    df["label_final"] = df["review_label"].fillna(df["label_first_pass"])
    if df["label_final"].isna().any():
        raise ValueError(f"Unlabeled rows: {df.loc[df.label_final.isna(), 'sample_id'].tolist()}")
    bad = df.loc[~df["label_final"].isin(VALID), ["sample_id", "label_final"]]
    if len(bad):
        raise ValueError(f"Labels other than relevant/irrelevant remain (resolve them in the review sheet): {bad.to_dict('records')}")

    report = {
        "rows": int(len(df)),
        "label_counts": df["label_final"].value_counts().to_dict(),
        "by_split": pd.crosstab(df["split"], df["label_final"]).to_dict("index"),
        "resolved_in_review": int(df["resolved_in_review"].sum()),
    }
    if prelabels_csv and Path(prelabels_csv).exists():
        ai = pd.read_csv(prelabels_csv, encoding="utf-8-sig")[["sample_id", "ai_label"]]
        df = df.merge(ai, on="sample_id", how="left")
        three = (df["ai_label"] == df["label_first_pass"])
        b_ai, b_h = df["ai_label"] == "relevant", df["label_final"] == "relevant"
        po = float((b_ai == b_h).mean())
        pe = float(b_ai.mean() * b_h.mean() + (1 - b_ai.mean()) * (1 - b_h.mean()))
        report["ai_vs_human"] = {
            "three_class_agreement": round(float(three.mean()), 4),
            "binary_agreement": round(po, 4),
            "binary_cohen_kappa": round((po - pe) / (1 - pe), 4),
        }
    cols = ["sample_id", "video_uid", "platform", "stratum", "split", "sampling_weight", "label_first_pass",
            "review_label", "label_final", "resolved_in_review", "first_pass_note", "review_reason"]
    if "ai_label" in df:
        cols.insert(6, "ai_label")
    return df[cols].sort_values("sample_id").reset_index(drop=True), report
