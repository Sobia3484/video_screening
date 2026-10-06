"""Step 3-4: rules + models -> compare on development (cross-validation) -> test once -> screen all videos."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .evaluate import auc, best_threshold, bootstrap_ci, by_group, metrics
from .models import EMBEDDING_MODEL, encode_texts, make_estimator
from .rules import apply_rules
from .text import build_text

log = logging.getLogger("vidscreen")
SEED = 42
ORDER = ["tfidf", "hybrid_tfidf", "embed", "hybrid_embed"]  # simplest first (tie-break)
REVIEW_BAND = 0.15
THRESHOLD_BETA = 1.0  # F1: there is no mandatory manual review, so false positives end up in the final dataset
RULE_TO_BINARY = {"relevant": 1, "potentially_relevant": 1, "irrelevant": 0}  # recall-oriented mapping for the baseline


def _num(x):
    return float(x) if isinstance(x, (np.floating, float)) else (int(x) if isinstance(x, (np.integer,)) else x)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return _num(obj)


def load_frame(settings) -> pd.DataFrame:
    master = pd.read_csv(settings.intermediate_dir / "master_cleaned.csv", dtype={"video_id": str}, encoding="utf-8-sig").fillna("")
    labels = pd.read_csv(settings.root / "data" / "labeled" / "final_labels.csv", encoding="utf-8-sig")
    df = master.merge(labels[["video_uid", "label_final", "split", "sampling_weight", "sample_id"]], on="video_uid", how="left", validate="one_to_one")
    df["text"] = df.apply(build_text, axis=1)
    res = df.apply(lambda r: apply_rules(r["title"], r["description"], r["hashtags"], r["tags"]), axis=1)
    df["rule_label"] = [r.label for r in res]
    df["rule_reason"] = [r.reason for r in res]
    df["rule_terms"] = [";".join(r.child[:3] + r.topic[:3]) for r in res]
    df["topic_gate"] = [1.0 if r.topic_gate else 0.0 for r in res]
    df["y"] = df["label_final"].map({"relevant": 1, "irrelevant": 0})
    return df


def _oof_proba(kind, X, y, strat, seed=SEED, folds=5):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), strat):
        est = make_estimator(kind, seed).fit(_take(X, tr), y[tr])
        oof[te] = est.predict_proba(_take(X, te))[:, 1]
    return oof


def _take(X, idx):
    return X[idx]


def _row(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    m = metrics(y, pred)
    m["auc"] = auc(y, proba)
    m["threshold"] = round(float(threshold), 2)
    return m


def run_screening(settings, stage: str = "all", use_embeddings: bool = True, out_dir: Optional[Path] = None,
                  encoder: Optional[Callable] = None, embedding_model: str = EMBEDDING_MODEL) -> dict:
    out_dir = Path(out_dir) if out_dir else settings.final_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_frame(settings)
    texts = df["text"].to_numpy()
    feats = {"tfidf": texts}
    if use_embeddings:
        enc = encoder or (lambda t: encode_texts(t, settings.cache_dir, embedding_model))
        feats["embed"] = np.asarray(enc(list(texts)))
    kinds = list(feats)

    dev_idx = np.where(df["split"] == "dev")[0]
    test_idx = np.where(df["split"] == "test")[0]
    lab_idx = np.where(df["y"].notna())[0]
    y_dev = df["y"].to_numpy()[dev_idx].astype(int)
    strat = (df["y"].astype("Int64").astype(str) + df["platform"]).to_numpy()[dev_idx]
    gate = df["topic_gate"].to_numpy()

    # ---------- 1. development: 5-fold cross-validation (+ rules baseline)
    oof = {}
    for kind in kinds:
        log.info("Cross-validating %s on %d development videos ...", kind, len(dev_idx))
        oof[kind] = _oof_proba(kind, _take(feats[kind], dev_idx), y_dev, strat)
        oof["hybrid_" + kind] = oof[kind] * gate[dev_idx]
    cands = [c for c in ORDER if c in oof]
    half = {c: _row(y_dev, oof[c], 0.5) for c in cands}
    top = max(h["f1"] for h in half.values())
    near = [c for c in cands if top - half[c]["f1"] <= 0.01]          # indistinguishable on F1 -> higher recall, then simpler
    selected = max(near, key=lambda c: (half[c]["recall"], -cands.index(c)))
    thresholds = {c: best_threshold(y_dev, oof[c], beta=THRESHOLD_BETA) for c in cands}
    threshold = thresholds[selected]
    dev_table = {c: _row(y_dev, oof[c], thresholds[c]) for c in cands}
    dev_at_half = half
    rules_dev = metrics(y_dev, df["rule_label"].map(RULE_TO_BINARY).to_numpy()[dev_idx])
    report = {
        "stage": stage, "seed": SEED, "embedding_model": embedding_model if use_embeddings else None,
        "n_dev": int(len(dev_idx)), "n_test": int(len(test_idx)), "n_all": int(len(df)),
        "dev_cv": {"threshold_optimised": dev_table, "threshold_0.5": dev_at_half},
        "rules_baseline_dev": {**rules_dev, "note": "rules were tuned on this data, so this number is optimistic"},
        "selected": selected, "selected_threshold": threshold,
        "selection_rule": "highest dev-CV F1 (ties within 0.01 -> higher recall, then simpler); threshold maximises F1 on out-of-fold dev scores",
    }
    log.info("Selected: %s (threshold %.2f)", selected, threshold)
    if stage == "dev":
        _write_reports(report, out_dir, df=None)
        return report

    # ---------- 2. test: fit on development only, evaluate once
    y_test = df["y"].to_numpy()[test_idx].astype(int)
    test_table, fitted = {}, {}
    for kind in kinds:
        est = make_estimator(kind, SEED).fit(_take(feats[kind], dev_idx), y_dev)
        fitted[kind] = est
        p = est.predict_proba(_take(feats[kind], test_idx))[:, 1]
        for name, proba in ((kind, p), ("hybrid_" + kind, p * gate[test_idx])):
            thr = thresholds[name]
            test_table[name] = _row(y_test, proba, thr)
            if name == selected:
                pred = (proba >= thr).astype(int)
                test_df = df.iloc[test_idx].assign(pred=pred)
                report["test_selected"] = {
                    **test_table[name],
                    "ci95": {k: bootstrap_ci(y_test, pred, k, seed=SEED) for k in ("precision", "recall", "f1")},
                    "by_platform": by_group(test_df, "y", "pred"),
                    "population_weighted": metrics(y_test, pred, weights=df["sampling_weight"].to_numpy()[test_idx]),
                }
    report["test_all_candidates"] = {c: test_table[c] for c in cands}
    report["rules_baseline_test"] = metrics(y_test, df["rule_label"].map(RULE_TO_BINARY).to_numpy()[test_idx])

    # ---------- 3. screen every video: human label where available, else the selected model
    base = selected.replace("hybrid_", "")
    final_est = make_estimator(base, SEED).fit(_take(feats[base], lab_idx), df["y"].to_numpy()[lab_idx].astype(int))
    proba = final_est.predict_proba(feats[base])[:, 1]
    if selected.startswith("hybrid_"):
        proba = proba * gate
    pred = (proba >= threshold).astype(int)
    human = df["y"].notna().to_numpy()
    df["probability"] = np.round(proba, 4)
    df["screening_label"] = np.where(human, df["label_final"].fillna(""), np.where(pred == 1, "relevant", "irrelevant"))
    df["label_source"] = np.where(human, "human", "model")
    df["method"] = np.where(human, "human_label", selected)
    df["needs_review"] = (~human) & ((np.abs(proba - threshold) < REVIEW_BAND) | (df["rule_label"].map(RULE_TO_BINARY).to_numpy() != pred))
    _write_outputs(df, report, out_dir)
    _write_reports(report, out_dir, df)
    return report


def _write_outputs(df: pd.DataFrame, report: dict, out_dir: Path) -> None:
    keep = ["video_uid", "platform", "video_id", "url", "title", "screening_label", "label_source", "method", "probability",
            "needs_review", "rule_label", "rule_reason", "rule_terms", "query_ids", "query_subtopics"]
    df[keep].to_csv(out_dir / "screening_results.csv", index=False, encoding="utf-8-sig")
    extra = ["screening_label", "label_source", "method", "probability", "needs_review"]
    master_cols = [c for c in df.columns if c not in set(extra) | {"text", "rule_label", "rule_reason", "rule_terms", "topic_gate", "y",
                                                                     "label_final", "split", "sampling_weight", "sample_id"}]
    rel = df[df["screening_label"] == "relevant"]
    irr = df[df["screening_label"] == "irrelevant"]
    rel[master_cols + extra].to_csv(out_dir / "relevant_videos.csv", index=False, encoding="utf-8-sig")
    irr[master_cols + extra].to_csv(out_dir / "irrelevant_videos.csv", index=False, encoding="utf-8-sig")
    report["funnel"] = {
        "cleaned_videos": int(len(df)),
        "human_labeled": int((df["label_source"] == "human").sum()),
        "model_labeled": int((df["label_source"] == "model").sum()),
        "relevant": int(len(rel)), "irrelevant": int(len(irr)),
        "relevant_by_platform": rel["platform"].value_counts().to_dict(),
        "irrelevant_by_platform": irr["platform"].value_counts().to_dict(),
        "model_labeled_flagged_for_optional_review": int(df["needs_review"].sum()),
    }


def _fmt(m: dict) -> str:
    return f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m.get('auc', float('nan')):.3f} | {m.get('threshold', '')}"


def _write_reports(report: dict, out_dir: Path, df) -> None:
    report = _clean(report)
    (out_dir / "screening_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# Screening report", "",
         f"Seed {report['seed']}. Development {report['n_dev']} videos, test {report['n_test']}, all cleaned videos {report['n_all']}.", "",
         "## Development: 5-fold cross-validation (threshold maximises F1 on out-of-fold scores)", "",
         "| method | precision | recall | F1 | AUC | threshold |", "|---|---|---|---|---|---|"]
    for c, m in report["dev_cv"]["threshold_optimised"].items():
        L.append(f"| {c} | {_fmt(m)} |")
    r = report["rules_baseline_dev"]
    L += ["", f"Rules baseline on development (optimistic, rules were tuned here): precision {r['precision']}, recall {r['recall']}, F1 {r['f1']}.", "",
          f"**Selected method:** `{report['selected']}` (threshold {report['selected_threshold']}). Rule: {report['selection_rule']}."]
    if "test_selected" in report:
        t = report["test_selected"]
        L += ["", "## Test set (final pipeline; the test split was viewed twice, see limitations)", "", "| method | precision | recall | F1 | AUC | threshold |", "|---|---|---|---|---|---|"]
        for c, m in report["test_all_candidates"].items():
            L.append(f"| {c}{' (selected)' if c == report['selected'] else ''} | {_fmt(m)} |")
        rb = report["rules_baseline_test"]
        L += ["", f"Rules baseline on test: precision {rb['precision']}, recall {rb['recall']}, F1 {rb['f1']}.", "",
              f"Selected method, 95% bootstrap CI: precision {t['ci95']['precision']}, recall {t['ci95']['recall']}, F1 {t['ci95']['f1']}.", "",
              "Per platform (test):", "", "| platform | n | precision | recall | F1 |", "|---|---|---|---|---|"]
        for p, m in t["by_platform"].items():
            L.append(f"| {p} | {m['n']} | {m['precision']} | {m['recall']} | {m['f1']} |")
        w = t["population_weighted"]
        L += ["", f"Population-weighted (sampling weights): precision {w['precision']}, recall {w['recall']}, F1 {w['f1']}."]
    if "funnel" in report:
        f = report["funnel"]
        L += ["", "## Screening flow", "", f"{f['cleaned_videos']} cleaned videos -> **{f['relevant']} relevant**, {f['irrelevant']} irrelevant "
              f"({f['human_labeled']} labeled by a human, {f['model_labeled']} by the model).",
              f"Relevant by platform: {f['relevant_by_platform']}. Model-labeled videos near the decision threshold or disagreeing with the rules (optional audit list): {f['model_labeled_flagged_for_optional_review']}."]
    L += ["", "## Limitations (to state in the report)", "",
          "- 300 sample videos were pre-annotated by an AI assistant from title/description/hashtags and verified row by row by the author; the AI assistant saw the text of all 300 videos including the test portion, so test results should be read as slightly optimistic.",
          "- Ground-truth labels reflect what the video shows when opened; models only see title, description, hashtags and tags (no transcript), so videos with uninformative captions (mostly TikTok) cannot be judged reliably.",
          "- Rules were tuned on the development split only. The pipeline was run twice: the first full run had a bug in the development cross-validation (features and labels were misaligned), which made its model selection and thresholds invalid. After that run the test metrics had been seen. The bug was fixed, and the decision threshold was changed from F2 to F1 because the workflow has no mandatory manual review (a reason independent of the test numbers); no other change was made. The test split was therefore looked at twice, so test results should be read with that in mind.",
          "- The sample is stratified (not proportional); population-weighted figures use sampling weights."]
    (out_dir / "screening_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")