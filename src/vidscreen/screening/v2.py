"""Revised screening: one combined model vs separate TikTok / YouTube models, text only vs text + features.

Inputs are title, description, hashtags, tags and simple metadata. NOT used: transcripts, views, likes, query information.
Model comparison uses repeated, creator-grouped, stratified cross-validation over all human-labeled videos.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .evaluate import bootstrap_ci, metrics
from .features import NUMERIC, add_features
from .models import EMBEDDING_MODEL, encode_texts
from .pipeline import RULE_TO_BINARY, load_frame

log = logging.getLogger("vidscreen")
SEED = 42
ZONE_LOW, ZONE_HIGH = 0.25, 0.75   # calibrated-probability bands: <=0.25 confident irrelevant, >=0.75 confident relevant, else audit
TIE = 0.01
V1_FILES = ["relevant_videos.csv", "irrelevant_videos.csv", "screening_results.csv", "screening_report.md", "screening_report.json"]


@dataclass(frozen=True)
class Spec:
    rep: str      # tfidf | embed
    feats: bool   # False = text only, True = text + numeric features
    scope: str    # combined | per_platform
    gate: bool    # multiply by the rule gate (a strong topic term must be present)

    @property
    def name(self) -> str:
        return f"{self.rep}|{'text+features' if self.feats else 'text'}|{self.scope}|{'gate' if self.gate else 'nogate'}"

    @property
    def complexity(self) -> int:
        return int(self.rep == "embed") + int(self.feats) + int(self.scope == "per_platform") + int(self.gate)


class Data:
    def __init__(self, df: pd.DataFrame, emb: Optional[np.ndarray]):
        self.texts = df["text"].to_numpy()
        self.num = df[NUMERIC].to_numpy(dtype=float)
        self.platform = df["platform"].to_numpy()
        self.gate = df["topic_hit"].to_numpy(dtype=float)
        self.emb = emb


class Model:
    def __init__(self, rep: str, feats: bool, data: Data, seed: int = SEED):
        self.rep, self.feats, self.data, self.seed = rep, feats, data, seed
        self.const = None

    def _tfidf(self, texts, fit: bool):
        if fit:
            for min_df in (2, 1):
                try:
                    self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True).fit(texts)
                    self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, sublinear_tf=True).fit(texts)
                    break
                except ValueError:
                    continue
        return [self.w.transform(texts), self.c.transform(texts)]

    def _X(self, idx, fit: bool):
        parts = self._tfidf(self.data.texts[idx], fit) if self.rep == "tfidf" else [sparse.csr_matrix(self.data.emb[idx])]
        if self.feats:
            if fit:
                self.sc = StandardScaler().fit(self.data.num[idx])
            parts.append(sparse.csr_matrix(self.sc.transform(self.data.num[idx])))
        return sparse.hstack(parts).tocsr()

    def fit(self, idx, y):
        y = np.asarray(y).astype(int)
        if len(set(y)) < 2:
            self.const = float(y.mean())
            return self
        self.lr = LogisticRegression(C=4.0 if self.rep == "tfidf" else 1.0, class_weight="balanced", max_iter=4000,
                                     random_state=self.seed).fit(self._X(idx, True), y)
        return self

    def predict(self, idx):
        if self.const is not None:
            return np.full(len(idx), self.const)
        return self.lr.predict_proba(self._X(idx, False))[:, 1]


def fit_predict(spec: Spec, data: Data, train_idx, y_train, test_idx) -> np.ndarray:
    train_idx, test_idx, y_train = np.asarray(train_idx), np.asarray(test_idx), np.asarray(y_train)
    if spec.scope == "combined":
        return Model(spec.rep, spec.feats, data).fit(train_idx, y_train).predict(test_idx)
    out = np.zeros(len(test_idx))
    for p in np.unique(data.platform):
        tr, te = data.platform[train_idx] == p, data.platform[test_idx] == p
        if te.any():
            out[te] = Model(spec.rep, spec.feats, data).fit(train_idx[tr], y_train[tr]).predict(test_idx[te])
    return out


def base_specs(use_embeddings: bool) -> list:
    reps = ["tfidf"] + (["embed"] if use_embeddings else [])
    return [Spec(r, f, s, g) for r, f, s, g in product(reps, (False, True), ("combined", "per_platform"), (False, True))]


def grouped_cv(df, data, lab_idx, specs, repeats=3, folds=5, seed=SEED) -> dict:
    """Mean out-of-fold probability per labeled video, over `repeats` creator-grouped stratified CV runs."""
    y = df["y"].to_numpy()[lab_idx].astype(int)
    groups = (df["platform"] + ":" + df["creator_id"].where(df["creator_id"] != "", df["video_uid"])).to_numpy()[lab_idx]
    strat = (df["platform"].to_numpy()[lab_idx] + y.astype(str))
    base = sorted({(s.rep, s.feats, s.scope) for s in specs})
    acc = {b: np.zeros(len(lab_idx)) for b in base}
    cnt = np.zeros(len(lab_idx))
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(np.zeros(len(y)), strat, groups):
            for b in base:
                acc[b][te] += fit_predict(Spec(*b, False), data, lab_idx[tr], y[tr], lab_idx[te])
            cnt[te] += 1
    oof = {}
    for s in specs:
        p = acc[(s.rep, s.feats, s.scope)] / np.maximum(cnt, 1)
        oof[s] = p * data.gate[lab_idx] if s.gate else p
    return oof


def best_f1_threshold(y, p) -> float:
    best, bt = -1.0, 0.5
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        f = metrics(y, p >= t)["f1"]
        if f > best + 1e-12:
            best, bt = f, float(t)
    return bt


def thresholds_for(spec: Spec, y, p, platform) -> dict:
    if spec.scope == "combined":
        t = best_f1_threshold(y, p)
        return {pl: t for pl in np.unique(platform)}
    return {pl: best_f1_threshold(y[platform == pl], p[platform == pl]) for pl in np.unique(platform)}


def apply_thresholds(p, platform, th) -> np.ndarray:
    return np.array([pp >= th[pl] for pp, pl in zip(p, platform)]).astype(int)


def _auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def evaluate_candidates(y, platform, oof: dict) -> pd.DataFrame:
    rows = []
    for s, p in oof.items():
        th = thresholds_for(s, y, p, platform)
        pred = apply_thresholds(p, platform, th)
        m = metrics(y, pred)
        row = {"candidate": s.name, "rep": s.rep, "features": "text+features" if s.feats else "text", "scope": s.scope,
               "gate": s.gate, "complexity": s.complexity, "auc": round(_auc(y, p), 4),
               "pr_auc": round(float(average_precision_score(y, p)), 4), "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 4),
               "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]}
        aucs = []
        for pl in np.unique(platform):
            k = platform == pl
            a = _auc(y[k], p[k])
            aucs.append(a)
            mm = metrics(y[k], pred[k])
            row.update({f"{pl}_auc": round(a, 4), f"{pl}_precision": mm["precision"], f"{pl}_recall": mm["recall"], f"{pl}_f1": mm["f1"]})
        row["macro_auc"] = round(float(np.nanmean(aucs)), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)


def select(table: pd.DataFrame) -> str:
    """Highest macro (per-platform) AUC; candidates within 0.01 of it are tied -> least complex, then higher AUC."""
    top = table["macro_auc"].max()
    tied = table[table["macro_auc"] >= top - TIE]
    return tied.sort_values(["complexity", "macro_auc"], ascending=[True, False]).iloc[0]["candidate"]


def calibration(y, p, bins=5) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
             "mean_score": round(float(p[idx == b].mean()), 3), "observed_relevant_rate": round(float(y[idx == b].mean()), 3)}
            for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.01, 0.99)
    return np.log(p / (1 - p))


def fit_calibrator(p, y) -> LogisticRegression:
    """Platt scaling on out-of-fold scores: turns the raw model score (compressed by class weighting) into a probability."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), np.asarray(y).astype(int))


def apply_calibrator(cal: LogisticRegression, p) -> np.ndarray:
    return cal.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def ece(y, p, bins=5) -> float:
    c = calibration(y, p, bins)
    return round(float(sum(r["n"] * abs(r["mean_score"] - r["observed_relevant_rate"]) for r in c.to_dict("records")) / len(y)), 4)


def ablation_summary(table: pd.DataFrame) -> list:
    """Mean change in macro AUC when one design choice is switched, over all otherwise-identical candidate pairs."""
    cols = ["rep", "features", "scope", "gate"]
    out = []
    for factor, a, b, label in (("features", "text", "text+features", "add duration/length/signal features"),
                                ("gate", False, True, "add rule gate"),
                                ("scope", "combined", "per_platform", "separate TikTok/YouTube models instead of one"),
                                ("rep", "tfidf", "embed", "pretrained embeddings instead of TF-IDF")):
        piv = table.pivot_table(index=[c for c in cols if c != factor], columns=factor, values="macro_auc")
        if a in piv.columns and b in piv.columns:
            d = (piv[b] - piv[a]).dropna()
            out.append({"change": label, "mean_macro_auc_change": round(float(d.mean()), 4), "pairs": int(len(d)), "pairs_improved": int((d > 0).sum())})
    return out


def error_category(r) -> str:
    if r["promo_hit"] and not r["instruct_hit"]:
        return "advert / promotion wording"
    if r["cert_hit"]:
        return "certification / exam / student wording"
    if r["adult_hit"] and not r["child_n"]:
        return "adult marker"
    if not r["topic_hit"]:
        return "no topic term in text"
    if not r["child_n"]:
        return "no child term in text"
    if r["word_count"] < 6:
        return "very short / uninformative text"
    return "other"


def coefficient_tables(spec: Spec, data: Data, lab_idx, y) -> pd.DataFrame:
    groups = [("all", lab_idx, y)] if spec.scope == "combined" else [
        (p, lab_idx[data.platform[lab_idx] == p], y[data.platform[lab_idx] == p]) for p in np.unique(data.platform)]
    rows = []
    for name, idx, yy in groups:
        m = Model(spec.rep, spec.feats, data).fit(idx, yy)
        if m.const is not None:
            continue
        coef = m.lr.coef_[0]
        n_num = len(NUMERIC) if spec.feats else 0
        if spec.rep == "tfidf":
            words = m.w.get_feature_names_out()
            rows += [{"model": name, "kind": "word", "feature": w, "coefficient": float(c)} for w, c in zip(words, coef[:len(words)])]
        if spec.feats:
            rows += [{"model": name, "kind": "numeric", "feature": f, "coefficient": float(c)} for f, c in zip(NUMERIC, coef[len(coef) - n_num:])]
    return pd.DataFrame(rows)


def archive_v1(out_dir: Path) -> None:
    arch = out_dir / "v1_initial"
    for f in V1_FILES:
        src = out_dir / f
        if src.exists() and not (arch / f).exists():
            arch.mkdir(exist_ok=True)
            shutil.move(str(src), str(arch / f))


def run_v2(settings, stage: str = "all", use_embeddings: bool = True, out_dir: Optional[Path] = None,
           encoder: Optional[Callable] = None, repeats: int = 3, folds: int = 5, embedding_model: str = EMBEDDING_MODEL) -> dict:
    out_dir = Path(out_dir) if out_dir else settings.final_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_features(load_frame(settings))
    emb = None
    if use_embeddings:
        emb = np.asarray((encoder or (lambda t: encode_texts(t, settings.cache_dir, embedding_model)))(list(df["text"])))
    data = Data(df, emb)
    lab_idx = np.where(df["y"].notna())[0]
    y = df["y"].to_numpy()[lab_idx].astype(int)
    platform = df["platform"].to_numpy()[lab_idx]

    specs = base_specs(use_embeddings)
    log.info("Creator-grouped CV: %d candidates, %d repeats x %d folds, %d labeled videos", len(specs), repeats, folds, len(lab_idx))
    oof = grouped_cv(df, data, lab_idx, specs, repeats, folds)
    table = evaluate_candidates(y, platform, oof)
    chosen_name = select(table)
    chosen = next(s for s in specs if s.name == chosen_name)
    p_raw = oof[chosen]
    calibrator = fit_calibrator(p_raw, y)
    p_sel = apply_calibrator(calibrator, p_raw)          # monotone, so ranking and decisions are unchanged
    thr = thresholds_for(chosen, y, p_sel, platform)

    prev = float(y.mean())
    rules_pred = df["rule_label"].map(RULE_TO_BINARY).to_numpy()[lab_idx]
    baselines = {
        "all_relevant": metrics(y, np.ones_like(y)),
        "rules_only (potentially->relevant; tuned on the development split, optimistic)": metrics(y, rules_pred),
        "rule gate only (strong topic term present)": metrics(y, data.gate[lab_idx].astype(int)),
    }
    pred_sel = apply_thresholds(p_sel, platform, thr)
    lab = df.iloc[lab_idx].assign(y_true=y, score=p_sel, pred=pred_sel)
    err = lab[lab["y_true"] != lab["pred"]].copy()
    err["error_type"] = np.where(err["pred"] == 1, "false_positive", "false_negative")
    err["category"] = err.apply(error_category, axis=1)
    cal = calibration(y, p_sel)
    cal_raw = calibration(y, p_raw)
    zone_oof = np.where(p_sel >= ZONE_HIGH, "relevant_confident", np.where(p_sel <= ZONE_LOW, "irrelevant_confident", "audit"))
    zone_quality = [{"zone": z, "videos": int((zone_oof == z).sum()), "observed_relevant_rate": round(float(y[zone_oof == z].mean()), 3)}
                    for z in ("irrelevant_confident", "audit", "relevant_confident") if (zone_oof == z).any()]
    report = {
        "stage": stage, "seed": SEED, "repeats": repeats, "folds": folds, "n_labeled": int(len(lab_idx)), "prevalence_relevant": round(prev, 3),
        "per_platform_labeled": {p: {"relevant": int(y[platform == p].sum()), "irrelevant": int((1 - y[platform == p]).sum())} for p in np.unique(platform)},
        "selected": chosen.name, "thresholds": {k: round(v, 2) for k, v in thr.items()},
        "selection_rule": "highest macro (per-platform) AUC over pooled out-of-fold scores; candidates within 0.01 are tied -> least complex",
        "baselines": baselines, "embedding_model": embedding_model if use_embeddings else None,
        "selected_metrics": table[table.candidate == chosen.name].iloc[0].to_dict(),
        "calibration_ece_raw": ece(y, p_raw), "calibration_ece_calibrated": ece(y, p_sel),
        "brier_raw": round(float(brier_score_loss(y, np.clip(p_raw, 0, 1))), 4), "brier_calibrated": round(float(brier_score_loss(y, p_sel)), 4),
        "zone_bands": {"irrelevant_confident": f"<= {ZONE_LOW}", "relevant_confident": f">= {ZONE_HIGH}", "audit": "in between"},
        "zone_quality_oof": zone_quality, "ablation": ablation_summary(table),
        "error_counts": err.groupby(["error_type", "category"]).size().rename("n").reset_index().to_dict("records"),
    }
    table.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat([cal_raw.assign(scale="raw score"), cal.assign(scale="calibrated")]).to_csv(out_dir / "calibration.csv", index=False, encoding="utf-8-sig")
    # interpretable companion (TF-IDF + logistic regression): the selected model may be embedding-based and not directly readable
    c_all = coefficient_tables(Spec("tfidf", False, "combined", False), data, lab_idx, y).assign(model="TF-IDF companion, all videos")
    c_pl = coefficient_tables(Spec("tfidf", False, "per_platform", False), data, lab_idx, y)
    c_pl["model"] = "TF-IDF companion, " + c_pl["model"].astype(str)
    coefs = pd.concat([c_all, c_pl], ignore_index=True)
    coefs.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    err[["video_uid", "platform", "title", "score", "y_true", "pred", "error_type", "category"]].to_csv(out_dir / "cv_errors.csv", index=False, encoding="utf-8-sig")

    if stage == "all":
        _final_screening(df, data, chosen, lab_idx, y, thr, calibrator, report, out_dir)
    _write_report(report, table, cal_raw, cal, coefs, out_dir)
    (out_dir / "screening_report.json").write_text(json.dumps(_clean(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _final_screening(df, data, chosen: Spec, lab_idx, y, thr, calibrator, report, out_dir: Path) -> None:
    archive_v1(out_dir)
    all_idx = np.arange(len(df))
    p = fit_predict(chosen, data, lab_idx, y, all_idx)
    if chosen.gate:
        p = p * data.gate
    p = apply_calibrator(calibrator, p)
    th = np.array([thr[pl] for pl in data.platform])
    human = df["y"].notna().to_numpy()
    model_label = np.where(p >= th, "relevant", "irrelevant")
    out = df.copy()
    out["score"] = np.round(p, 4)
    out["screening_label"] = np.where(human, df["label_final"].fillna(""), model_label)
    out["label_source"] = np.where(human, "human", "model")
    zone = np.where(p >= ZONE_HIGH, "relevant_confident", np.where(p <= ZONE_LOW, "irrelevant_confident", "audit"))
    out["zone"] = np.where(human, "human_label", zone)
    out["model_spec"] = np.where(human, "human_label", chosen.name)
    keep = ["video_uid", "platform", "video_id", "url", "title", "screening_label", "label_source", "zone", "score", "model_spec",
            "rule_label", "rule_reason", "query_ids", "query_subtopics"]
    out[keep].to_csv(out_dir / "screening_results.csv", index=False, encoding="utf-8-sig")
    drop = {"text", "rule_label", "rule_reason", "rule_terms", "topic_gate", "y", "label_final", "split", "sampling_weight", "sample_id", "score",
            "screening_label", "label_source", "zone", "model_spec", "word_count"} | set(NUMERIC)
    base_cols = [c for c in out.columns if c not in drop]
    extra = ["screening_label", "label_source", "zone", "score", "model_spec"]
    rel = out[out["screening_label"] == "relevant"]
    rel[base_cols + extra].to_csv(out_dir / "relevant_videos.csv", index=False, encoding="utf-8-sig")
    rel[rel["zone"].isin(["relevant_confident", "human_label"])][base_cols + extra].to_csv(out_dir / "relevant_confident_videos.csv", index=False, encoding="utf-8-sig")
    out[out["screening_label"] == "irrelevant"][base_cols + extra].to_csv(out_dir / "irrelevant_videos.csv", index=False, encoding="utf-8-sig")
    out[out["zone"] == "audit"][base_cols + extra].to_csv(out_dir / "audit_videos.csv", index=False, encoding="utf-8-sig")

    # secondary consistency check: development-only fit, original 105 test videos (already inspected earlier)
    dev = np.where(df["split"] == "dev")[0]
    test = np.where(df["split"] == "test")[0]
    pt = fit_predict(chosen, data, dev, df["y"].to_numpy()[dev].astype(int), test)
    if chosen.gate:
        pt = pt * data.gate[test]
    pt = apply_calibrator(calibrator, pt)
    pred_t = apply_thresholds(pt, data.platform[test], thr)
    yt = df["y"].to_numpy()[test].astype(int)
    report["secondary_check_original_test"] = {**metrics(yt, pred_t), "ci95_f1": bootstrap_ci(yt, pred_t, "f1", seed=SEED),
                                               "note": "The 105 test videos were inspected in the earlier experiment, so this is a consistency check, not an unseen test"}
    model_rows = out[~human]
    report["funnel"] = {
        "cleaned_videos": int(len(df)), "human_labeled": int(human.sum()), "model_labeled": int((~human).sum()),
        "relevant": int((out["screening_label"] == "relevant").sum()), "irrelevant": int((out["screening_label"] == "irrelevant").sum()),
        "relevant_by_platform": rel["platform"].value_counts().to_dict(),
        "irrelevant_by_platform": out[out["screening_label"] == "irrelevant"]["platform"].value_counts().to_dict(),
        "model_labeled_zones": model_rows["zone"].value_counts().to_dict(),
        "relevant_confident_subset": int(rel["zone"].isin(["relevant_confident", "human_label"]).sum()),
    }


def _write_report(report, table, cal_raw, cal, coefs, out_dir: Path) -> None:
    L = ["# Screening report (revised: score-based, separate vs combined platform models)", "",
         f"Seed {report['seed']}. {report['n_labeled']} human-labeled videos; {report['repeats']} x {report['folds']}-fold CV, stratified by label and platform, "
         "grouped by creator (a creator never appears in both training and validation).", "",
         "Inputs: title, description, hashtags, tags, duration (z-scored within platform), title/description length, hashtag and tag counts, and rule signals "
         "(child term, topic term, promotion, instruction, adult, certification, pet). **Not used:** transcripts, views, likes, query information.", "",
         "## Baselines (on all labeled videos)", "", "| baseline | precision | recall | F1 |", "|---|---|---|---|"]
    for k, m in report["baselines"].items():
        L.append(f"| {k} | {m['precision']} | {m['recall']} | {m['f1']} |")
    cols = ["candidate", "macro_auc", "auc", "pr_auc", "brier", "precision", "recall", "f1", "youtube_auc", "tiktok_auc", "youtube_f1", "tiktok_f1"]
    L += ["", "## Candidate comparison (pooled out-of-fold scores; F1 columns use the F1-optimal threshold per candidate, so they are slightly optimistic)", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table.to_dict("records"):
        L.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    s = report["selected_metrics"]
    L += ["", f"**Selected:** `{report['selected']}` (rule: {report['selection_rule']}). Decision thresholds (calibrated probability): {report['thresholds']}.", "",
          f"Selected model: macro AUC {s['macro_auc']}, precision {s['precision']}, recall {s['recall']}, F1 {s['f1']}.", "",
          "## What the design choices did (mean change in macro AUC over otherwise identical candidate pairs)", "",
          "| change | mean macro-AUC change | pairs | pairs that improved |", "|---|---|---|---|"]
    for r in report["ablation"]:
        L.append(f"| {r['change']} | {r['mean_macro_auc_change']:+.3f} | {r['pairs']} | {r['pairs_improved']} |")
    L += ["", "## Calibration of the selected model (out-of-fold)", "",
          f"Raw score: expected calibration error {report['calibration_ece_raw']}, Brier {report['brier_raw']}. After Platt scaling (monotone, ranking unchanged): "
          f"expected calibration error {report['calibration_ece_calibrated']}, Brier {report['brier_calibrated']} (the calibrator is fitted on the same out-of-fold scores, so this is slightly optimistic).", "",
          "| scale | score bin | videos | mean score | observed relevant rate |", "|---|---|---|---|---|"]
    for name, tab in (("raw", cal_raw), ("calibrated", cal)):
        for r in tab.to_dict("records"):
            L.append(f"| {name} | {r['bin']} | {r['n']} | {r['mean_score']} | {r['observed_relevant_rate']} |")
    L += ["", f"## Zones (calibrated probability bands {report['zone_bands']}), observed on the labeled videos", "",
          "| zone | videos | observed relevant rate |", "|---|---|---|"]
    for r in report["zone_quality_oof"]:
        L.append(f"| {r['zone']} | {r['videos']} | {r['observed_relevant_rate']} |")
    if len(coefs):
        L += ["", "## Patterns (interpretable TF-IDF companion model; largest word coefficients)", ""]
        for name, g in coefs.groupby("model"):
            w = g[g.kind == "word"].sort_values("coefficient")
            if len(w):
                L += [f"**{name}, towards relevant:** " + ", ".join(w.tail(15).iloc[::-1].feature), "",
                      f"**{name}, towards irrelevant:** " + ", ".join(w.head(15).feature), ""]
    if report["error_counts"]:
        L += ["", "## Error categories (out-of-fold, selected model)", "", "| error | category | n |", "|---|---|---|"]
        for r in report["error_counts"]:
            L.append(f"| {r['error_type']} | {r['category']} | {r['n']} |")
    if "secondary_check_original_test" in report:
        t = report["secondary_check_original_test"]
        L += ["", "## Secondary consistency check on the original 105 test videos", "",
              f"Fitted on the 195 development videos only: precision {t['precision']}, recall {t['recall']}, F1 {t['f1']} (95% CI {t['ci95_f1']}). {t['note']}."]
    if "funnel" in report:
        f = report["funnel"]
        L += ["", "## Screening flow", "", f"{f['cleaned_videos']} cleaned videos -> **{f['relevant']} relevant**, {f['irrelevant']} irrelevant "
              f"({f['human_labeled']} human labels kept, {f['model_labeled']} scored by the model).",
              f"Relevant by platform: {f['relevant_by_platform']}. Zones of model-scored videos: {f['model_labeled_zones']}. "
              f"Confident-relevant subset (for sensitivity analysis): {f['relevant_confident_subset']}."]
    L += ["", "## Limitations", "",
          "- Labels were pre-annotated by an AI assistant from text fields and verified by the author; the assistant saw all 300 videos, including the earlier test portion.",
          "- Ground truth reflects what the video shows when opened; models only see text fields and simple metadata.",
          "- The earlier 105-video test split had already been inspected, so model development here uses creator-grouped cross-validation over all 300 labels and the old test split is only a secondary check.",
          "- YouTube has few labeled irrelevant videos, so YouTube metrics have wide uncertainty.",
          "- F1-optimal thresholds are tuned on the same out-of-fold scores, which is slightly optimistic."]
    (out_dir / "screening_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")