#!/usr/bin/env python
"""Step 3: relevance screening with pretrained models.

    python scripts/train_screening_model.py                 # A) compare models by cross-validation on DEV labels only
    python scripts/train_screening_model.py --final-test    # B) evaluate the chosen model ONCE on the TEST labels
    python scripts/train_screening_model.py --predict-all   # C) score all cleaned videos (relevant / irrelevant / uncertain)

Models: tfidf_lr (baseline), minilm_lr and e5_lr (pretrained multilingual transformer embeddings + logistic regression).
The test labels are never used in A. B can run only once (lock file) so the test result stays honest.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

from vidscreen.config import load_settings  # noqa: E402
from vidscreen.screening.evaluate import binary_metrics, bootstrap_ci, choose_thresholds, three_way_report  # noqa: E402
from vidscreen.screening.features import build_text  # noqa: E402
from vidscreen.screening.models import CANDIDATES, Embedder, cv_oof, embedding_classifier, tfidf_pipeline  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def features_for(name, texts, cache_dir):
    kind, emb_name, prefix = CANDIDATES[name]
    if kind == "tfidf":
        return tfidf_pipeline(), np.asarray(list(texts), dtype=object)
    return embedding_classifier(), Embedder(emb_name, cache_dir, prefix).encode(list(texts))


def main() -> int:
    s = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tfidf_lr", "minilm_lr"], choices=list(CANDIDATES))
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--predict-all", action="store_true")
    ap.add_argument("--force-retest", action="store_true", help="Allow a second look at the test set (must be disclosed)")
    ap.add_argument("--skip-test-check", action="store_true")
    ap.add_argument("--master", default=str(s.intermediate_dir / "master_cleaned.csv"))
    ap.add_argument("--labels", default=str(s.root / "data" / "labeled" / "final_labels_300.csv"))
    ap.add_argument("--out-dir", default=str(s.root / "data" / "screening"))
    ap.add_argument("--scores-out", default=str(s.intermediate_dir / "screening_scores.csv"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = s.cache_dir / "embeddings"
    master = pd.read_csv(args.master, dtype={"video_id": str}, encoding="utf-8-sig")
    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    df = master.merge(labels[["video_uid", "label_final", "split"]], on="video_uid", how="left")
    df["text"] = build_text(df)
    df["y"] = (df["label_final"] == "relevant").astype(int)
    dev, test = (df["split"] == "dev").to_numpy(), (df["split"] == "test").to_numpy()
    print(f"Videos: {len(df)} | labeled dev: {dev.sum()} | labeled test: {test.sum()} (test not touched unless --final-test)")

    choice_path, lock_path = out / "model_choice.json", out / "TEST_USED.json"

    # ------------------------------------------------------------------ B) final test (once)
    if args.final_test:
        if not choice_path.exists():
            print("Run step A first (no model chosen yet).")
            return 1
        if lock_path.exists() and not args.force_retest:
            print(f"The test set was already used ({lock_path.read_text()[:120]}...). Refusing a second look. "
                  "Use --force-retest only if you will disclose it in the report.")
            return 1
        choice = json.loads(choice_path.read_text())
        est, X = features_for(choice["model"], df["text"], cache_dir)
        fitted = clone(est).fit(X[dev], df.loc[dev, "y"].to_numpy())
        prob = fitted.predict_proba(X[test])[:, 1]
        yt = df.loc[test, "y"].to_numpy()
        res = {"model": choice["model"], "run_utc": now(), "n_test": int(test.sum()),
               "at_0.5": binary_metrics(yt, prob), "ci95_at_0.5": bootstrap_ci(yt, prob),
               "three_way": three_way_report(yt, prob, choice["t_low"], choice["t_high"]), "by_platform": {}}
        for plat in ("youtube", "tiktok"):
            m = (df.loc[test, "platform"] == plat).to_numpy()
            res["by_platform"][plat] = binary_metrics(yt[m], prob[m])
        (out / "test_results.json").write_text(json.dumps(res, indent=2))
        pd.DataFrame({"video_uid": df.loc[test, "video_uid"], "platform": df.loc[test, "platform"], "title": df.loc[test, "title"],
                      "label_final": df.loc[test, "label_final"], "prob_relevant": prob.round(4)}).to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
        lock_path.write_text(json.dumps({"used_utc": now(), "model": choice["model"]}))
        print(json.dumps({k: res[k] for k in ("model", "at_0.5", "ci95_at_0.5", "three_way", "by_platform")}, indent=2))
        return 0

    # ------------------------------------------------------------------ C) score everything
    if args.predict_all:
        if not choice_path.exists():
            print("Run step A first.")
            return 1
        if not lock_path.exists() and not args.skip_test_check:
            print("Run step B (--final-test) first so the chosen model is evaluated before it is applied.")
            return 1
        choice = json.loads(choice_path.read_text())
        est, X = features_for(choice["model"], df["text"], cache_dir)
        labeled = (dev | test)
        fitted = clone(est).fit(X[labeled], df.loc[labeled, "y"].to_numpy())
        prob = fitted.predict_proba(X)[:, 1]
        model_label = np.where(prob >= choice["t_high"], "relevant", np.where(prob <= choice["t_low"], "irrelevant", "uncertain"))
        scores = pd.DataFrame({"video_uid": df["video_uid"], "platform": df["platform"], "title": df["title"], "url": df["url"],
                               "prob_relevant": prob.round(4), "model_label": model_label,
                               "human_label": df["label_final"].fillna(""), })
        scores["screening_label"] = np.where(scores["human_label"] != "", scores["human_label"], scores["model_label"])
        scores["label_source"] = np.where(scores["human_label"] != "", "human", "model")
        Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.scores_out, index=False, encoding="utf-8-sig")
        print(f"Wrote {args.scores_out}")
        print(pd.crosstab(scores["screening_label"], scores["platform"], margins=True).to_string())
        return 0

    # ------------------------------------------------------------------ A) model comparison on DEV only
    d = df[dev].reset_index(drop=True)
    y = d["y"].to_numpy()
    strata = (y * 10 + (d["platform"] == "tiktok").astype(int)).to_numpy()
    rows, oof_all = [], {}
    for name in args.models:
        try:
            est, X_all = features_for(name, df["text"][dev], cache_dir)
        except Exception as exc:  # no internet / library missing: keep going with the other models
            print(f"[skip] {name}: {exc}")
            continue
        oof = cv_oof(est, X_all, y, strata)
        oof_all[name] = oof
        m = binary_metrics(y, oof)
        for plat in ("youtube", "tiktok"):
            mp = (d["platform"] == plat).to_numpy()
            m[f"f1_{plat}"] = binary_metrics(y[mp], oof[mp])["f1"]
        rows.append({"model": name, **m})
        print(f"{name:10s} F1={m['f1']}  precision={m['precision']}  recall={m['recall']}  AUC={m['auc']}  "
              f"(YouTube F1 {m['f1_youtube']}, TikTok F1 {m['f1_tiktok']})")
    if not rows:
        print("No model could run.")
        return 1
    res = pd.DataFrame(rows)
    best = rows[0]
    for r in rows[1:]:
        if r["f1"] > best["f1"] + 0.01:  # a more complex model must win clearly
            best = r
    t_low, t_high = choose_thresholds(y, oof_all[best["model"]])
    res.to_csv(out / "cv_results.csv", index=False)
    oof_df = d[["video_uid", "platform", "title", "label_final"]].copy()
    for name, oof in oof_all.items():
        oof_df[f"prob_{name}"] = oof.round(4)
    oof_df.to_csv(out / "cv_oof_predictions.csv", index=False, encoding="utf-8-sig")
    best_oof = oof_all[best["model"]]
    errors = oof_df[(best_oof >= 0.5).astype(int) != y].assign(prob=best_oof[(best_oof >= 0.5).astype(int) != y].round(3))
    errors.to_csv(out / "cv_errors_best_model.csv", index=False, encoding="utf-8-sig")
    choice = {"model": best["model"], "t_low": t_low, "t_high": t_high, "cv_f1": best["f1"], "cv_precision": best["precision"],
              "cv_recall": best["recall"], "decided_utc": now(), "models_compared": [r["model"] for r in rows],
              "rule": "highest cross-validated F1 on dev; a more complex model must beat the simpler one by > 0.01"}
    choice_path.write_text(json.dumps(choice, indent=2))
    print(f"\nChosen model: {best['model']}  | thresholds: relevant >= {t_high}, irrelevant <= {t_low}, else uncertain")
    print(f"Saved results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
