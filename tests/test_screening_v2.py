import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from vidscreen.config import load_settings
from vidscreen.screening.features import NUMERIC, add_features
from vidscreen.screening.v2 import Spec, ablation_summary, apply_calibrator, base_specs, fit_calibrator, run_v2, select
from vidscreen.screening.text import build_text

ROOT = Path(__file__).resolve().parents[1]


def fake_encoder(texts):
    out = np.zeros((len(texts), 64))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, sum(map(ord, w)) % 64] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


def make_world(tmp: Path, n=240):
    rng = np.random.default_rng(1)
    rel = ["Infant CPR how to step {}", "Baby choking first aid {}", "Child AED demonstration {}", "Neonatal resuscitation lecture {}"]
    irr = ["Funny nurse day {}", "Adult CPR certification {}", "Baby sleep tips {}", "Cooking pasta {}"]
    rows, labels = [], []
    for i in range(n):
        is_rel = bool(rng.random() < 0.6)
        plat = "youtube" if i % 2 else "tiktok"
        uid = f"{plat}:v{i}"
        rows.append({"video_uid": uid, "platform": plat, "video_id": f"v{i}", "url": f"https://x/{i}", "creator_id": f"c{i % 40}",
                     "title": (rel if is_rel else irr)[int(rng.integers(0, 4))].format(i), "description": "d" if plat == "youtube" else "",
                     "hashtags": "cpr|fyp" if plat == "tiktok" else "", "tags": "", "duration_seconds": float(rng.integers(20, 600)),
                     "query_ids": "Q001", "query_subtopics": "cpr"})
        if i % 3 != 0:
            labels.append({"video_uid": uid, "label_final": "relevant" if is_rel else "irrelevant",
                           "split": "test" if rng.random() < 0.3 else "dev", "sampling_weight": 2.0, "sample_id": f"S{i:03d}"})
    (tmp / "data" / "labeled").mkdir(parents=True)
    (tmp / "int").mkdir()
    (tmp / "final").mkdir()
    pd.DataFrame(rows).to_csv(tmp / "int" / "master_cleaned.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(labels).to_csv(tmp / "data" / "labeled" / "final_labels.csv", index=False, encoding="utf-8-sig")
    return len(labels)


def settings_for(tmp: Path):
    return replace(load_settings(root=ROOT), root=tmp, intermediate_dir=tmp / "int", final_dir=tmp / "final", cache_dir=tmp / "cache")


class FeatureTests(unittest.TestCase):
    def test_duration_is_normalised_within_platform(self):
        df = pd.DataFrame({"platform": ["tiktok"] * 3 + ["youtube"] * 3, "duration_seconds": [10, 20, 40, 600, 1200, 2400],
                           "title": ["Infant CPR"] * 6, "description": [""] * 6, "hashtags": [""] * 6, "tags": [""] * 6})
        df["text"] = df.apply(build_text, axis=1)
        d = add_features(df)
        self.assertAlmostEqual(float(d[d.platform == "tiktok"].duration_z.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(d[d.platform == "youtube"].duration_z.mean()), 0.0, places=6)
        self.assertTrue(set(NUMERIC) <= set(d.columns))
        self.assertTrue((d.child_n > 0).all() and (d.topic_hit == 1).all())


class SelectionTests(unittest.TestCase):
    def test_tie_goes_to_the_simplest(self):
        t = pd.DataFrame([{"candidate": "a", "macro_auc": 0.84, "complexity": 3}, {"candidate": "b", "macro_auc": 0.835, "complexity": 1},
                          {"candidate": "c", "macro_auc": 0.80, "complexity": 0}])
        self.assertEqual(select(t), "b")

    def test_specs(self):
        self.assertEqual(len(base_specs(False)), 8)
        self.assertEqual(len(base_specs(True)), 16)
        self.assertEqual(Spec("tfidf", False, "combined", False).complexity, 0)


class CalibrationTests(unittest.TestCase):
    def test_platt_scaling_is_monotone_and_fixes_compressed_scores(self):
        rng = np.random.default_rng(0)
        y = (rng.random(400) < 0.7).astype(int)
        raw = np.where(y == 1, rng.uniform(0.45, 0.75, 400), rng.uniform(0.05, 0.45, 400))   # compressed, under-confident scores
        cal = fit_calibrator(raw, y)
        p = apply_calibrator(cal, raw)
        order = np.argsort(raw)
        self.assertTrue(np.all(np.diff(p[order]) >= -1e-12))
        self.assertGreater(p[y == 1].mean(), raw[y == 1].mean())

    def test_ablation_summary(self):
        t = pd.DataFrame([
            {"rep": "tfidf", "features": "text", "scope": "combined", "gate": False, "macro_auc": 0.80},
            {"rep": "tfidf", "features": "text+features", "scope": "combined", "gate": False, "macro_auc": 0.70},
            {"rep": "tfidf", "features": "text", "scope": "combined", "gate": True, "macro_auc": 0.84},
            {"rep": "tfidf", "features": "text+features", "scope": "combined", "gate": True, "macro_auc": 0.74}])
        r = {x["change"]: x for x in ablation_summary(t)}
        self.assertAlmostEqual(r["add duration/length/signal features"]["mean_macro_auc_change"], -0.10, places=3)
        self.assertAlmostEqual(r["add rule gate"]["mean_macro_auc_change"], 0.04, places=3)


class RunTests(unittest.TestCase):
    def test_cv_stage_writes_comparison_only(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            make_world(tmp)
            rep = run_v2(settings_for(tmp), "cv", use_embeddings=False, repeats=1, folds=3)
            self.assertTrue((tmp / "final" / "model_comparison.csv").exists())
            self.assertFalse((tmp / "final" / "relevant_videos.csv").exists())
            self.assertIn("|", rep["selected"])
            self.assertGreater(rep["selected_metrics"]["macro_auc"], 0.9)

    def test_all_stage_with_embeddings_keeps_human_labels_and_archives_v1(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            n_lab = make_world(tmp)
            (tmp / "final" / "relevant_videos.csv").write_text("old", encoding="utf-8")
            rep = run_v2(settings_for(tmp), "all", use_embeddings=True, encoder=fake_encoder, repeats=1, folds=3)
            self.assertEqual((tmp / "final" / "v1_initial" / "relevant_videos.csv").read_text(encoding="utf-8"), "old")
            res = pd.read_csv(tmp / "final" / "screening_results.csv")
            self.assertEqual(len(res), 240)
            self.assertEqual(int((res.label_source == "human").sum()), n_lab)
            f = rep["funnel"]
            self.assertEqual(f["relevant"] + f["irrelevant"], 240)
            self.assertEqual(len(pd.read_csv(tmp / "final" / "relevant_videos.csv")), f["relevant"])
            self.assertTrue(set(res.zone) <= {"human_label", "audit", "relevant_confident", "irrelevant_confident"})
            self.assertTrue((tmp / "final" / "audit_videos.csv").exists())
            self.assertIn("secondary_check_original_test", rep)
            self.assertTrue(((res.score >= 0) & (res.score <= 1)).all())
            self.assertIn("zone_quality_oof", rep)
            self.assertTrue((tmp / "final" / "screening_report.md").read_text(encoding="utf-8").count("## Zones") == 1)
            self.assertIn("embed", " ".join(pd.read_csv(tmp / "final" / "model_comparison.csv").candidate))


if __name__ == "__main__":
    unittest.main()