import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from vidscreen.config import load_settings
from vidscreen.eda.features import build_features, detect_language
from vidscreen.eda.runner import run_eda
from vidscreen.eda.stats import cliffs_delta, compare_two, describe, gini, magnitude

ROOT = Path(__file__).resolve().parents[1]


def frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    titles = ["Infant CPR step by step", "Adult and child CPR training", "Newborn resuscitation lecture", "Heimlich for babies and CPR", "How to use an AED on a child", "Baby choking first aid"]
    rows = []
    for i in range(n):
        plat = "youtube" if i % 2 else "tiktok"
        views = int(10 ** rng.uniform(1, 6))
        rows.append({
            "video_uid": f"{plat}:v{i}", "platform": plat, "video_id": f"v{i}", "url": f"https://x/{i}", "title": titles[i % len(titles)] + f" {i}",
            "description": "A longer description about the topic" if plat == "youtube" else "", "creator_name": f"c{i % 30}", "creator_id": f"c{i % 30}",
            "creator_followers": float(10 ** rng.uniform(2, 6)), "duration_seconds": float(rng.integers(10, 900) if plat == "youtube" else rng.integers(5, 120)),
            "upload_date": f"202{rng.integers(0, 6)}-0{rng.integers(1, 9)}-1{rng.integers(0, 9)}", "views": views, "likes": int(views * rng.uniform(0.005, 0.05)),
            "comments": int(views * rng.uniform(0, 0.002)), "shares": np.nan if plat == "youtube" else int(views * 0.002), "saves": np.nan if plat == "youtube" else int(views * 0.004),
            "hashtags": "cpr|firstaid" if i % 3 else "", "tags": "cpr|infant" if plat == "youtube" else "", "language_clean": "en", "country_clean": "Unknown",
            "query_count": int(rng.integers(1, 5)), "has_transcript": "False", "label_source": "model" if i % 4 else "human",
            "zone": "relevant_confident" if i % 4 else "human_label", "retrieved_at_utc": "2026-10-01T17:00:00Z"})
    return pd.DataFrame(rows)


class StatsTests(unittest.TestCase):
    def test_cliffs_delta(self):
        self.assertAlmostEqual(cliffs_delta([3, 4, 5], [1, 2, 3]), 8 / 9, places=6)
        self.assertAlmostEqual(cliffs_delta([1, 2, 3], [3, 4, 5]), -8 / 9, places=6)
        self.assertEqual(cliffs_delta([1, 2], [1, 2]), 0.0)
        self.assertEqual(magnitude(0.1), "negligible")
        self.assertEqual(magnitude(-0.5), "large")

    def test_compare_two_detects_a_clear_difference(self):
        rng = np.random.default_rng(1)
        c = compare_two(rng.normal(10, 1, 200), rng.normal(5, 1, 200), n_boot=200)
        self.assertLess(c["p_value"], 0.001)
        self.assertGreater(c["cliffs_delta"], 0.9)
        self.assertLessEqual(c["median_diff_ci_low"], c["median_diff"])
        self.assertLessEqual(c["median_diff"], c["median_diff_ci_high"])
        same = compare_two(rng.normal(0, 1, 300), rng.normal(0, 1, 300), n_boot=200)
        self.assertLess(abs(same["cliffs_delta"]), 0.15)

    def test_describe_and_gini(self):
        self.assertEqual(describe([1, 2, 3, np.nan])["n"], 3)
        self.assertAlmostEqual(gini([1, 1, 1, 1]), 0.0, places=6)
        self.assertGreater(gini([0, 0, 0, 100]), 0.7)


class FeatureTests(unittest.TestCase):
    def test_groups_rates_and_labels(self):
        f = build_features(frame())
        self.assertTrue({"duration_group", "like_rate", "age_group", "subtopic", "language_detected", "follower_group", "age_days"} <= set(f.columns))
        low = f[f["views"] < 100]
        self.assertTrue(low["like_rate"].isna().all())               # rates need at least 100 views
        row = lambda t: f[f.title.str.startswith(t)].iloc[0]
        self.assertEqual(row("Infant CPR").age_group, "infant")
        self.assertEqual(row("Adult and child").age_group, "all ages (adult + child/infant)")
        self.assertEqual(row("Newborn").age_group, "newborn / neonatal")
        self.assertEqual(row("Heimlich for babies").subtopic, "CPR and choking")
        self.assertEqual(row("How to use an AED").subtopic, "AED")
        self.assertEqual(row("Newborn resuscitation").subtopic, "neonatal resuscitation")
        self.assertTrue((f.age_days >= 0).all())

    def test_duration_groups(self):
        d = frame(8)
        d["duration_seconds"] = [30, 60, 299, 300, 599, 600, 1199, 1200]
        g = build_features(d)["duration_group"].astype(str).tolist()
        self.assertEqual(g, ["under 1 min", "1-5 min", "1-5 min", "5-10 min", "5-10 min", "10-20 min", "10-20 min", "20+ min"])

    def test_language_heuristic(self):
        self.assertEqual(detect_language("Hi")[0], "und")
        self.assertEqual(detect_language("How to do infant CPR and what to do when the baby is choking")[0], "en")
        self.assertEqual(detect_language("Как сделать искусственное дыхание ребенку при остановке сердца")[0], "ru")
        self.assertEqual(detect_language("बच्चे को सीपीआर कैसे दें यह वीडियो देखें")[0], "hi")


class RunTests(unittest.TestCase):
    def test_end_to_end_main_only(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "final").mkdir()
            frame().to_csv(tmp / "final" / "relevant_videos.csv", index=False, encoding="utf-8-sig")
            s = replace(load_settings(root=ROOT), root=tmp, final_dir=tmp / "final")
            done = run_eda(s, ("main", "confident"), out_root=tmp / "reports")     # confident file is missing: skipped, not an error
            self.assertEqual(set(done), {"main"})
            rep = (tmp / "reports" / "main" / "EDA_report.md").read_text(encoding="utf-8")
            for title in ("0. Data quality", "1. Derived features", "2. Dataset overview", "3. Video duration", "4. Views", "5. Engagement"):
                self.assertIn(title, rep)
            self.assertGreaterEqual(len(list((tmp / "reports" / "main" / "figures").glob("*.png"))), 10)
            self.assertTrue((tmp / "final" / "relevant_features.csv").exists())
            self.assertIn("duplicate video ids: 0", rep)

    def test_one_section_only(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "final").mkdir()
            frame().to_csv(tmp / "final" / "relevant_videos.csv", index=False, encoding="utf-8-sig")
            s = replace(load_settings(root=ROOT), root=tmp, final_dir=tmp / "final")
            run_eda(s, ("main",), ["duration"], out_root=tmp / "reports")
            rep = (tmp / "reports" / "main" / "EDA_report.md").read_text(encoding="utf-8")
            self.assertIn("3. Video duration", rep)
            self.assertNotIn("4. Views", rep)


if __name__ == "__main__":
    unittest.main()
