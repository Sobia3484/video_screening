import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from vidscreen.config import load_settings
from vidscreen.screening.evaluate import best_threshold, bootstrap_ci, metrics
from vidscreen.screening.pipeline import run_screening
from vidscreen.screening.rules import apply_rules
from vidscreen.screening.text import fold

ROOT = Path(__file__).resolve().parents[1]


class RuleTests(unittest.TestCase):
    def label(self, title, desc="", hashtags="", tags=""):
        return apply_rules(title, desc, hashtags, tags).label

    def test_core_cases(self):
        self.assertEqual(self.label("Infant CPR step by step"), "relevant")
        self.assertEqual(self.label("Heimlich para niños"), "relevant")
        self.assertEqual(self.label("Reanimación cardiopulmonar pediátrica"), "relevant")
        self.assertEqual(self.label("Bayi tersedak, ini caranya"), "relevant")
        self.assertEqual(self.label("Adult CPR training"), "irrelevant")
        self.assertEqual(self.label("Dog CPR"), "irrelevant")
        self.assertEqual(self.label("Funny nurse day"), "irrelevant")
        self.assertEqual(self.label("CPR basics"), "potentially_relevant")
        self.assertEqual(self.label("Baby CPR class! Enroll now, limited spots left"), "potentially_relevant")

    def test_compound_hashtags_and_stuffed_animal(self):
        self.assertEqual(self.label("Keep this close", hashtags="toddlersafety|antichoking"), "relevant")
        self.assertEqual(self.label("Check, call, care: 30 compressions 2 breaths. Practice on a stuffed animal or doll"), "potentially_relevant")

    def test_fold_keeps_non_latin(self):
        self.assertEqual(fold("Reanimación"), "reanimacion")
        self.assertEqual(fold("Ребёнок"), "ребёнок")


class MetricTests(unittest.TestCase):
    def test_metrics_and_threshold(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        m = metrics(y, np.array([1, 1, 0, 1, 0, 0]))
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(m["precision"], 0.6667, places=3)
        t = best_threshold(y, np.array([0.9, 0.6, 0.35, 0.3, 0.2, 0.1]))
        self.assertTrue(0.2 <= t <= 0.8)
        lo, hi = bootstrap_ci(y, np.array([1, 1, 0, 1, 0, 0]), "f1", n_boot=200)
        self.assertLessEqual(lo, hi)


def fake_encoder(texts):
    """Hashing bag-of-words 'embedding' so the plumbing can be tested without downloading a model."""
    out = np.zeros((len(texts), 64))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, hash(w) % 64] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


class PipelineTests(unittest.TestCase):
    def make_world(self, tmp: Path):
        """150 videos; 2/3 are human-labeled with a RANDOM dev/test split, unlabeled rows are interleaved."""
        rng = np.random.default_rng(0)
        rows, labels = [], []
        rel = ["Infant CPR how to step {}", "Baby choking first aid {}", "Child AED demonstration {}", "Neonatal resuscitation lecture {}"]
        irr = ["Funny nurse day {}", "Adult CPR certification {}", "Baby sleep tips {}", "Cooking pasta {}"]
        for i in range(150):
            is_rel = bool(rng.random() < 0.5)
            title = (rel if is_rel else irr)[int(rng.integers(0, 4))].format(i)
            uid = f"{'youtube' if i % 2 else 'tiktok'}:v{i}"
            rows.append({"video_uid": uid, "platform": uid.split(":")[0], "video_id": f"v{i}", "url": f"https://x/{i}", "title": title,
                         "description": "", "hashtags": "", "tags": "", "query_ids": "Q001", "query_subtopics": "cpr", "duration_seconds": 60})
            if i % 3 != 0:
                labels.append({"video_uid": uid, "label_final": "relevant" if is_rel else "irrelevant",
                               "split": "test" if rng.random() < 0.3 else "dev", "sampling_weight": 2.0, "sample_id": f"S{i:03d}"})
        self.n_labeled = len(labels)
        (tmp / "data" / "labeled").mkdir(parents=True)
        (tmp / "int").mkdir()
        pd.DataFrame(rows).to_csv(tmp / "int" / "master_cleaned.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(labels).to_csv(tmp / "data" / "labeled" / "final_labels.csv", index=False, encoding="utf-8-sig")

    def test_full_run(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self.make_world(tmp)
            s = replace(load_settings(root=ROOT), root=tmp, intermediate_dir=tmp / "int", final_dir=tmp / "final", cache_dir=tmp / "cache")
            rep = run_screening(s, "all", use_embeddings=True, encoder=fake_encoder)
            self.assertIn(rep["selected"], {"tfidf", "hybrid_tfidf", "embed", "hybrid_embed"})
            res = pd.read_csv(tmp / "final" / "screening_results.csv")
            self.assertEqual(len(res), 150)
            human = res[res.label_source == "human"]
            self.assertEqual(len(human), self.n_labeled)                       # human labels are kept as they are
            f = rep["funnel"]
            self.assertEqual(f["relevant"] + f["irrelevant"], 150)
            self.assertEqual(len(pd.read_csv(tmp / "final" / "relevant_videos.csv")), f["relevant"])
            self.assertTrue((tmp / "final" / "screening_report.md").exists())
            self.assertIn("test_selected", rep)

    def test_cv_pairs_texts_with_the_right_labels(self):
        # Regression: CV once used the first rows of the full matrix instead of the development rows (AUC ~0.5).
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self.make_world(tmp)
            lab = pd.read_csv(tmp / "data" / "labeled" / "final_labels.csv")
            lab = lab.sample(frac=1, random_state=1)                      # dev rows are NOT the first rows of the file
            lab.to_csv(tmp / "data" / "labeled" / "final_labels.csv", index=False)
            master = pd.read_csv(tmp / "int" / "master_cleaned.csv").sample(frac=1, random_state=2)
            master.to_csv(tmp / "int" / "master_cleaned.csv", index=False)
            s = replace(load_settings(root=ROOT), root=tmp, intermediate_dir=tmp / "int", final_dir=tmp / "final", cache_dir=tmp / "cache")
            rep = run_screening(s, "dev", use_embeddings=False)
            self.assertGreater(rep["dev_cv"]["threshold_0.5"]["tfidf"]["auc"], 0.9)

    def test_dev_cv_features_and_labels_are_aligned(self):
        """Regression: cross-validation must use the development rows only (it once used the first rows of the whole table)."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self.make_world(tmp)
            s = replace(load_settings(root=ROOT), root=tmp, intermediate_dir=tmp / "int", final_dir=tmp / "final", cache_dir=tmp / "cache")
            rep = run_screening(s, "dev", use_embeddings=False)
            self.assertGreater(rep["dev_cv"]["threshold_0.5"]["tfidf"]["auc"], 0.9)

    def test_dev_stage_never_touches_test(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self.make_world(tmp)
            s = replace(load_settings(root=ROOT), root=tmp, intermediate_dir=tmp / "int", final_dir=tmp / "final", cache_dir=tmp / "cache")
            rep = run_screening(s, "dev", use_embeddings=False)
            self.assertNotIn("test_selected", rep)
            self.assertFalse((tmp / "final" / "relevant_videos.csv").exists())


if __name__ == "__main__":
    unittest.main()