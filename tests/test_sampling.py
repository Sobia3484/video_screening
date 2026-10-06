import unittest

import pandas as pd

from vidscreen.screening.sampling import allocate, draw_sample, primary_stratum


def fake_population():
    rows = []
    for platform, sizes in {"youtube": {"cpr": 40, "aed": 6, "choking|multi": 0}, "tiktok": {"cpr": 60, "aed": 30}}.items():
        for sub, n in sizes.items():
            for i in range(n):
                rows.append({"video_uid": f"{platform}:{sub}{i}", "platform": platform, "query_subtopics": sub})
    for i in range(10):
        rows.append({"video_uid": f"youtube:m{i}", "platform": "youtube", "query_subtopics": "aed|cpr"})
    return pd.DataFrame(rows)


class SamplingTests(unittest.TestCase):
    def test_strata_labels(self):
        self.assertEqual(primary_stratum("cpr"), "cpr")
        self.assertEqual(primary_stratum("aed|cpr"), "multi")
        self.assertEqual(primary_stratum(""), "unknown")

    def test_allocate_caps_and_redistributes(self):
        a = allocate({"a": 3, "b": 100, "c": 100}, 30)
        self.assertEqual(a, {"a": 3, "b": 14, "c": 13})
        self.assertEqual(sum(allocate({"a": 2, "b": 3}, 50).values()), 5)  # cannot exceed what exists

    def test_sample_properties(self):
        pop = fake_population()
        s1 = draw_sample(pop, per_platform=30, seed=7)
        s2 = draw_sample(pop, per_platform=30, seed=7)
        self.assertEqual(list(s1.video_uid), list(s2.video_uid))                   # reproducible
        self.assertEqual(s1.groupby("platform").size().to_dict(), {"tiktok": 30, "youtube": 30})
        self.assertTrue(s1.video_uid.is_unique)
        self.assertEqual(set(s1.split), {"dev", "test"})
        self.assertEqual(list(s1.sample_id[:2]), ["S001", "S002"])
        # weights re-create stratum sizes
        recon = s1.groupby(["platform", "stratum"]).sampling_weight.sum().round().astype(int)
        self.assertEqual(recon[("tiktok", "cpr")], 60)


if __name__ == "__main__":
    unittest.main()
