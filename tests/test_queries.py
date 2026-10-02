import unittest
from pathlib import Path

from vidscreen.queries import load_queries

ROOT = Path(__file__).resolve().parents[1]


class QueryFileTests(unittest.TestCase):
    def test_35_unique_queries(self):
        qs = load_queries(ROOT / "config" / "queries.csv")
        self.assertEqual(len(qs), 35)
        self.assertEqual(len({q.id for q in qs}), 35)
        self.assertEqual(len({q.file_stem for q in qs}), 35)
        self.assertEqual(qs[0].file_stem, "Q001_pediatric_cpr")
        self.assertEqual(qs[-1].language, "ur-roman")

    def test_ids_sequential(self):
        qs = load_queries(ROOT / "config" / "queries.csv")
        self.assertEqual([q.id for q in qs], [f"Q{i:03d}" for i in range(1, 36)])


if __name__ == "__main__":
    unittest.main()
