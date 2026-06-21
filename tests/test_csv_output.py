import csv
import os
import tempfile
import unittest

from models.candidate import CandidateFeatures
from output.csv_generator import generate_csv, validate_csv


class TestCSVGenerator(unittest.TestCase):
    def setUp(self):
        self.features = []
        for i in range(100):
            feat = CandidateFeatures(
                candidate_id=f"CAND_{i + 1:07d}",
                jd_fit=float(100 - i),
                evidence=float(90 - i),
                credibility=float(80 - i),
                ranking_score=float(100 - i),
                review_score=float(90 - i),
                final_score=float(100 - i),
                rank=i + 1,
                reasoning=f"Senior ML Engineer with {100 - i:.1f} yrs; {100 - i} AI core skills; response rate 0.80.",
            )
            self.features.append(feat)

    def test_generate_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            fname = f.name

        try:
            generate_csv(self.features, fname)

            with open(fname, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 100)
            self.assertEqual(
                list(rows[0].keys()),
                ["candidate_id", "rank", "score", "reasoning"],
            )
        finally:
            os.unlink(fname)

    def test_validate_csv_valid(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            fname = f.name

        try:
            generate_csv(self.features, fname)
            self.assertTrue(validate_csv(fname))
        finally:
            os.unlink(fname)

    def test_validate_csv_invalid_row_count(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            writer.writerow(["CAND_0001", "1", "100.0", "Test"])
            fname = f.name

        try:
            self.assertFalse(validate_csv(fname))
        finally:
            os.unlink(fname)

    def test_monotonically_decreasing_scores(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            fname = f.name

        try:
            generate_csv(self.features, fname)
            with open(fname, "r") as f:
                reader = csv.DictReader(f)
                scores = [float(row["score"]) for row in reader]
            for i in range(len(scores) - 1):
                self.assertGreaterEqual(scores[i], scores[i + 1])
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    unittest.main()
