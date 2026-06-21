import csv
import logging
from typing import List

from models.candidate import CandidateFeatures

logger = logging.getLogger(__name__)


def generate_csv(
    features: List[CandidateFeatures],
    filepath: str,
) -> None:
    expected_fields = ["candidate_id", "rank", "score", "reasoning"]

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=expected_fields)
        writer.writeheader()

        written_ids = set()

        for feat in features:
            if feat.candidate_id in written_ids:
                logger.warning(f"Duplicate candidate_id: {feat.candidate_id}")
                continue

            writer.writerow({
                "candidate_id": feat.candidate_id,
                "rank": feat.rank,
                "score": f"{feat.final_score:.4f}",
                "reasoning": feat.reasoning,
            })
            written_ids.add(feat.candidate_id)

    logger.info(f"Generated CSV with {len(features)} rows at {filepath}")


def validate_csv(filepath: str) -> bool:
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        rows = list(reader)
        if len(rows) != 100:
            logger.error(f"Expected 100 rows, got {len(rows)}")
            return False

        ids = set()
        ranks = set()
        prev_score = float("inf")

        for row in rows:
            cid = row.get("candidate_id", "")
            rank = int(row.get("rank", 0))
            score = float(row.get("score", 0))

            if cid in ids:
                logger.error(f"Duplicate candidate_id: {cid}")
                return False
            ids.add(cid)

            if rank in ranks:
                logger.error(f"Duplicate rank: {rank}")
                return False
            ranks.add(rank)

            if score > prev_score + 1e-6:
                logger.error(f"Scores not monotonically decreasing: "
                             f"rank {rank} score {score} > prev {prev_score}")
                return False
            prev_score = score

        if ranks != set(range(1, 101)):
            logger.error(f"Ranks not 1-100: {sorted(ranks)}")
            return False

        logger.info("CSV validation passed")
        return True
