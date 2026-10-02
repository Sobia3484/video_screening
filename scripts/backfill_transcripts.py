#!/usr/bin/env python
"""Fill the YouTube transcripts that were blocked during extraction.

    python scripts/backfill_transcripts.py

Safe to stop and re-run at any time: finished transcripts are cached, only missing ones are retried.
No TikTok credits are used. Completed `.partial.csv` files are renamed to `.csv` automatically.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging  # noqa: E402

from vidscreen.backfill import backfill_dir, needs_retry  # noqa: E402
from vidscreen.cache import JsonCache  # noqa: E402
from vidscreen.config import load_settings  # noqa: E402
from vidscreen.logging_setup import setup_logging  # noqa: E402
from vidscreen.transcripts import YouTubeTranscriptFetcher  # noqa: E402


def main() -> int:
    s = load_settings()
    setup_logging(s.log_dir)
    log = logging.getLogger("vidscreen")
    fetcher = YouTubeTranscriptFetcher(
        s.tr_languages, JsonCache(s.cache_dir), sleep=s.tr_yt_sleep,
        block_threshold=s.tr_block_threshold, workers=1,
    )
    results = backfill_dir(s.raw_dir, fetcher)
    filled = sum(r["updated"] for r in results)
    missing = sum(r["remaining"] for r in results)
    log.info("Done: %d transcripts filled this run, %d rows still missing.", filled, missing)
    if missing:
        log.info("Still missing -> run this script again later (or from another network).")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
