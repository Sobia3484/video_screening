"""Fill missing YouTube transcripts in already-written CSVs (no re-search, no TikTok credits).

A row needs a retry when its transcript was blocked by YouTube (`blocked`, `skipped_blocked`)
or failed with an unexpected error (`error:*`). Definitive outcomes (ok, no_transcript,
disabled, ...) are never touched. When a `.partial.csv` has nothing left to retry it is
renamed to the final `.csv`.
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

from .utils import word_count

log = logging.getLogger("vidscreen")
csv.field_size_limit(2**31 - 1)  # sys.maxsize overflows on Windows
RETRY_STATUSES = {"blocked", "skipped_blocked"}


def needs_retry(row: dict) -> bool:
    status = row.get("transcript_status", "")
    return (
        row.get("platform") == "youtube"
        and row.get("metadata_status") == "ok"
        and (status in RETRY_STATUSES or status.startswith("error:"))
    )


def _is_retry_status(status: str) -> bool:
    return status in RETRY_STATUSES or status.startswith("error:")


def backfill_csv(path: Path, fetcher) -> dict:
    """Update one CSV in place. Returns stats; `path` key is the (possibly renamed) file."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    updated = 0
    for row in rows:
        if not needs_retry(row):
            continue
        if fetcher.halted:
            break
        res = fetcher.fetch(row["video_id"])
        if _is_retry_status(res.status):
            continue  # still blocked/failing: leave the row as it was
        row["transcript"] = res.text or ""
        row["transcript_status"] = res.status
        row["transcript_source"] = res.source
        row["transcript_language"] = res.language
        row["transcript_is_auto"] = "" if res.is_auto is None else ("True" if res.is_auto else "False")
        row["transcript_word_count"] = word_count(res.text) if res.text else 0
        updated += 1

    remaining = sum(needs_retry(r) for r in rows)
    if updated:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)

    final_path = path
    if remaining == 0 and path.name.endswith(".partial.csv"):
        final_path = path.with_name(path.name.replace(".partial.csv", ".csv"))
        os.replace(path, final_path)
        log.info("%s is now complete -> %s", path.name, final_path.name)
    return {"path": final_path, "updated": updated, "remaining": remaining, "rows": len(rows)}


def backfill_dir(raw_dir: Path, fetcher) -> list[dict]:
    files = sorted(Path(raw_dir).glob("Q*.csv"))  # matches both .csv and .partial.csv
    results = []
    for path in files:
        if fetcher.halted:
            log.warning("YouTube is blocking again - stopping. Run this script again later; progress is saved.")
            break
        stats = backfill_csv(path, fetcher)
        log.info("%s: filled %d, still missing %d", stats["path"].name, stats["updated"], stats["remaining"])
        results.append(stats)
    return results
