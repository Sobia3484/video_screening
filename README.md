# Video Screening — Infant & Child CPR and Choking Management

Stage 1 of the project: **query-driven data extraction** from YouTube and TikTok.
35 search queries → candidate videos → metadata + transcript → **one CSV per query**
(each CSV contains both YouTube and TikTok rows).

```
Queries (config/queries.csv)
   ├── YouTube : yt-dlp search + metadata      ── youtube-transcript-api (captions)
   └── TikTok  : ScrapeCreators keyword search ── caption (WebVTT) URL from the same response
                     ↓
        data/raw/Q001_pediatric_cpr.csv … Q035_*.csv  (+ _manifest.csv run log)
```

Later stages (merge → dedupe → clean → screen → EDA) read from `data/raw/` and write to
`data/intermediate/` and `data/final/`.

## Folder structure

```
video-screening/
├── config/
│   ├── queries.csv          # the 35 final queries (id, text, subtopic, language)
│   └── settings.yaml        # all tunable settings
├── data/
│   ├── raw/                 # one CSV per query + _manifest.csv
│   ├── cache/               # API responses / metadata / transcripts (git-ignored)
│   ├── intermediate/        # master_raw, deduplicated, cleaned  (next stages)
│   └── final/               # relevant_videos.csv                (next stages)
├── docs/                    # data_dictionary.md, extraction_design.md
├── logs/                    # one log file per run
├── notebooks/               # data_collection / screening / EDA (later)
├── scripts/
│   ├── smoke_test.py        # run FIRST: 1 query, tiny sizes
│   └── run_extraction.py    # full run / resume / selected queries
├── src/vidscreen/           # extractors, transcripts, pipeline, schema, CLI
├── tests/                   # offline unit tests
├── .env.example
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
# edit .env and paste your ScrapeCreators key
```

## Run

```bash
python scripts/smoke_test.py          # 1) verify YouTube + TikTok + transcripts (uses 1 TikTok credit)
python scripts/run_extraction.py      # 2) full run, all 35 queries, both platforms
```

Useful options:

| Command | What it does |
|---|---|
| `--platform youtube` / `--platform tiktok` | Run one platform only |
| `--queries Q001 Q014` | Only these queries |
| `--limit 5` | Only the first 5 queries |
| `--force` | Rebuild finished CSVs (cached data is reused, no extra credits) |
| `--no-transcripts` | Metadata only (fast first pass) |
| `--yt-results 100` / `--tt-pages 3` | Override sizes for this run |

**Resume is automatic.** A query with a finished CSV is skipped. If a run is interrupted, just run
the same command again. If TikTok credits run out mid-way, that query is written as
`Qxxx_*.partial.csv`, flagged `partial` in `_manifest.csv`, and finished on the next run.

## TikTok credit budget (100 free credits)

* 1 search page = 1 credit. Default `pages_per_query: 2` → 35 × 2 = **70 credits**.
* Safety limits in `settings.yaml`: `max_credits_per_run: 90`, `credit_reserve: 5`.
* Every raw API page is cached, so re-runs never spend credits twice.
* After the smoke test, check how many TikTok rows 1 page returns. If it's low, raise
  `pages_per_query` only as far as your remaining credits allow.

## Output columns

See `docs/data_dictionary.md`. Highlights: `query_id`, `platform`, `video_id`, `url`, `title`,
`description`, `creator_*`, `duration_seconds`, `upload_date`, `views/likes/comments/shares/saves`,
`hashtags`, `tags`, `language`, `country_code`, `transcript`, `transcript_status`, `metadata_status`.
CSVs are UTF-8 with BOM so Urdu/Hindi text opens correctly in Excel.

## Known limitations (by design — handle in cleaning/screening)

* **Country** is filled only when the platform reports it (TikTok `region`). YouTube rows stay blank → `Unknown` in EDA. Never infer country from language.
* **Transcripts**: YouTube needs existing captions; TikTok uses auto-captions when TikTok provides them.
  Missing transcript ≠ irrelevant video. Videos with `no_transcript` / `no_caption_available` can be sent to faster-whisper later.
* `transcript_status = blocked` means YouTube blocked your IP; extraction pauses transcripts after 5 in a row. Retry later or from another network.
* If yt-dlp asks to "confirm you're not a bot", export a `cookies.txt` and set `youtube.cookies_file`.
* Keep `yt-dlp` updated: `pip install -U yt-dlp`.
* The same video can appear under several queries; that is expected. Deduplication happens in the next stage, and `query_id` is kept so query overlap stays measurable.
* Collect public data only, respect platform terms, and don't redistribute raw transcripts.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```
