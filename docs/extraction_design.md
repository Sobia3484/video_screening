# Extraction design notes

**Why these tools**
* YouTube search + metadata: `yt-dlp` — free, no API key, rich fields (views, likes, tags, upload date).
* YouTube transcripts: `youtube-transcript-api` — reads existing captions, manual preferred over auto-generated.
* TikTok: ScrapeCreators keyword search — reliable and credit-based. Its response already contains
  stats, hashtags, region and a caption (WebVTT) URL, so TikTok transcripts cost no extra credits.

**Caching (`data/cache/`)** — namespaces: `yt_search`, `yt_video`, `yt_transcript`, `tt_search`, `tt_caption`.
A video returned by several queries is fetched once. Permanent outcomes (no captions, private video) are cached;
temporary ones (IP block, network error) are not, so they are retried on the next run.

**Reliability** — retries with exponential backoff; per-video failures never stop a query; a failed video still
gets a row with `metadata_status` explaining why; CSVs are written atomically; incomplete queries are saved as `.partial.csv`.

**Not included yet (planned fallbacks)** — faster-whisper speech-to-text for videos without captions;
YouTube Data API for channel country; ScrapeCreators profile endpoint for TikTok creator country.
