# Data dictionary — per-query CSVs (`data/raw/Qxxx_<slug>.csv`)

| Column | Type | Description |
|---|---|---|
| query_id | str | e.g. `Q002` — links to `config/queries.csv` |
| query_text | str | The search text used |
| platform | str | `youtube` or `tiktok` |
| query_rank | int | Position in that platform's results for this query (1 = top) |
| video_id | str | Platform video id (dedupe key together with `platform`) |
| url | str | Canonical video URL |
| title | str | YouTube title. **TikTok: caption text is stored here** |
| description | str | YouTube description. TikTok: empty (single caption field) |
| creator_name | str | Channel / TikTok nickname |
| creator_id | str | YouTube `channel_id` / TikTok handle |
| creator_url | str | Channel or profile URL |
| creator_followers | int | Subscribers / followers when available |
| duration_seconds | float | Video length in seconds |
| upload_date | date | `YYYY-MM-DD` (UTC) |
| views, likes, comments | int | Engagement counts at retrieval time (likes may be hidden on YouTube → blank) |
| shares, saves | int | TikTok only |
| hashtags | str | Pipe-separated, lower-case, from caption/title/description |
| tags | str | YouTube tags, pipe-separated |
| category | str | YouTube category |
| language | str | Platform-reported language; often blank or `un` |
| country_code | str | Only when platform reports it (TikTok video `region`); never inferred |
| country_source | str | Where `country_code` came from |
| transcript | str | Full transcript text |
| transcript_status | str | `ok`, `no_transcript`, `disabled`, `video_unavailable`, `age_restricted`, `blocked`, `skipped_blocked`, `no_caption_available`, `empty_caption`, `caption_download_failed`, `not_requested`, `error:<Type>` |
| transcript_source | str | `youtube_captions`, `tiktok_auto_caption`, `tiktok_caption` |
| transcript_language | str | Language code of the transcript |
| transcript_is_auto | bool | Auto-generated captions? |
| transcript_word_count | int | Words in transcript |
| metadata_status | str | `ok`, `unavailable: …`, `failed: …` |
| thumbnail_url | str | Thumbnail/cover URL (TikTok URLs expire) |
| retrieved_at_utc | str | Extraction timestamp |

`data/raw/_manifest.csv` logs each query run: rows per platform, transcripts found,
TikTok credits spent, status (`complete`/`partial`) and notes.

`config/queries.csv` carries a `subtopic` label per **query** (cpr, neonatal, bls, choking, …).
It describes the query, not the video; the video's true subtopic is decided during screening.
