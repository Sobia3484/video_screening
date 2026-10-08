from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Stage 3 — Robustness + Final EDA
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MAIN_FILE = ROOT / "data" / "final" / "relevant_videos.csv"
CONFIDENT_FILE = ROOT / "data" / "final" / "relevant_confident_videos.csv"

OUT_DIR = ROOT / "reports" / "eda" / "stage3"
TABLE_DIR = OUT_DIR / "tables"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CREATOR_CAP = 3


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clean_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_median(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan
    return series.median()


def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan
    return series.mean()


def pct(value, total):
    if total == 0:
        return np.nan
    return 100 * value / total


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("[Stage 3] Loading datasets...")

main = pd.read_csv(MAIN_FILE)
confident = pd.read_csv(CONFIDENT_FILE)

print(f"[Stage 3] Main dataset: {len(main):,}")
print(f"[Stage 3] Confident dataset: {len(confident):,}")


# ------------------------------------------------------------
# Basic preparation
# ------------------------------------------------------------

numeric_cols = [
    "duration_seconds",
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "creator_followers",
    "query_count",
    "score",
]

main = clean_numeric(main, numeric_cols)
confident = clean_numeric(confident, numeric_cols)


# ------------------------------------------------------------
# 1. Main vs confident subset
# ------------------------------------------------------------

print("[Stage 3] Comparing main and confident datasets...")

comparison_rows = []

for name, df in [
    ("main", main),
    ("confident", confident),
]:

    row = {
        "dataset": name,
        "videos": len(df),
        "youtube": int((df["platform"].str.lower() == "youtube").sum())
        if "platform" in df.columns else np.nan,
        "tiktok": int((df["platform"].str.lower() == "tiktok").sum())
        if "platform" in df.columns else np.nan,
        "median_duration_seconds": safe_median(
            df["duration_seconds"]
        ) if "duration_seconds" in df.columns else np.nan,
        "median_views": safe_median(
            df["views"]
        ) if "views" in df.columns else np.nan,
        "median_likes": safe_median(
            df["likes"]
        ) if "likes" in df.columns else np.nan,
        "median_comments": safe_median(
            df["comments"]
        ) if "comments" in df.columns else np.nan,
    }

    comparison_rows.append(row)


comparison = pd.DataFrame(comparison_rows)

comparison.to_csv(
    TABLE_DIR / "01_main_vs_confident.csv",
    index=False
)


# ------------------------------------------------------------
# 2. Platform comparison
# ------------------------------------------------------------

print("[Stage 3] Platform robustness comparison...")

platform_rows = []

for dataset_name, df in [
    ("main", main),
    ("confident", confident),
]:

    if "platform" not in df.columns:
        continue

    for platform, group in df.groupby("platform", dropna=False):

        platform_rows.append({
            "dataset": dataset_name,
            "platform": platform,
            "videos": len(group),
            "share_percent": round(
                pct(len(group), len(df)), 2
            ),
            "median_duration_seconds": safe_median(
                group["duration_seconds"]
            ),
            "median_views": safe_median(
                group["views"]
            ),
            "median_likes": safe_median(
                group["likes"]
            ),
            "median_comments": safe_median(
                group["comments"]
            ),
        })


platform_summary = pd.DataFrame(platform_rows)

platform_summary.to_csv(
    TABLE_DIR / "02_platform_robustness.csv",
    index=False
)


# ------------------------------------------------------------
# 3. Creator concentration
# ------------------------------------------------------------

print("[Stage 3] Checking creator concentration...")

def creator_analysis(df, dataset_name):

    if "creator_id" in df.columns:
        creator_col = "creator_id"
    elif "creator_name" in df.columns:
        creator_col = "creator_name"
    else:
        return pd.DataFrame()

    creators = (
        df[creator_col]
        .fillna("UNKNOWN")
        .astype(str)
    )

    counts = creators.value_counts()

    result = {
        "dataset": dataset_name,
        "videos": len(df),
        "creators": len(counts),
        "median_videos_per_creator": counts.median(),
        "max_videos_per_creator": counts.max(),
        "top_10_video_share_percent": round(
            pct(counts.head(10).sum(), len(df)), 2
        ),
        "top_20_video_share_percent": round(
            pct(counts.head(20).sum(), len(df)), 2
        ),
    }

    return pd.DataFrame([result])


creator_main = creator_analysis(main, "main")
creator_confident = creator_analysis(confident, "confident")

creator_summary = pd.concat(
    [creator_main, creator_confident],
    ignore_index=True
)

creator_summary.to_csv(
    TABLE_DIR / "03_creator_concentration.csv",
    index=False
)


# ------------------------------------------------------------
# 4. Creator-capped robustness dataset
# ------------------------------------------------------------

print(
    f"[Stage 3] Applying creator cap: "
    f"{CREATOR_CAP} videos per creator..."
)


def cap_creators(df, cap=3):

    if "creator_id" in df.columns:
        creator_col = "creator_id"
    elif "creator_name" in df.columns:
        creator_col = "creator_name"
    else:
        print("[Stage 3] No creator column found; skipping cap.")
        return df.copy()

    work = df.copy()

    work["_creator_key"] = (
        work[creator_col]
        .fillna("UNKNOWN")
        .astype(str)
    )

    # Deterministic shuffle before taking the cap.
    work = work.sample(
        frac=1,
        random_state=SEED
    )

    capped = (
        work
        .groupby("_creator_key", group_keys=False)
        .head(cap)
        .copy()
    )

    capped = capped.drop(columns=["_creator_key"])

    return capped


main_capped = cap_creators(main, CREATOR_CAP)

confident_capped = cap_creators(
    confident,
    CREATOR_CAP
)


main_capped.to_csv(
    TABLE_DIR / "04_main_creator_capped.csv",
    index=False
)

confident_capped.to_csv(
    TABLE_DIR / "05_confident_creator_capped.csv",
    index=False
)


# ------------------------------------------------------------
# 5. Compare creator-capped vs original
# ------------------------------------------------------------

robustness_rows = []

for name, original, capped in [
    ("main", main, main_capped),
    ("confident", confident, confident_capped),
]:

    robustness_rows.append({
        "dataset": name,
        "original_videos": len(original),
        "creator_capped_videos": len(capped),
        "retained_percent": round(
            pct(len(capped), len(original)), 2
        ),
        "original_median_duration": safe_median(
            original["duration_seconds"]
        ),
        "capped_median_duration": safe_median(
            capped["duration_seconds"]
        ),
        "original_median_views": safe_median(
            original["views"]
        ),
        "capped_median_views": safe_median(
            capped["views"]
        ),
        "original_median_likes": safe_median(
            original["likes"]
        ),
        "capped_median_likes": safe_median(
            capped["likes"]
        ),
        "original_median_comments": safe_median(
            original["comments"]
        ),
        "capped_median_comments": safe_median(
            capped["comments"]
        ),
    })


robustness = pd.DataFrame(robustness_rows)

robustness.to_csv(
    TABLE_DIR / "06_creator_cap_robustness.csv",
    index=False
)


# ------------------------------------------------------------
# 6. Subtopic robustness
# ------------------------------------------------------------

print("[Stage 3] Checking subtopic stability...")


def subtopic_table(df, dataset_name):

    if "query_subtopics" not in df.columns:
        return pd.DataFrame()

    rows = []

    for value, count in (
        df["query_subtopics"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(15)
        .items()
    ):

        rows.append({
            "dataset": dataset_name,
            "subtopic": value,
            "videos": count,
            "share_percent": round(
                pct(count, len(df)), 2
            )
        })

    return pd.DataFrame(rows)


subtopics = pd.concat(
    [
        subtopic_table(main, "main"),
        subtopic_table(confident, "confident"),
        subtopic_table(main_capped, "main_creator_capped"),
    ],
    ignore_index=True
)

subtopics.to_csv(
    TABLE_DIR / "07_subtopic_robustness.csv",
    index=False
)


# ------------------------------------------------------------
# 7. Language robustness
# ------------------------------------------------------------

print("[Stage 3] Checking language stability...")


def language_table(df, dataset_name):

    column = None

    for candidate in [
        "language_clean",
        "language"
    ]:
        if candidate in df.columns:
            column = candidate
            break

    if column is None:
        return pd.DataFrame()

    rows = []

    for language, count in (
        df[column]
        .fillna("und")
        .astype(str)
        .value_counts()
        .head(15)
        .items()
    ):

        rows.append({
            "dataset": dataset_name,
            "language": language,
            "videos": count,
            "share_percent": round(
                pct(count, len(df)), 2
            )
        })

    return pd.DataFrame(rows)


languages = pd.concat(
    [
        language_table(main, "main"),
        language_table(confident, "confident"),
        language_table(main_capped, "main_creator_capped"),
    ],
    ignore_index=True
)

languages.to_csv(
    TABLE_DIR / "08_language_robustness.csv",
    index=False
)


# ------------------------------------------------------------
# 8. Generate final Stage 3 report
# ------------------------------------------------------------

print("[Stage 3] Writing report...")


report = []

report.append("# Stage 3 — Robustness Analysis\n")

report.append("## 1. Purpose\n")

report.append(
    "Stage 3 checks whether the main EDA findings remain similar "
    "when uncertain model-selected videos are removed and when "
    "creator concentration is controlled by limiting each creator "
    f"to at most {CREATOR_CAP} videos.\n"
)

report.append("## 2. Dataset sizes\n")

report.append(
    f"- Main relevant dataset: **{len(main):,} videos**\n"
)

report.append(
    f"- Confident relevant subset: **{len(confident):,} videos**\n"
)

report.append(
    f"- Creator-capped main dataset: **{len(main_capped):,} videos**\n"
)

report.append(
    f"- Creator-capped confident dataset: **{len(confident_capped):,} videos**\n"
)

report.append("\n## 3. Main vs confident subset\n")

main_platform = (
    main["platform"].value_counts(normalize=True) * 100
    if "platform" in main.columns else pd.Series()
)

conf_platform = (
    confident["platform"].value_counts(normalize=True) * 100
    if "platform" in confident.columns else pd.Series()
)

for platform in sorted(
    set(main_platform.index).union(conf_platform.index)
):

    m = main_platform.get(platform, 0)
    c = conf_platform.get(platform, 0)

    report.append(
        f"- **{platform}:** main {m:.1f}% vs confident {c:.1f}%\n"
    )


report.append("\n## 4. Creator concentration\n")

if not creator_summary.empty:

    for _, row in creator_summary.iterrows():

        report.append(
            f"- **{row['dataset']}**: "
            f"{int(row['creators']):,} creators; "
            f"median {row['median_videos_per_creator']:.1f} videos/creator; "
            f"top 10 creators account for "
            f"{row['top_10_video_share_percent']:.1f}% of videos.\n"
        )


report.append("\n## 5. Creator-cap robustness\n")

for _, row in robustness.iterrows():

    report.append(
        f"- **{row['dataset']}:** retaining "
        f"{row['retained_percent']:.1f}% of videos after the "
        f"{CREATOR_CAP}-video creator cap changed median views from "
        f"{row['original_median_views']:.1f} to "
        f"{row['capped_median_views']:.1f}.\n"
    )


report.append("\n## 6. Interpretation\n")

report.append(
    "The main and confident datasets are compared to determine "
    "whether conclusions depend strongly on lower-confidence "
    "model decisions. The creator-capped analysis checks whether "
    "results are dominated by a small number of prolific creators.\n"
)

report.append(
    "If platform proportions, median duration, engagement patterns, "
    "subtopic distributions and language distributions remain broadly "
    "similar after these checks, the Stage 2 findings can be considered "
    "reasonably robust to creator concentration and model confidence.\n"
)

report.append("\n## 7. Important limitation\n")

report.append(
    "The creator cap is a sensitivity analysis, not a correction to "
    "the original dataset. Search-result sampling can still favour "
    "certain creators, platforms and types of videos.\n"
)

report.append("\n## 8. Generated tables\n")

for filename in [
    "01_main_vs_confident.csv",
    "02_platform_robustness.csv",
    "03_creator_concentration.csv",
    "06_creator_cap_robustness.csv",
    "07_subtopic_robustness.csv",
    "08_language_robustness.csv",
]:

    report.append(f"- `{filename}`\n")


report_path = OUT_DIR / "stage3_robustness_report.md"

report_path.write_text(
    "".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# Final console summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("STAGE 3 COMPLETE")
print("=" * 60)

print(f"Main dataset:              {len(main):,}")
print(f"Confident dataset:         {len(confident):,}")
print(f"Main creator-capped:       {len(main_capped):,}")
print(f"Confident creator-capped:  {len(confident_capped):,}")

print()
print(f"Report:")
print(f"  {report_path}")

print()
print("Tables:")
print(f"  {TABLE_DIR}")

print("=" * 60)