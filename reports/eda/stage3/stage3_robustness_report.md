# Stage 3 — Robustness Analysis
## 1. Purpose
Stage 3 checks whether the main EDA findings remain similar when uncertain model-selected videos are removed and when creator concentration is controlled by limiting each creator to at most 3 videos.
## 2. Dataset sizes
- Main relevant dataset: **1,309 videos**
- Confident relevant subset: **1,167 videos**
- Creator-capped main dataset: **1,075 videos**
- Creator-capped confident dataset: **954 videos**

## 3. Main vs confident subset
- **tiktok:** main 56.0% vs confident 53.6%
- **youtube:** main 44.0% vs confident 46.4%

## 4. Creator concentration
- **main**: 815 creators; median 1.0 videos/creator; top 10 creators account for 14.5% of videos.
- **confident**: 718 creators; median 1.0 videos/creator; top 10 creators account for 15.6% of videos.

## 5. Creator-cap robustness
- **main:** retaining 82.1% of videos after the 3-video creator cap changed median views from 9969.0 to 7188.0.
- **confident:** retaining 81.8% of videos after the 3-video creator cap changed median views from 9537.0 to 7105.5.

## 6. Interpretation
The main and confident datasets are compared to determine whether conclusions depend strongly on lower-confidence model decisions. The creator-capped analysis checks whether results are dominated by a small number of prolific creators.
If platform proportions, median duration, engagement patterns, subtopic distributions and language distributions remain broadly similar after these checks, the Stage 2 findings can be considered reasonably robust to creator concentration and model confidence.

## 7. Important limitation
The creator cap is a sensitivity analysis, not a correction to the original dataset. Search-result sampling can still favour certain creators, platforms and types of videos.

## 8. Generated tables
- `01_main_vs_confident.csv`
- `02_platform_robustness.csv`
- `03_creator_concentration.csv`
- `06_creator_cap_robustness.csv`
- `07_subtopic_robustness.csv`
- `08_language_robustness.csv`
