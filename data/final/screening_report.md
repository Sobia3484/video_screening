# Screening report (revised: score-based, separate vs combined platform models)

Seed 42. 300 human-labeled videos; 3 x 5-fold CV, stratified by label and platform, grouped by creator (a creator never appears in both training and validation).

Inputs: title, description, hashtags, tags, duration (z-scored within platform), title/description length, hashtag and tag counts, and rule signals (child term, topic term, promotion, instruction, adult, certification, pet). **Not used:** transcripts, views, likes, query information.

## Baselines (on all labeled videos)

| baseline | precision | recall | F1 |
|---|---|---|---|
| all_relevant | 0.6833 | 1.0 | 0.8119 |
| rules_only (potentially->relevant; tuned on the development split, optimistic) | 0.7576 | 0.9756 | 0.8529 |
| rule gate only (strong topic term present) | 0.7782 | 0.9756 | 0.8658 |

## Candidate comparison (pooled out-of-fold scores; F1 columns use the F1-optimal threshold per candidate, so they are slightly optimistic)

| candidate | macro_auc | auc | pr_auc | brier | precision | recall | f1 | youtube_auc | tiktok_auc | youtube_f1 | tiktok_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| embed|text|combined|gate | 0.851 | 0.889 | 0.942 | 0.151 | 0.857 | 0.932 | 0.892 | 0.811 | 0.890 | 0.924 | 0.842 |
| tfidf|text|per_platform|gate | 0.841 | 0.874 | 0.925 | 0.133 | 0.850 | 0.942 | 0.893 | 0.810 | 0.871 | 0.922 | 0.845 |
| embed|text|per_platform|gate | 0.836 | 0.880 | 0.931 | 0.165 | 0.845 | 0.956 | 0.897 | 0.778 | 0.894 | 0.923 | 0.854 |
| tfidf|text|combined|gate | 0.835 | 0.871 | 0.928 | 0.135 | 0.846 | 0.937 | 0.889 | 0.809 | 0.860 | 0.926 | 0.827 |
| embed|text|combined|nogate | 0.831 | 0.870 | 0.935 | 0.159 | 0.830 | 0.927 | 0.876 | 0.794 | 0.868 | 0.920 | 0.807 |
| tfidf|text|per_platform|nogate | 0.815 | 0.849 | 0.912 | 0.144 | 0.810 | 0.956 | 0.877 | 0.791 | 0.839 | 0.916 | 0.816 |
| embed|text|per_platform|nogate | 0.809 | 0.849 | 0.919 | 0.176 | 0.807 | 0.961 | 0.877 | 0.745 | 0.873 | 0.916 | 0.816 |
| tfidf|text|combined|nogate | 0.807 | 0.844 | 0.914 | 0.145 | 0.821 | 0.937 | 0.875 | 0.784 | 0.831 | 0.922 | 0.800 |
| embed|text+features|combined|nogate | 0.800 | 0.845 | 0.908 | 0.151 | 0.851 | 0.893 | 0.871 | 0.719 | 0.881 | 0.891 | 0.840 |
| embed|text+features|combined|gate | 0.799 | 0.849 | 0.908 | 0.152 | 0.829 | 0.922 | 0.873 | 0.724 | 0.874 | 0.901 | 0.830 |
| tfidf|text+features|combined|nogate | 0.792 | 0.831 | 0.903 | 0.153 | 0.811 | 0.942 | 0.871 | 0.713 | 0.871 | 0.902 | 0.825 |
| tfidf|text+features|combined|gate | 0.792 | 0.839 | 0.904 | 0.153 | 0.818 | 0.942 | 0.875 | 0.720 | 0.864 | 0.906 | 0.830 |
| tfidf|text+features|per_platform|nogate | 0.778 | 0.822 | 0.900 | 0.161 | 0.826 | 0.927 | 0.874 | 0.679 | 0.876 | 0.897 | 0.839 |
| tfidf|text+features|per_platform|gate | 0.777 | 0.833 | 0.902 | 0.163 | 0.859 | 0.893 | 0.876 | 0.686 | 0.868 | 0.900 | 0.835 |
| embed|text+features|per_platform|gate | 0.776 | 0.826 | 0.897 | 0.173 | 0.852 | 0.898 | 0.874 | 0.681 | 0.870 | 0.900 | 0.832 |
| embed|text+features|per_platform|nogate | 0.775 | 0.813 | 0.895 | 0.171 | 0.848 | 0.898 | 0.872 | 0.674 | 0.877 | 0.897 | 0.832 |

**Selected:** `embed|text|combined|gate` (rule: highest macro (per-platform) AUC over pooled out-of-fold scores; candidates within 0.01 are tied -> least complex). Decision thresholds (calibrated probability): {'tiktok': 0.66, 'youtube': 0.66}.

Selected model: macro AUC 0.8505, precision 0.8565, recall 0.9317, F1 0.8925.

## What the design choices did (mean change in macro AUC over otherwise identical candidate pairs)

| change | mean macro-AUC change | pairs | pairs that improved |
|---|---|---|---|
| add duration/length/signal features | -0.042 | 8 | 0 |
| add rule gate | +0.012 | 8 | 5 |
| separate TikTok/YouTube models instead of one | -0.013 | 8 | 2 |
| pretrained embeddings instead of TF-IDF | +0.005 | 8 | 4 |

## Calibration of the selected model (out-of-fold)

Raw score: expected calibration error 0.1664, Brier 0.1508. After Platt scaling (monotone, ranking unchanged): expected calibration error 0.077, Brier 0.1282 (the calibrator is fitted on the same out-of-fold scores, so this is slightly optimistic).

| scale | score bin | videos | mean score | observed relevant rate |
|---|---|---|---|---|
| raw | 0.0-0.2 | 50 | 0.02 | 0.12 |
| raw | 0.2-0.4 | 33 | 0.33 | 0.333 |
| raw | 0.4-0.6 | 65 | 0.523 | 0.677 |
| raw | 0.6-0.8 | 129 | 0.698 | 0.938 |
| raw | 0.8-1.0 | 23 | 0.832 | 1.0 |
| calibrated | 0.0-0.2 | 43 | 0.032 | 0.116 |
| calibrated | 0.2-0.4 | 6 | 0.317 | 0.167 |
| calibrated | 0.4-0.6 | 18 | 0.536 | 0.278 |
| calibrated | 0.6-0.8 | 67 | 0.716 | 0.627 |
| calibrated | 0.8-1.0 | 166 | 0.868 | 0.916 |

## Zones (calibrated probability bands {'irrelevant_confident': '<= 0.25', 'relevant_confident': '>= 0.75', 'audit': 'in between'}), observed on the labeled videos

| zone | videos | observed relevant rate |
|---|---|---|
| irrelevant_confident | 44 | 0.114 |
| audit | 66 | 0.485 |
| relevant_confident | 190 | 0.884 |

## Patterns (interpretable TF-IDF companion model; largest word coefficients)

**TF-IDF companion, all videos, towards relevant:** infant, cpr, choking, child, infant cpr, how to, neonatal resuscitation, resuscitation, how, infants, aed, life, fypシ, after birth, of the

**TF-IDF companion, all videos, towards irrelevant:** adult, childbirtheducation, nicubaby, paramedic, my, viral, para, babytok, school, fyp viral, their, old, his, uses, when

**TF-IDF companion, tiktok, towards relevant:** infant, cpr, choking, aed, infant cpr, save, how to, life, heimlich, fypシ, birth, after birth, cprtraining, medicine, children

**TF-IDF companion, tiktok, towards irrelevant:** pediatrics, paramedic, childbirtheducation, nicubaby, medical, certified, my, his, para, fyp viral, viral, our, be, neonatal, was

**TF-IDF companion, youtube, towards relevant:** infant, child, neonatal, neonatal resuscitation, infant cpr, cpr, paediatric, how, infant choking, infants, resuscitation, child choking, choking, of the, an infant

**TF-IDF companion, youtube, towards irrelevant:** adult, training first, of foreign, upper airway, upper, their, management of, heimlich maneuver, body, maneuver, pregnant, year old, adult choking, breathing practice, foreign


## Error categories (out-of-fold, selected model)

| error | category | n |
|---|---|---|
| false_negative | certification / exam / student wording | 2 |
| false_negative | no child term in text | 6 |
| false_negative | no topic term in text | 5 |
| false_negative | other | 3 |
| false_negative | very short / uninformative text | 1 |
| false_positive | adult marker | 2 |
| false_positive | advert / promotion wording | 1 |
| false_positive | certification / exam / student wording | 8 |
| false_positive | no child term in text | 5 |
| false_positive | other | 13 |

## Secondary consistency check on the original 105 test videos

Fitted on the 195 development videos only: precision 0.8158, recall 0.9538, F1 0.8794 (95% CI (0.8169, 0.9333)). The 105 test videos were inspected in the earlier experiment, so this is a consistency check, not an unseen test.

## Screening flow

1929 cleaned videos -> **1309 relevant**, 620 irrelevant (300 human labels kept, 1629 scored by the model).
Relevant by platform: {'tiktok': 733, 'youtube': 576}. Zones of model-scored videos: {'relevant_confident': 962, 'audit': 370, 'irrelevant_confident': 297}. Confident-relevant subset (for sensitivity analysis): 1167.

## Limitations

- Labels were pre-annotated by an AI assistant from text fields and verified by the author; the assistant saw all 300 videos, including the earlier test portion.
- Ground truth reflects what the video shows when opened; models only see text fields and simple metadata.
- The earlier 105-video test split had already been inspected, so model development here uses creator-grouped cross-validation over all 300 labels and the old test split is only a secondary check.
- YouTube has few labeled irrelevant videos, so YouTube metrics have wide uncertainty.
- F1-optimal thresholds are tuned on the same out-of-fold scores, which is slightly optimistic.
