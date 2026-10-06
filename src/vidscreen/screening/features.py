"""Numeric / signal features for the revised screening (no views, no likes, no query information, no transcript).

Duration is z-scored WITHIN each platform so that a 60-second TikTok and a 60-second YouTube video are not
compared on the same raw scale. All features are computed without looking at labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import ADULT, CERT, CHILD, INSTRUCT, PET, PROMO, TOPIC
from .text import fold

NUMERIC = [
    "duration_z", "title_len_log", "desc_len_log", "has_description", "n_hashtags", "n_tags",
    "child_n", "topic_n", "topic_hit", "promo_hit", "instruct_hit", "adult_hit", "cert_hit", "pet_hit", "is_tiktok",
]


def _count(rx, texts) -> np.ndarray:
    return np.array([len({m.group(0) for m in rx.finditer(t)}) for t in texts], dtype=float)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    head = [fold(f"{t} {h.replace('|', ' ')} {g.replace('|', ' ')}") for t, h, g in zip(d["title"], d["hashtags"], d["tags"])]
    body = [fold(x or "") for x in d["description"]]
    allt = [f"{a} {b}" for a, b in zip(head, body)]

    log_dur = np.log1p(pd.to_numeric(d["duration_seconds"], errors="coerce").fillna(0))
    grp = log_dur.groupby(d["platform"])
    d["duration_z"] = ((log_dur - grp.transform("mean")) / grp.transform("std").replace(0, 1)).fillna(0)
    d["title_len_log"] = np.log1p(d["title"].str.len())
    d["desc_len_log"] = np.log1p(d["description"].str.len())
    d["has_description"] = (d["description"].str.len() > 0).astype(float)
    d["n_hashtags"] = d["hashtags"].map(lambda s: len([x for x in s.split("|") if x]))
    d["n_tags"] = d["tags"].map(lambda s: len([x for x in s.split("|") if x]))
    d["child_n"] = _count(CHILD, allt)
    d["topic_n"] = _count(TOPIC, allt)
    d["topic_hit"] = (d["topic_n"] > 0).astype(float)
    for name, rx in (("promo_hit", PROMO), ("instruct_hit", INSTRUCT), ("adult_hit", ADULT), ("cert_hit", CERT), ("pet_hit", PET)):
        d[name] = (_count(rx, allt) > 0).astype(float)
    d["is_tiktok"] = (d["platform"] == "tiktok").astype(float)
    d["word_count"] = d["text"].str.split().str.len()
    return d