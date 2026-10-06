"""Reproducible stratified sample for manual labeling (platform x query-subtopic strata).

Design (see docs/screening_criteria.md, section 9):
* equal share per stratum within each platform (small strata are taken whole, the leftover share is
  redistributed to the other strata), chosen at random;
* a development / test split is made inside every stratum, BEFORE any rule or model exists;
* every row gets a sampling_weight (= stratum size / number sampled) so metrics can later be
  re-weighted to represent the whole 1,929-video population.
"""
from __future__ import annotations

import random

import pandas as pd


def primary_stratum(query_subtopics: str) -> str:
    """Subtopic of the query that found the video, or 'multi' when several subtopic-queries found it."""
    parts = sorted({p for p in (query_subtopics or "").split("|") if p})
    if len(parts) == 1:
        return parts[0]
    return "multi" if parts else "unknown"


def allocate(sizes: dict, total: int) -> dict:
    """Equal-share allocation with caps (water-filling). Deterministic."""
    alloc = {k: 0 for k in sizes}
    open_keys = {k for k, v in sizes.items() if v > 0}
    remaining = total
    while remaining > 0 and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        progressed = False
        for i, key in enumerate(sorted(open_keys, key=lambda k: (-sizes[k], k))):
            want = share + (1 if i < extra else 0)
            give = min(want, sizes[key] - alloc[key])
            if give > 0:
                alloc[key] += give
                remaining -= give
                progressed = True
            if alloc[key] >= sizes[key]:
                open_keys.discard(key)
        if not progressed:
            break
    return alloc


def draw_sample(df: pd.DataFrame, per_platform: int = 150, test_every: int = 3, seed: int = 42) -> pd.DataFrame:
    """Return the sampled rows with columns: stratum, split, sampling_weight, sample_id."""
    rng = random.Random(seed)
    work = df.copy()
    work["stratum"] = work["query_subtopics"].map(primary_stratum)
    picked = []
    for platform in sorted(work["platform"].unique()):
        plat = work[work["platform"] == platform]
        sizes = plat["stratum"].value_counts().to_dict()
        alloc = allocate(sizes, per_platform)
        for stratum in sorted(sizes):
            pool = plat[plat["stratum"] == stratum]
            n = alloc[stratum]
            if n == 0:
                continue
            idx = rng.sample(list(pool.index), n)  # random members of the stratum, random order
            chunk = work.loc[idx].copy()
            chunk["split"] = ["test" if i % test_every == 0 else "dev" for i in range(len(chunk))]
            chunk["sampling_weight"] = round(len(pool) / n, 4)
            picked.append(chunk)
    sample = pd.concat(picked)
    order = list(sample.index)
    rng.shuffle(order)  # labeler sees a mixed order, not strata blocks
    sample = sample.loc[order].reset_index(drop=True)
    sample["sample_id"] = [f"S{i:03d}" for i in range(1, len(sample) + 1)]
    return sample
