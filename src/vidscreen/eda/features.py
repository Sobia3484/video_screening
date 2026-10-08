"""EDA features. Everything is derived from the screened dataset; no labels or model outputs are changed."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from ..screening.rules import _rx
from ..screening.text import fold

DURATION_BINS = [0, 60, 300, 600, 1200, np.inf]
DURATION_LABELS = ["under 1 min", "1-5 min", "5-10 min", "10-20 min", "20+ min"]
FOLLOWER_BINS = [-1, 999, 9_999, 99_999, 999_999, np.inf]
FOLLOWER_LABELS = ["under 1K", "1K-10K", "10K-100K", "100K-1M", "1M+"]

INFANT = _rx([r"infant\w*", r"bab(?:y|ies)\w*", r"bebe\w*", r"lactante\w*", r"bayi\b", r"bebek\w*", r"младен\w*", r"груднич\w*", r"shishu\b"])
NEWBORN = _rx([r"newborn\w*", r"neonat\w*", r"recien\s*nacid\w*", r"recem[\s-]*nascid\w*", r"nouveau[\s-]*ne\w*", r"новорожд\w*", r"नवजात\w*", r"نوزاد\w*", r"nrp\b"])
CHILD = _rx([r"child\w*", r"kids?\b", r"toddler\w*", r"p(?:a)?ediatr\w*", r"peds\b", r"nin[oa]s?\b", r"crianca\w*", r"enfants?\b", r"anak\b",
             r"cocuk\w*", r"ребен\w*", r"ребён\w*", r"дет(?:ей|ск\w*|и\b)", r"bach(?:on|cho|che|chon|chey|ay|ey|a|i)\b", r"بچ\w*",
             r"बच्च\w*", r"طفل\w*", r"اطفال", r"أطفال"])
ADULT = _rx([r"adults?\b", r"adulto\w*"])

SUB = {
    "cpr": _rx([r"c\.?p\.?r\w*", r"rcp\b", r"cardiopulmon\w*", r"cardio[\s-]*pulmon\w*", r"resuscit\w*", r"ressuscit\w*", r"reanima\w*", r"сердечно", r"реанимац\w*"]),
    "choking": _rx([r"(?:anti)?chok\w*", r"heimlich\w*", r"back\s+blow\w*", r"chest\s+thrust\w*", r"abdominal\s+thrust\w*", r"atraganta\w*",
                    r"desengasg\w*", r"engasg\w*", r"tersedak", r"подавил\w*", r"поперхну\w*", r"dam\s+ghut\w*", r"اختناق"]),
    "bls": _rx([r"bls\b", r"(?:basic\s+)?life\s+support", r"pals\b", r"nls\b", r"bantuan\s+hidup", r"temel\s+yasam"]),
    "aed": _rx([r"\baed\b", r"defibril\w*", r"desfibril\w*"]),
    "neonatal": _rx([r"neonat\w*", r"newborn\w*", r"nrp\b", r"recien\s*nacid\w*", r"recem[\s-]*nascid\w*", r"новорожд\w*", r"नवजात\w*", r"نوزاد\w*"]),
    "rescue_breathing": _rx([r"rescue\s+breath\w*", r"mouth[\s-]*to[\s-]*mouth", r"boca\s+a\s+boca", r"bag[\s-]*(?:valve[\s-]*)?mask", r"ppv\b", r"napas\s+buatan", r"suni\s+solunum"]),
    "chest_compressions": _rx([r"chest\s+compression\w*", r"compression\w*", r"compress(?:ao|oes)\b", r"compresion(?:es)?\b", r"pijat\s+jantung", r"kalp\s+masaj\w*"]),
    "airway_obstruction": _rx([r"airway\s+obstruct\w*", r"fbao\b", r"foreign\s+(?:body|object)\w*"]),
}
SUBTOPIC_LABELS = {
    "combined": "CPR and choking", "neonatal_resus": "neonatal resuscitation", "choking": "choking", "airway_obstruction": "airway obstruction",
    "aed": "AED", "bls": "BLS", "rescue_breathing": "rescue breathing", "chest_compressions": "chest compressions", "cpr": "CPR",
    "neonatal_other": "neonatal (other)", "unspecified": "not specified in text",
}

_STOP = {
    "en": "the and of to is in you for it this that with are on be your how what when can if do not".split(),
    "es": "el la los las de que y en un una para con por como es se su lo al más muy".split(),
    "pt": "o a os as de que e em um uma para com por como é se seu não mais muito você".split(),
    "fr": "le la les des et en un une pour avec par comme est se son pas plus vous que".split(),
    "id": "yang dan di ke dari untuk dengan ini itu pada adalah tidak anda bayi anak".split(),
    "tr": "ve bir bu da de için ile çok daha olan gibi nasıl bebek çocuk".split(),
    "hi_ur_roman": "ka ki ke hai hain aur ko mein se par bachon bache bachay kaise karen".split(),
}


def detect_language(text: str) -> tuple:
    """(language code, method). langdetect when installed, otherwise a script + stop-word heuristic. Short text -> 'und'."""
    t = re.sub(r"https?://\S+|[#@]\w+", " ", text or "")
    letters = sum(ch.isalpha() or unicodedata.category(ch) in ("Mn", "Mc") for ch in t)   # vowel signs of Indic scripts are not 'alpha'
    if letters < 25:
        return "und", "too_short"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(t), "langdetect"
    except ImportError:
        return _heuristic_language(t), "heuristic"
    except Exception:
        return "und", "error"


def _heuristic_language(t: str) -> str:
    counts = {"hi": 0, "ar": 0, "ru": 0, "th": 0, "ko": 0, "ja": 0, "zh": 0, "latin": 0}
    for ch in t:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: counts["hi"] += 1
        elif 0x0600 <= o <= 0x06FF: counts["ar"] += 1
        elif 0x0400 <= o <= 0x04FF: counts["ru"] += 1
        elif 0x0E00 <= o <= 0x0E7F: counts["th"] += 1
        elif 0xAC00 <= o <= 0xD7AF: counts["ko"] += 1
        elif 0x3040 <= o <= 0x30FF: counts["ja"] += 1
        elif 0x4E00 <= o <= 0x9FFF: counts["zh"] += 1
        elif ch.isalpha(): counts["latin"] += 1
    top = max(counts, key=counts.get)
    if top != "latin" and counts[top] >= counts["latin"]:
        return top
    words = re.findall(r"[^\W\d_]+", t.lower())
    scores = {lang: sum(w in set(sw) for w in words) for lang, sw in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "und"


def _count_mask(rx, texts) -> np.ndarray:
    return np.array([bool(rx.search(t)) for t in texts])


def build_features(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    d = df.copy()
    for col in ("title", "description", "hashtags", "tags", "language_clean", "country_clean", "creator_id", "creator_name"):
        d[col] = d[col].fillna("").astype(str) if col in d else ""
    for col in ("duration_seconds", "views", "likes", "comments", "shares", "saves", "creator_followers", "query_count"):
        d[col] = pd.to_numeric(d[col], errors="coerce") if col in d else np.nan
    d["upload_dt"] = pd.to_datetime(d["upload_date"], errors="coerce")
    ref = pd.Timestamp(reference_date) if reference_date is not None else pd.to_datetime(d["retrieved_at_utc"], errors="coerce", utc=True).max()
    ref = ref.tz_localize(None).normalize() if getattr(ref, "tzinfo", None) is not None else (ref.normalize() if pd.notna(ref) else pd.Timestamp.today().normalize())
    d["reference_date"] = ref.date().isoformat()

    d["duration_min"] = d["duration_seconds"] / 60
    d["duration_group"] = pd.cut(d["duration_seconds"], DURATION_BINS, labels=DURATION_LABELS, right=False)
    d["is_short_form"] = d["duration_seconds"] <= 60
    d["upload_year"] = d["upload_dt"].dt.year
    d["upload_month"] = d["upload_dt"].dt.month
    d["upload_year_month"] = d["upload_dt"].dt.strftime("%Y-%m")
    d["age_days"] = (ref - d["upload_dt"]).dt.days.clip(lower=0)

    ok = d["views"] >= 100          # rates of videos with very few views are unstable
    for name, col in (("like_rate", "likes"), ("comment_rate", "comments"), ("share_rate", "shares"), ("save_rate", "saves")):
        d[name] = np.where(ok & d[col].notna(), d[col] / d["views"] * 100, np.nan)
    d["follower_group"] = pd.cut(d["creator_followers"], FOLLOWER_BINS, labels=FOLLOWER_LABELS)
    d["title_chars"] = d["title"].str.len()
    d["title_words"] = d["title"].str.split().str.len()
    d["n_hashtags"] = d["hashtags"].map(lambda s: len([x for x in s.split("|") if x]))
    d["has_description"] = d["description"].str.len() > 0

    texts = [fold(f"{t} {h.replace('|', ' ')} {g.replace('|', ' ')} {b}") for t, h, g, b in zip(d["title"], d["hashtags"], d["tags"], d["description"])]
    inf, new, chi, adu = (_count_mask(rx, texts) for rx in (INFANT, NEWBORN, CHILD, ADULT))
    d["age_infant"], d["age_newborn"], d["age_child"], d["age_adult_mentioned"] = inf, new, chi, adu
    d["age_group"] = np.select(
        [adu & (inf | new | chi), new & ~adu, inf & chi, inf, chi],
        ["all ages (adult + child/infant)", "newborn / neonatal", "infant and child", "infant", "child"], default="not stated in text")
    for k, rx in SUB.items():
        d[f"sub_{k}"] = _count_mask(rx, texts)
    s = {k: d[f"sub_{k}"].to_numpy() for k in SUB}
    prim = np.select(
        [s["choking"] & s["cpr"], s["neonatal"] & (s["cpr"] | s["bls"] | s["rescue_breathing"] | s["chest_compressions"]), s["choking"], s["airway_obstruction"],
         s["aed"], s["bls"], s["rescue_breathing"], s["chest_compressions"], s["cpr"], s["neonatal"]],
        ["combined", "neonatal_resus", "choking", "airway_obstruction", "aed", "bls", "rescue_breathing", "chest_compressions", "cpr", "neonatal_other"], default="unspecified")
    d["subtopic"] = [SUBTOPIC_LABELS[x] for x in prim]

    lang_text = np.where(d["platform"] == "youtube", (d["title"] + ". " + d["description"].str[:600]).str.strip(), d["title"])
    det = [detect_language(t) for t in lang_text]
    d["language_detected"] = [x[0] for x in det]
    d["language_method"] = [x[1] for x in det]
    return d.drop(columns=["upload_dt"])
