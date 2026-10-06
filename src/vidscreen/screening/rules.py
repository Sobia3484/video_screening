"""Transparent keyword rules. Every decision lists the terms that triggered it.

Built from the approved criteria (docs/screening_criteria.md) and tuned on the DEVELOPMENT split only.
Languages covered: English, Spanish, Portuguese, French, Indonesian, Turkish, Russian, Hindi/Urdu (script + Roman), Arabic.
Text is matched after folding Latin diacritics (reanimación -> reanimacion).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import fold

W = r"(?<![^\W_])"  # not preceded by a letter/digit  (underscore allowed as a separator)


def _rx(terms) -> re.Pattern:
    return re.compile("|".join(f"{W}(?:{t})" for t in terms), re.IGNORECASE | re.UNICODE)


# ---- who: infant / child / newborn
CHILD = _rx([
    r"infant\w*", r"bab(?:y|ies)\w*", r"newborn\w*", r"neonat\w*", r"child\w*", r"kids?\b", r"toddler\w*",
    r"p(?:a)?ediatr\w*", r"peds\b", r"paeds\b", r"pals\b", r"nrp\b", r"under\s*(?:1|one)\s*year",
    r"bebe\w*", r"nin[oa]s?\b", r"lactante\w*", r"recien\s*nacid\w*", r"infantil\w*", r"pediatr\w*",
    r"crianca\w*", r"nenem\w*", r"recem[\s-]*nascid\w*", r"enfants?\b", r"nourrisson\w*", r"nouveau[\s-]*ne\w*",
    r"bayi\b", r"anak\b", r"balita\b", r"bebek\w*", r"cocuk\w*",
    r"ребен\w*", r"ребён\w*", r"младен\w*", r"малыш\w*", r"дет(?:ей|ск\w*|и\b|ям\b)", r"новорожд\w*", r"груднич\w*",
    r"bach(?:on|cho|cha|che|chon|chey|ay|ey|a|i)\b", r"bachh?\w*", r"shishu\b", r"nawzaid\w*",
    r"بچ\w*", r"نوزاد\w*", r"बच्च\w*", r"शिशु\w*", r"नवजात\w*", r"طفل\w*", r"اطفال", r"أطفال", r"رضيع\w*", r"مولود\w*",
])

# ---- what: CPR / resuscitation / choking / AED / BLS
TOPIC = _rx([
    r"c\.?p\.?r\w*", r"rcp\b", r"cardiopulmon\w*", r"cardio[\s-]*pulmon\w*", r"resuscit\w*", r"ressuscit\w*", r"reanima\w*",
    r"bls\b", r"pals\b", r"nrp\b", r"nls\b", r"(?:basic\s+)?life\s+support", r"\baed\b", r"defibril\w*", r"desfibril\w*",
    r"chest\s+compression\w*", r"compression\w*", r"compress(?:ao|oes)\b", r"compresion(?:es)?\b",
    r"rescue\s+breath\w*", r"mouth[\s-]*to[\s-]*mouth", r"boca\s+a\s+boca", r"bag[\s-]*(?:valve[\s-]*)?mask", r"ppv\b",
    r"(?:anti)?chok\w*", r"heimlich\w*", r"back\s+blow\w*", r"chest\s+thrust\w*", r"abdominal\s+thrust\w*",
    r"airway\s+obstruct\w*", r"fbao\b", r"foreign\s+(?:body|object)\w*",
    r"atraganta\w*", r"desengasg\w*", r"engasg\w*", r"asfixi\w*",
    r"dam\s+ghut\w*", r"реанимац\w*", r"подавил\w*", r"поперхну\w*", r"искусственн\w+\s+дыхани\w*", r"сердечно",
    r"resusitasi\w*", r"tersedak", r"henti\s+jantung", r"pijat\s+jantung", r"bantuan\s+hidup", r"napas\s+buatan",
    r"resusitasyon\w*", r"kalp\s+masaj\w*", r"temel\s+yasam", r"suni\s+solunum",
    r"إنعاش", r"انعاش", r"اختناق", r"सीपीआर", r"दम\s+घुट\w*", r"گلا\s+گھ\w*",
])
TOPIC_WEAK = _rx([r"first\s+aid", r"emergenc\w*", r"primeros\s+auxilios", r"primeiros\s+socorros", r"life\s*sav\w*",
                  r"sav\w*\s+(?:a\s+|your\s+|their\s+|the\s+)?(?:\w+\s+){0,3}li(?:fe|ves)\b"])
ADULT = _rx([r"adults?\b", r"adulto\w*"])
PET = _rx([r"dogs?\b", r"puppy\b", r"puppies\b", r"cats?\b", r"pets?\b", r"canine\w*", r"feline\w*", r"(?<!stuffed )animals?\b", r"horses?\b"])
PROMO = _rx([
    r"enrol\w*", r"enroll\w*", r"regist(?:er|ration)\w*", r"sign\s*up", r"book\s+(?:now|your|a)", r"class(?:es)?\s+(?:dates?|available|schedule)",
    r"join\s+us", r"workshop", r"promo\s*code", r"discount", r"coupon", r"link\s+in\s+bio", r"spots?\s+left", r"limited\s+spots",
    r"dm\s+(?:us|me)", r"call\s+us", r"visit\s+our", r"order\s+now", r"shop\s+now",
])
INSTRUCT = _rx([
    r"how\s+to", r"step[\s-]*by[\s-]*step", r"steps?\b", r"tutorial\w*", r"demonstrat\w*", r"demo\b", r"training", r"learn\w*",
    r"guide\w*", r"techni\w*", r"what\s+to\s+do", r"procedure\w*", r"skills?\b", r"lesson\w*", r"explain\w*", r"tips?\b",
    r"como\s+", r"paso\s+a\s+paso", r"passo\s+a\s+passo", r"practice\w*", r"manoeuv\w*", r"maneuver\w*", r"maniobra\w*", r"manobra\w*",
])
CERT = _rx([r"certif\w*", r"recertif\w*", r"renewal", r"exam\w*", r"osce\b", r"nursing\s*stud\w*", r"nursingschool", r"student\w*", r"assessment\w*"])
STRONG_INSTRUCT = _rx([r"how\s+to", r"step[\s-]*by[\s-]*step", r"tutorial\w*", r"demonstrat\w*", r"what\s+to\s+do", r"technique\w*", r"procedure\w*", r"cheat\s+sheet", r"paso\s+a\s+paso", r"passo\s+a\s+passo"])


@dataclass
class RuleResult:
    label: str                      # relevant | potentially_relevant | irrelevant
    reason: str
    child: list = field(default_factory=list)
    topic: list = field(default_factory=list)
    topic_gate: bool = False        # at least one strong topic term (used by the hybrid models)


def _hits(rx: re.Pattern, text: str) -> list:
    return sorted({m.group(0).strip() for m in rx.finditer(text)})[:8]


def apply_rules(title: str, description: str, hashtags: str, tags: str) -> RuleResult:
    head = fold(f"{title} {hashtags.replace('|', ' ')} {tags.replace('|', ' ')}")
    body = fold(description or "")
    allt = f"{head} {body}"
    child, topic = _hits(CHILD, allt), _hits(TOPIC, allt)
    gate = bool(topic)
    if not topic:
        weak = bool(TOPIC_WEAK.search(allt))
        if weak and child:
            return RuleResult("potentially_relevant", "child_and_first_aid_only", child, [], False)
        return RuleResult("irrelevant", "no_topic_term", child, [], False)
    if PET.search(allt) and not child:
        return RuleResult("irrelevant", "pet_or_animal", child, topic, gate)
    if child:
        if PROMO.search(allt) and not INSTRUCT.search(allt):
            return RuleResult("potentially_relevant", "promo_without_instruction", child, topic, gate)
        if CERT.search(allt) and not STRONG_INSTRUCT.search(allt):
            return RuleResult("potentially_relevant", "certification_or_exam_without_procedure", child, topic, gate)
        return RuleResult("relevant", "child_and_topic", child, topic, gate)
    if ADULT.search(allt):
        return RuleResult("irrelevant", "adult_only", child, topic, gate)
    return RuleResult("potentially_relevant", "topic_without_age", child, topic, gate)
