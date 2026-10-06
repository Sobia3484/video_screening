# Screening Criteria — Infant & Child CPR and Choking Management Videos

**Version:** 2.0  **Date:** 2026-10-05
**Applies to:** the 1,929 cleaned videos in `data/intermediate/master_cleaned.csv` (YouTube + TikTok)

> **What changed in 2.0:** the published guidelines the criteria rest on are now cited (sections 2, 3, 6 and 11).
> **No screening decision was changed.** The labeled sample and the models were built under versions 1.0 to 1.2, whose decisions are identical to the ones below.

## 1. Purpose

Decide, for every video, whether it is about **Infant & Child CPR and Choking Management**, meaning that it can teach a viewer how to respond when an infant, child or newborn needs resuscitation or is choking.
Only *Relevant* videos enter the final dataset (`data/final/relevant_videos.csv`) used for EDA.

## 2. Basis in published guidelines

The age groups, the topic scope and the meaning of "management" are taken from the 2025 American Heart Association (AHA) and American Academy of Pediatrics (AAP) resuscitation guidelines. Those guidelines incorporate the evidence reviews of the International Liaison Committee on Resuscitation (ILCOR) [1].

| Criterion in this document | Source | Where |
|---|---|---|
| Infant = younger than about 1 year; child = about 1 year until puberty; from puberty adult BLS applies | AHA/AAP 2025 Part 6 [1] | section 4.1, Scope of the Guidelines |
| Puberty is defined for teaching purposes as breast development in females and axillary hair in males | AHA/AAP 2025 Part 6 [1] | section 4.1 |
| Newborn infants at birth are covered by a separate guideline (neonatal resuscitation); pediatric BLS may be applied to newborns under 28 days | AHA/AAP 2025 Part 6 [1] and Part 5 [2] | section 4.1 of [1] |
| Pediatric BLS topics: starting CPR, high-quality CPR, chest compression technique, opening the airway, AED / defibrillation, breaths for inadequate breathing, foreign body airway obstruction (FBAO) | AHA/AAP 2025 Part 6 [1] | abstract and sections 6 to 8 |
| Choking (FBAO): mild (coughing, can make sounds) versus severe; response for infants is repeated cycles of 5 back blows and 5 chest thrusts, for children 5 back blows and 5 abdominal thrusts, and CPR if the victim becomes unresponsive | AHA/AAP 2025 Part 6 [1] | section 6.3 |
| Rescue breaths: respiratory causes are the major cause of cardiac arrest in infants and children, and breaths in addition to compressions improve survival in out-of-hospital arrest | AHA/AAP 2025 Part 6 [1] | Top 10 take-home messages |
| AED in children: use a pediatric attenuator for children under 8 years if available | AHA/AAP 2025 Part 6 [1] | section 8.2 |
| Neonatal resuscitation: care of the newborn, from normal transition to assisted ventilation, oxygen, and advanced resuscitation | AHA/AAP 2025 Part 5 [2] | abstract |
| Pediatric advanced life support (PALS) is a separate guideline for children's resuscitation by trained providers | AHA/AAP 2025 Part 8 [3] | title and scope |
| Guidelines address lay rescuers and health professionals, so no audience restriction is applied | AHA/AAP 2025 Part 6 [1] | abstract |
| Prevention is a separate link of the pediatric chain of survival from the response to an arrest or choking | AHA/AAP 2025 Part 6 [1] | section 5.1, The Pediatric Chain of Survival |
| Adult resuscitation is covered by separate adult guidelines | AHA 2025 Part 7 [4] | title |

Age definitions differ slightly between councils: for example the 2025 Korean CPR guidelines define a child as 1 to under 8 years [5]. This project adopts the broader AHA/AAP definition (1 year until puberty).

## 3. Definitions used

* **Infant:** under about 1 year. **Child:** about 1 year until puberty. **Newborn / neonate:** a baby at or soon after birth, whose resuscitation is covered by the neonatal resuscitation guideline [2]. **All ages:** a video that covers adults, children and infants counts, because it contains child and infant content.
* **CPR / BLS topics:** starting CPR, chest compressions, rescue breaths / ventilation, opening the airway, AED use, basic life support, pediatric advanced life support when it teaches resuscitation.
* **Choking:** foreign body airway obstruction and its first-aid response (back blows, chest or abdominal thrusts, Heimlich manoeuvre, CPR when unresponsive).
* **Management:** the response, meaning what to do. Prevention, recognition of other illnesses, and stories about an event are not management.

## 4. Labels

| Label | Meaning | Final? |
|---|---|---|
| **Relevant** | Meets all three conditions in section 5 | Yes |
| **Irrelevant** | Fails at least one condition | Yes |
| *Potentially relevant* | Used only during labeling when the text could not decide; every such video was opened and resolved to Relevant or Irrelevant | No (temporary) |

## 5. A video is Relevant when ALL three hold

1. **Age group:** infant, child or newborn (all-ages videos count), as defined in section 3.
2. **Topic:** CPR, choking / FBAO, BLS, AED use, chest compressions, rescue breathing, or neonatal / pediatric resuscitation, as listed in section 2.
3. **Content:** the video teaches, demonstrates or explains what to do.

## 6. Decisions on borderline cases

Column "Basis" says whether the decision follows a guideline or is a decision of the author.

| Case | Decision | Basis |
|---|---|---|
| Prevention only (how to avoid choking, no response shown) | Irrelevant | Prevention is a different link from the response [1]; the project topic is management (author's scope decision) |
| Prevention plus what to do if choking happens | Relevant | The response is present [1] |
| News / drama / reaction with a story only | Irrelevant | Author |
| News / drama where the procedure is explained or demonstrated | Relevant | Author; content condition |
| Mixed video (long first-aid course) with a meaningful instructional infant/child CPR or choking part | Relevant | Author |
| Neonatal resuscitation, including professional training | Relevant | Neonatal guideline [2]; inclusion decided by the author |
| Adult CPR only | Irrelevant | Adult guidelines [4] |
| Pet / animal CPR | Irrelevant | Human guidelines only (author) |
| Advert or course promotion with no procedure shown | Irrelevant | Author; content condition |
| Demonstration on a manikin or doll | Relevant | Author; counts as demonstration |
| PALS when it actually teaches child resuscitation | Relevant | PALS guideline [3]; an advert without teaching is Irrelevant |
| Baby topics unrelated to CPR / choking (feeding, sleep, general health) | Irrelevant | Outside the topics in section 2 |
| Language | No restriction | Author |

**Operational rule for mixed videos** (so it can be applied from text): title, description, hashtags or tags explicitly mention infant/child CPR or choking instruction -> Relevant; only a generic course title -> opened and resolved during labeling.

## 7. Information used when labeling

Title, description, hashtags, tags, creator, duration and the video link. Transcripts are **not** used in screening, so YouTube and TikTok are judged on the same information.

## 8. Subtopics (relevant videos, used in EDA)

`cpr` (starting CPR, technique) · `neonatal` [2] · `bls` · `choking` and `airway_obstruction` (FBAO, section 6.3 of [1]) · `chest_compressions` · `aed` (section 8 of [1]) · `rescue_breathing` · `combined` (CPR + choking).

## 9. Sampling, models and evaluation (fixed before any rule was written)

* **Sample:** 300 of the 1,929 videos, 150 per platform, stratified by platform x query-subtopic, random within strata, seed 42; each video has a `sampling_weight`.
* **Split:** 195 development and 105 test videos, assigned before any rule or model existed.
* **Candidates:** keyword rules, TF-IDF + logistic regression, multilingual sentence embeddings + logistic regression, and hybrids with a rule gate.
* **Selection:** 5-fold stratified cross-validation on the development split; threshold maximises F1 of Relevant on out-of-fold scores (no mandatory manual review step).
* **Evaluation:** precision, recall, F1, AUC and confusion counts on the test split with bootstrap 95% confidence intervals, per platform and population-weighted.
* **Reporting flow** follows the PRISMA 2020 idea of reporting how many records were identified, removed and included at each stage [6].
* The 300 human labels are final. For the other videos the selected method decides; videos near the threshold or disagreeing with the rules are listed as an optional audit list.

## 10. Limits of these criteria

* They judge **relevance to the topic, not the correctness or quality** of the medical advice shown. The 2025 update, for example, removed the two-finger technique for infant chest compressions in favour of the one-hand or two-thumb encircling technique [1]; earlier videos may still teach the older technique and are still counted as relevant.
* The guideline basis is the 2025 AHA/AAP set. Other councils (for example the European Resuscitation Council) use similar but not identical age definitions and techniques; this document does not claim equivalence with them.
* The rule-based keyword lexicon is author-constructed (see `docs/rules_lexicon.md`); it is not a published list.

## 11. References

1. Joyner BL Jr, Dewan M, Bavare A, et al. Part 6: Pediatric Basic Life Support: 2025 American Heart Association and American Academy of Pediatrics Guidelines for Cardiopulmonary Resuscitation and Emergency Cardiovascular Care. *Circulation*. 2025;152(16_suppl_2):S424-S447. doi:10.1161/CIR.0000000000001370. Co-published in *Pediatrics*. https://cpr.heart.org/en/resuscitation-science/cpr-and-ecc-guidelines/pediatric-basic-life-support (accessed 2026-10-05).
2. Part 5: Neonatal Resuscitation: 2025 American Heart Association and American Academy of Pediatrics Guidelines for Cardiopulmonary Resuscitation and Emergency Cardiovascular Care. *Circulation*. 2025;152(16_suppl_2):S385-S423. doi:10.1161/CIR.0000000000001367. Also *Pediatrics*. 2026;157(1):e2025074352. doi:10.1542/peds.2025-074352. https://cpr.heart.org/en/resuscitation-science/cpr-and-ecc-guidelines/neonatal-resuscitation (accessed 2026-10-05).
3. Part 8: Pediatric Advanced Life Support: 2025 American Heart Association and American Academy of Pediatrics Guidelines for Cardiopulmonary Resuscitation and Emergency Cardiovascular Care. *Circulation*. 2025;152(16_suppl_2). [Add the page range and DOI from https://cpr.heart.org/en/resuscitation-science/cpr-and-ecc-guidelines/pediatric-advanced-life-support before submitting.]
4. Part 7: Adult Basic Life Support: 2025 American Heart Association Guidelines for Cardiopulmonary Resuscitation and Emergency Cardiovascular Care. *Circulation*. 2025;152(suppl 2). [Add the page range and DOI from https://cpr.heart.org/en/resuscitation-science/cpr-and-ecc-guidelines/adult-basic-life-support.]
5. 2025 Korean Guidelines for Cardiopulmonary Resuscitation: Part 7. Pediatric basic life support. *Clinical and Experimental Emergency Medicine*. https://www.ceemjournal.org/journal/view.php?number=684 (accessed 2026-10-05).
6. Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*. 2021;372:n71. doi:10.1136/bmj.n71.