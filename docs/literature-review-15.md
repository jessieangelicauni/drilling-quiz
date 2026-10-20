# Literature Review — 15 Sources (2024–2026) & Research Gaps

**Scope:** Personalized Korean drill-quiz generation — LLM item generation,
psychometric item/learner calibration, and Korean L2 learning technology.
**Selection criteria:** published 2024–2026, from IEEE, Springer, Elsevier,
Wiley, Taylor & Francis, or SAGE. No arXiv, no MDPI, no preprint-only
sources. All 15 are drawn from the 41-reference list already verified
against the Crossref API for `paper/KO-DRILL_IEEE.docx`.
**Note on "open access":** publisher reputability and peer-review status
were verified for every entry; gold/hybrid open-access status was not
independently re-confirmed per article (several Elsevier/Wiley/SAGE venues
are hybrid) and should be checked against the publisher page if OA is a
hard requirement for submission.

**Note on link verification:** all 15 DOIs below were resolved against
`https://doi.org/` and confirmed to redirect to a live publisher page — 7 resolve directly; the other 8 (Wiley/SAGE/Taylor & Francis) return HTTP 403 to automated tools because those publishers block scripted requests, not because the link is broken — the DOI redirect itself succeeds (confirmed via the redirect `Location` header), and a normal browser click reaches the article.

---

## A. Psychometric Calibration — Why Elo Fails for Generated Items

### 1. Keeping Elo Alive: Evaluating and Improving Measurement Properties of Learning Systems Based on Elo Ratings
M. Bolsinova, B. Gergely, M. J. S. Brinkhuis — *British Journal of Mathematical and Statistical Psychology*, 79(1), 95–110, Feb. 2026. Wiley.
**Link:** https://doi.org/10.1111/bmsp.12395

Proves analytically and by simulation that Elo's rating variance inflates
without bound when item selection is adaptive *and* item difficulty is
calibrated dynamically at the same time — exactly the regime an LLM
generating items on demand forces. Proposes Parallel Elo as a lightweight
fix. **This is the single most load-bearing citation in the paper's design
rationale (Section III-B).**

### 2. Toward Psychometric Learning Analytics: Augmenting the Urnings Algorithm with Response Times
B. Gergely, M. Bolsinova — *Journal of Educational Measurement*, 63(1), Mar. 2026. Wiley.
**Link:** https://doi.org/10.1111/jedm.70026

Extends the Urnings tracker (the fix for gap above) with response-time
data, from the same research lineage that operates Math Garden — direct
evidence this isn't an outsider's critique of Elo but a refinement by the
team that runs one of the largest Elo-based adaptive-learning deployments.

---

## B. Can an LLM Judge Its Own Difficulty? — No

### 3. Item Difficulty Modeling Using Fine-Tuned Small and Large Language Models
M. Li, H. Jiao, T. Zhou, N. Zhang, S. Peters, R. W. Lissitz — *Educational and Psychological Measurement*, 85(6), 1065–1090, Dec. 2025. SAGE.
**Link:** https://doi.org/10.1177/00131644251344973

Fine-tuned BERT/RoBERTa-class encoders (hundreds of millions of
parameters) *beat* much larger LLMs at predicting item difficulty from
text, winning a shared-task benchmark. Direct empirical support for using
a small KLUE-RoBERTa encoder rather than an LLM for the difficulty-prior
role.

### 4. Leveraging LLM Respondents for Item Evaluation: A Psychometric Analysis
Y. Liu, S. Bhandari, Z. A. Pardos — *British Journal of Educational Technology*, 56(3), 1028–1052, May 2025. Wiley.
**Link:** https://doi.org/10.1111/bjet.13570

Simulates test-takers with LLMs and calibrates item parameters from the
responses: a single LLM's item-level parameters correlate well with
human-calibrated ones (Spearman above 0.8 for the strongest model
tested), but no single model reproduces the full human ability
distribution — each model's simulated population is narrower and less
variable than real test-takers, closed only by pooling several models.
**Not** "systematic divergence from human data" (an earlier draft of
this review mischaracterised it that way, corrected 2026-10-20 after
re-checking the source directly). The paper's actual use of this
source (Section I) is narrower than "LLMs can't judge difficulty": a
single LLM is not a population of learners, so nothing here licenses
one model self-rating its own generated item's difficulty.

---

## C. Korean-Specific NLP and Learner Data

### 5. Enriching the Korean Learner Corpus for Grammatical Error Correction and Writing Assessment
J. Song, K. Lim, J. Park — *Language Resources and Evaluation*, 60(1), Mar. 2026. Springer.
**Link:** https://doi.org/10.1007/s10579-025-09882-9

The most recent, largest error-annotated Korean learner corpus. Source
material for grounding a misconception taxonomy (used alongside KAGAS,
ACL 2023) in real, attested learner errors rather than invented categories.

### 6. KRongBERT: Enhanced Factorization-Based Morphological Approach for the Korean Pretrained Language Model
H. Yu, Y. Cho, G. Park, M. Kim — *Information Processing & Management*, 62(3), 104072, May 2025. Elsevier.
**Link:** https://doi.org/10.1016/j.ipm.2025.104072

Shows that explicitly modeling Korean's agglutinative morphology inside a
pretrained encoder improves downstream performance over a generic
multilingual tokenizer — direct justification for why a Korean-first model
(EXAONE) and a Korean-pretrained encoder (KLUE-RoBERTa) are not
interchangeable with a general multilingual LLM for this task.

---

## D. Generative AI in Korean / L2 Learning — Current Practice and Its Limits

### 7. Enhancing Korean as a Foreign Language Learning Through a Mobile-Assisted Blended Course: A Triarchic and Motivational Perspective
S.-H. Hong, S. C. Yang — *Language Teaching Research*, Sep. 2025. SAGE.
**Link:** https://doi.org/10.1177/13621688251368643

Recent empirical KFL (Korean-as-foreign-language) study reporting gains
from mobile-assisted blended instruction — but the system is not
adaptive/personalized at the item level, which is precisely the gap this
project targets.

### 8. Ethically Integrated Generative AI for Reading and Vocabulary Development in Higher Education
S. Rashid, S. Malik, F. Ghauri — *Asian-Pacific Journal of Second and Foreign Language Education*, 11(1), Sep. 2026. Springer (SpringerOpen, gold OA).
**Link:** https://doi.org/10.1186/s40862-026-00451-3

Experimental study of generative-AI-supported vocabulary/reading
development; representative of the current ceiling of generative-AI
language-learning research — content generation and delivery, but without
psychometric calibration of what is generated.

### 9. Impact of Generative AI Chatbots and Interaction Modes on the Speaking Proficiency of Adolescent EFL Learners
T.-Y. Tai, H. H.-J. Chen — *Computer Assisted Language Learning*, 1–30, Oct. 2025. Taylor & Francis.
**Link:** https://doi.org/10.1080/09588221.2025.2572999

Evaluates generative-AI chatbot interaction for L2 speaking; useful
counter-example showing generative AI research in language learning
currently concentrates on conversational practice, not structured,
difficulty-calibrated assessment items.

### 10. LLM-Generated Feedback Supports Learning if Learners Choose to Use It
D. R. Thomas, C. Borchers, S. Bhushan, E. Gatz, S. Gupta, K. R. Koedinger — *Lecture Notes in Computer Science*, vol. 16063, Springer, 2026.
**Link:** https://doi.org/10.1007/978-3-032-03870-8_33

Across 2,600+ lesson completions and 885 learners, on-demand LLM
explanatory feedback produced statistically significant learning gains in
only **two of seven** lessons after adjusting for selection bias. Direct
empirical motivation for the uncertainty-gated escalation *policy* (paper
Section IV-D) — uniform LLM feedback is not uniformly worth its cost — but
that policy is reported as an application decision, not promoted to a
research novelty, so this source motivates a design choice, not one of
the paper's two novelty claims.

**Correction (2026-10-20):** this entry previously cited a different
paper — J. Lin, S. Y. Zhao, A. Gurung, X. Meng & K. R. Koedinger,
"Retrieval-Augmented Lesson Generation for Scenario-Based Tutor
Training," *Journal of Computer Assisted Learning*, 42(6), Dec. 2026
(doi: 10.1002/jcal.70343) — under this same slot, with the 885-learner/
2-of-7-lesson finding wrongly attributed to it. That paper is about
generating training materials for human tutors, not measuring feedback
effectiveness on learners, and is not cited anywhere else in this
review or the paper's reference list once removed from this slot.

---

## E. Automatic Item and Distractor Generation — State of the Art

### 11. Transformer and Large Language Models for Automatic Multiple-Choice Question Generation: A Systematic Literature Review
H. W. Awalurahman, R. F. Aji, I. Budi — *IEEE Access*, 13, 127100–127112, 2025. IEEE (gold OA).
**Link:** https://doi.org/10.1109/ACCESS.2025.3590423

Recent systematic review confirming the field evaluates generated
questions mostly on fluency/surface metrics, not on psychometric or
pedagogical usefulness — general context for why automatic item
generation research hasn't closed the diagnosis gap (G1) or the
tracker-validity gap (G2) below on its own.

### 12. Distractor Generation for Multiple-Choice Questions with Predictive Prompting and Large Language Models
S. K. Bitew, J. Deleu, C. Develder, T. Demeester — in *Communications in Computer and Information Science*, Springer, 2025, pp. 48–63. (Springer-published peer-reviewed proceedings volume, not a standalone journal — flagged for transparency.)
**Link:** https://doi.org/10.1007/978-3-031-74627-7_4

State-of-the-art distractor generation via predictive prompting — but only
~53% of generated distractors were rated high-quality by teachers.
Distractors here are optimized purely for *plausibility*, never for
diagnosing *which* misconception a wrong answer reveals — the gap
Novelty 1 (misconception-conditioned distractors) directly addresses.

### 13. Efficient Unified Framework for Difficulty-Controllable Question Generation
X. Dong, Z. Li — *IEEE Access*, 14, 76358–76371, 2026. IEEE (gold OA).
**Link:** https://doi.org/10.1109/ACCESS.2026.3693277

Combines active-learning labeling with LoRA-based prompting to steer
generation toward a requested difficulty. One of the few papers to treat
difficulty control as a first-class generation objective rather than an
afterthought — but still relies on the generator's own difficulty
judgment, the same self-rating weakness source [4] documents directly;
general context, not the closest prior work to either of this paper's
two novelties specifically (an earlier draft of this review called it
the closest prior work to a difficulty-calibration novelty that has
since been removed from the paper entirely, see §9.18 in the project's
own decision log; corrected here rather than left pointing at something
that no longer exists).

### 14. Automatic Question-Answer Pairs Generation Using Pre-Trained Large Language Models in Higher Education
J. Ling, M. Afzaal — *Computers and Education: Artificial Intelligence*, 6, 100252, Jun. 2024. Elsevier.
**Link:** https://doi.org/10.1016/j.caeai.2024.100252

Establishes feasibility of LLM-generated QA pairs for higher-education
assessment at scale — representative baseline of "generate without
calibrate," general context for the same self-rating weakness [4]
documents directly; not tied to a specific numbered gap below.

### 15. QuerIA: Adaptive Question Generation and Evaluation in Higher Education Using Large Language Models and Contextual Learning
C. Badenes-Olmedo, P. Eyzaguirre — *Expert Systems with Applications*, 301, 130140, Mar. 2026. Elsevier.
**Link:** https://doi.org/10.1016/j.eswa.2025.130140

Among the most recent systems coupling generation with evaluation —
evaluation here means output quality checking, not psychometric difficulty
calibration against a learner-ability model, which is the distinction this
paper's contribution turns on.

---

## Research Gaps Identified

**Renumbered 2026-10-20** to match the paper's current state: an earlier
draft of both this review and the paper itself had a fourth, first gap
("generation and calibration studied as two separate problems") that
motivated a difficulty-calibration novelty since removed from the paper
entirely (PRD §9.18 — tested directly, the core claim did not hold).
That gap is dropped here too, not kept as an orphaned entry pointing at
a novelty that no longer exists. The paper now argues two novelties
against three gaps, not three against four; G1–G3 below are renumbered
to match the paper's own G1–G3 one-to-one, each evidenced by at least
two independent, recent (2025–2026) sources above, not asserted from a
single paper.

### G1 — Distractors are optimized for plausibility, not diagnosis
Evidenced by: [12] (predictive prompting improves plausibility, ceiling
~53% teacher-rated quality; no diagnostic labeling of *why* a distractor is
wrong). Cross-checked against Korean-specific error data in [5] — a
taxonomy of real, attested Korean learner errors exists and is
under-exploited by the distractor-generation literature.

### G2 — Psychometric rating systems are deployed outside their proven validity conditions
Evidenced by: [1] (analytically proves Elo's specific failure mode:
simultaneous adaptive selection + dynamic item calibration), [2] (the fix,
from the same research lineage that runs a major Elo-based deployment).
**No source in the Korean-/L2-learning literature set [7]–[10] uses
Urnings or any Elo alternative — all assume either a fixed pre-calibrated
bank or do not address item-level adaptivity at all.**

### G3 — LLM-generated feedback is applied uniformly though its benefit is not
Evidenced by: [10] (significant learning gains in only 2 of 7 lessons,
quantified with propensity-adjusted effect sizes of 0.28–0.33). Cross-
checked against the broader generative-AI-in-Korean/L2 set [7]–[9], none
of which gate or ration feedback generation by measurement uncertainty —
feedback is generated the same way for every learner, every time. This
gap motivates the uncertainty-gated escalation *policy* in the paper
(Section IV-D), reported as an application decision rather than a third
research novelty — the policy's own width-tracks-error premise was
tested directly and did not hold at the urn size this system uses
(validate_novelty2.py, corr(width, true error) ≈ −0.76 to −0.85
depending on urn size), so the paper deliberately claims only that the
policy targets observable-record ambiguity, not absolute measurement
error.

**Net observation:** every gap sits at the *intersection* of two literatures
that currently don't cite each other — psychometrics (A, B) and generative
NLP for education (D, E) — plus a Korean-specific data gap (C) that neither
literature currently draws on. That intersection is where this paper's
two novelties (Section IV of `paper/KO-DRILL_IEEE.docx`) are
positioned — misconception-conditioned distractors (G1) and Urnings'
demonstrated robustness to an uncalibrated item-difficulty prior (G2).
G3 is addressed by an application policy, not claimed as a third
novelty (see above).

---

*Full 41-reference list with verified Crossref metadata: see the References
section of `paper/KO-DRILL_IEEE.docx`. This file is a curated subset for
quick literature-review reference, not a replacement for it.*
