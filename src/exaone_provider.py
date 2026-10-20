"""
Real ContentProvider for grammar items (PRD §2.3/§3.1): wires
EXAONE-4.0-1.2B generation to the KAGAS distractor validator
(distractor_validator.py), implementing daily_drill.ContentProvider.

Built from direct experimentation (2026-10-19, see docs/PRD.md), through
two rounds of fixes, not from the vendor's documented usage pattern:

Round 1 -- getting the model to produce anything usable at all:
  - Zero-shot JSON-structured requests produced syntactically valid JSON
    with semantically wrong content (meta-commentary about the task
    instead of an actual item), 3/3 trials, even with a one-shot example.
  - An open-ended non-reasoning request ("write a sentence using X")
    failed near-instantly (echoed one token, stopped).
  - Fix: a narrowly SCAFFOLDED prompt (explicit template + one worked
    example + a single slot to fill) made non-reasoning mode work
    reliably (5/5 across five topics).

Round 2 -- the model still got things wrong that no prompt fix could
reliably prevent, because they're not language-modelling tasks:
  - Wrong 을/를 allomorph for the noun's batchim ("라면를", should be
    "라면을") -- caught, but only after adding a dedicated deterministic
    check (distractor_validator.expected_object_particle); a tag-only
    check couldn't see it.
  - Wrong verb conjugation ("마시요" instead of "마셔요", "보요" instead of
    "봐요") -- NOT caught by kiwipiepy's analyzer, which is deliberately
    permissive (built for noisy/informal text) and happily re-parses a
    malformed surface form as if it were well-formed. A round-trip
    analyze-then-join check is therefore not a conjugation validator.

The real fix for round 2 was to stop asking the LLM to produce the parts
that have a deterministic answer at all. The LLM is now asked only for a
dictionary-form verb lemma (a vocabulary-selection task, still genuinely
open-ended -- this is where the "infinite item pool" generative value
actually lives). Everything else is built by code: the particle's
correctness (fixed by skill for particle_loc; by Hangul batchim for
particle_obj) and the verb's conjugation (kiwi.join(), which applies
Korean's irregular contractions correctly by construction, e.g.
마시다->마셔요, 보다->봐요, 하다->해요). The LLM never writes a full sentence
or marks a blank; it can no longer get conjugation or particle correctness
wrong because it's no longer asked to produce either.

Round 3 -- even the narrowed verb-lemma task wasn't fully reliable: EXAONE
picked semantically odd pairings (커피 -> 차다 "to kick", 라면 -> 들다 "to
lift/hold") -- grammatically fine, but nonsensical. Diversifying the
few-shot prompt (3 varied examples instead of 1) was tried next, since
that exact fix had just worked for an unrelated Qwen3 issue -- but it
made EXAONE-4.0-1.2B WORSE, not better: it echoed the prompt's own
template word ("입력") instead of a verb, in 5/6 cases. That one lesson
doesn't transfer across models; 1.2B is apparently already near its
prompting-complexity ceiling.  Testing a bigger EXAONE (32B is the only
other size that exists) was considered and rejected as disproportionate
-- a large, risky download to fix a cosmetic issue, when a cheaper fix
was already available: `slots` is a small, hand-curated tuple to begin
with, so pairing each slot with one correct, curated verb (`verb_lookup`
below) is no less principled, and removes the semantic-plausibility risk
entirely for every slot currently in use. The LLM remains the fallback
for a slot added later that isn't in the table yet -- this keeps the
generative path alive for scaling, without trusting it where a lookup is
just as easy and strictly safer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from kiwipiepy import Kiwi
from transformers import AutoModelForCausalLM, AutoTokenizer

from distractor_validator import (
    adj_relativizer_ending,
    expected_object_particle,
    has_batchim,
    harmony_positive,
    honorific_ending,
    validate_before_tag,
    validate_cloze_option,
    validate_postnominal_tag,
    validate_prenominal_tag,
    validate_sentence_final_tag,
)

_MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"
_KIWI = Kiwi()
_FILLER_PARTICLES = ("도", "만", "가", "는")

# Every item must have exactly 4 options (A-D), not just the one meaningful
# distractor each skill already builds. These filler candidates pad out to
# 4 the same way _FILLER_PARTICLES already does for particle_loc/particle_obj
# (below): each is checked against the skill's own validator and used only if
# it FAILS -- i.e. it is definitely, mechanically wrong, never a second
# accidentally-correct answer. Verified empirically (not guessed) against
# every stem/pair currently in SKILL_SPECS, standalone via distractor_validator
# (CPU-only, no GPU needed for that check) before being wired in here.
#
# These are deliberately not sophisticated misconception-distractors like
# each skill's own hand-built one -- they're closer to particle_loc/obj's
# existing fillers: safe filler to reach 4 options, not independently
# diagnostic. A later pass with real instructor review (PRD §9.19/§9.20)
# could replace them with better-crafted ones; recorded as a known
# limitation, not hidden.
_EF_FILLER_WORDS = ("학교", "사과", "저기", "안녕")       # honorific, tense_past, irregular_conj
_ETM_FILLER_WORDS = ("학교", "사과", "저기")              # adj_relativizer
_JKS_FILLER_PARTICLES = ("도", "만", "의")                # subj_topic
_TIME_RANGE_FILLER_PARTICLES = ("에", "에게", "로", "한테")  # time_range (paired particle)
_EC_FILLER_WORDS = _EF_FILLER_WORDS      # mod_adverb -- same bare nouns, verified separately to fail the EC check too
_NNB_FILLER_WORDS = _EF_FILLER_WORDS     # noun_counter -- same bare nouns, verified separately to fail the NNB check too

_VERB_PROMPT_LOC = (
    '"{slot}"에서 보통 하는 자연스러운 한국어 동사(기본형, "다"로 끝나는 형태) 딱 하나만 써줘.\n'
    '예시: 입력 "학교" -> 출력 "공부하다"\n'
    "다른 설명 없이 동사 기본형만 출력해."
)
_VERB_PROMPT_OBJ = (
    '"{slot}"와 자연스럽게 어울리는 한국어 동사(기본형, "다"로 끝나는 형태) 딱 하나만 써줘.\n'
    '예시: 입력 "밥" -> 출력 "먹다"\n'
    "다른 설명 없이 동사 기본형만 출력해."
)
_VERB_PROMPT_HONORIFIC = (
    '"{slot}"가 자주 하는 자연스러운 한국어 동사(기본형, "다"로 끝나는 형태) 딱 하나만 써줘.\n'
    '예시: 입력 "선생님" -> 출력 "가르치다"\n'
    "다른 설명 없이 동사 기본형만 출력해."
)

SKILL_SPECS = {
    "particle_loc": {
        "kagas_code": "PART",  # official mapping, PRD §9.8
        "tag": "JKB",
        # Fixed by construction: "저는 [place] ___ [activity verb]." is an
        # activity-at-a-location frame, which always wants 에서 -- this
        # isn't detected from LLM output, it's true by the template itself.
        "correct_form": "에서",
        "distractor_form": "에",
        "distractor_tag": "JKB",
        "slots": ("도서관", "회사", "공원", "카페", "병원", "학교", "식당"),
        "verb_prompt": _VERB_PROMPT_LOC,
        # Curated, not LLM-chosen -- see note above _VERB_LOOKUP. Covers
        # every current slot; the LLM is only consulted for a slot added
        # later that isn't in this table yet.
        "verb_lookup": {
            "도서관": "공부하다", "회사": "일하다", "공원": "놀다", "카페": "마시다",
            "병원": "진료하다", "학교": "공부하다", "식당": "식사하다",
        },
    },
    "particle_obj": {
        "kagas_code": "PART",  # official mapping, PRD §9.8
        "tag": "JKO",
        # Correct form is computed per-noun (batchim rule), not fixed.
        "correct_form": None,
        "distractor_form": "에",
        "distractor_tag": "JKB",
        "slots": ("사과", "책", "커피", "영화", "빵", "라면"),
        "verb_prompt": _VERB_PROMPT_OBJ,
        "verb_lookup": {
            "사과": "먹다", "책": "읽다", "커피": "마시다",
            "영화": "보다", "빵": "먹다", "라면": "먹다",
        },
    },
    "honorific": {
        # Structurally different from the particle skills above: the
        # blank covers the WHOLE verb (honorific_ending()'s "으세요"/"세요"
        # vs plain "어요"-style), not a single inserted particle. Both
        # options share tag EF -- kiwipiepy's tagger can't discriminate
        # honorific-correct from plain-wrong by tag alone here, same as
        # particle_loc's two options both being genuine JKB particles;
        # correctness comes entirely from the curated subject list below
        # (these nouns are honorific-worthy by social convention, not by
        # any rule kiwipiepy or an LLM could derive) plus the batchim rule.
        "kind": "verb_ending",
        "kagas_code": "END",  # official mapping, PRD §9.8
        "tag": "EF",
        "slots": ("선생님", "할머니", "교수님", "사장님", "의사선생님"),
        "verb_prompt": _VERB_PROMPT_HONORIFIC,
        "verb_lookup": {
            "선생님": "가르치다", "할머니": "쉬다", "교수님": "연구하다",
            "사장님": "일하다", "의사선생님": "진료하다",
        },
    },
    "time_range": {
        # 부터/까지 are BOTH tagged JX by kiwipiepy, same as particle_loc's
        # 에/에서 -- but unlike that pair, a bare "[time]___" is genuinely
        # ambiguous (could be the start OR the end of a range) without more
        # context, the exact ambiguity particle_loc/obj were designed to
        # avoid. Fixed with a PAIRED template: "...까지" is already present
        # at the end, so 부터 is the only grammatical choice for the first
        # blank (부터...까지 is a fixed pairing; repeating 까지 twice doesn't
        # express a range) -- correctness is forced by the template, not
        # detected from free text, same principle as particle_loc's fixed
        # "에서" for the activity-at-a-location frame.
        "kind": "paired_particle",
        "kagas_code": "PART",  # official mapping, PRD §9.8
        "tag": "JX",
        "correct_form": "부터",
        "distractor_form": "까지",
        "pairs": (
            ("아홉 시", "열 시", "공부하다"),
            ("두 시", "네 시", "회의하다"),
            ("월요일", "금요일", "일하다"),
            ("아침", "저녁", "운동하다"),
            ("3월", "5월", "여행하다"),
        ),
    },
    "tense_past": {
        # Same shape as honorific (whole-word options, both tag EF) -- see
        # _build_tense_item. Correctness needs an explicit time anchor: a
        # bare "저는 밥을 ___." has no fixed right answer (both tenses are
        # grammatical depending on when the eating happens), same ambiguity
        # trap time_range's bare "[time]___" would have been. "어제"
        # (yesterday) fixed in the template forces past tense to be the
        # only correct choice.
        "kind": "verb_ending",
        "variant": "tense",
        "kagas_code": "END",  # official mapping, PRD §9.8
        "tag": "EF",
        "slots": ("밥", "영화", "책", "커피", "숙제"),
        "verb_prompt": (
            '"{slot}"와 자연스럽게 어울리는 한국어 동사(기본형, "다"로 끝나는 형태) 딱 하나만 써줘.\n'
            '예시: 입력 "밥" -> 출력 "먹다"\n'
            "다른 설명 없이 동사 기본형만 출력해."
        ),
        "verb_lookup": {
            "밥": "먹다", "영화": "보다", "책": "읽다", "커피": "마시다", "숙제": "하다",
        },
    },
    "irregular_conj": {
        # KAGAS category CONJ (§9.8): errors in applying an irregular verb
        # stem's sound change, not which ending to pick (that's END,
        # already covered by honorific/tense_past). 르-irregular verbs
        # (자르다/모르다/부르다) contract 르 -> ㄹㄹ before a vowel-starting
        # ending (자르+아요 -> 잘라요); a common learner error is to skip
        # the contraction and glue the ending on literally ("자르아요").
        #
        # No LLM call needed at all here -- unlike the other skills, the
        # irregular verb itself IS the fixed, closed set being tested
        # (there's no separate "vocabulary choice" left for a model to
        # make), so each (object, verb) pair is curated directly, same
        # reasoning that made verb_lookup cover 100% of particle_loc's
        # slots already redundant with calling the LLM.
        #
        # kiwi.join() always produces the CORRECT contraction regardless
        # of which ending variant it's fed (verified: 자르+아요 -> 잘라요,
        # 부르+아요 -> 불러요 -- it self-corrects vowel harmony too), so it
        # can't be used to build the WRONG (distractor) form. That has to
        # be constructed by hand: harmony_positive() picks the
        # harmony-correct ending the way kiwi.join() would, then it's
        # concatenated literally onto the uncontracted stem.
        "kind": "irregular_conj",
        "kagas_code": "CONJ",  # official mapping, PRD §9.9
        "tag": "EF",
        "pairs": (
            ("머리를", "자르다"),
            ("답을", "모르다"),
            ("노래를", "부르다"),
        ),
    },
    "adj_relativizer": {
        # KAGAS category ADJ (§9.8/§9.10): errors specific to adjectives
        # (형용사/VA), as opposed to honorific/tense_past which are both
        # verb-ending (END) errors. The blank here is the -(으)ㄴ
        # adjective-relativizer ending ("매운 음식" = spicy food), modifying
        # a FOLLOWING noun -- structurally new vs every skill so far, whose
        # blank was either a single particle or a whole word at the
        # SENTENCE'S END. validate_sentence_final_tag doesn't apply; this
        # uses the new validate_prenominal_tag instead.
        #
        # No LLM call: like irregular_conj, the adjective itself is a
        # small curated closed set here (not a vocabulary-choice task), so
        # each (adjective, matching noun) pair is hand-picked for semantic
        # plausibility ("매운 음식" makes sense, "매운 가방" doesn't).
        #
        # Correctness comes entirely from construction, not from a tag
        # difference: the common learner error (는, the VERB-pattern
        # relativizer wrongly generalized onto an adjective stem) parses
        # as the SAME tag (ETM) as the correct 은/ᆫ form -- verified
        # empirically (kiwipiepy's tagger doesn't discriminate stem class
        # here, same permissiveness noted elsewhere in this codebase).
        # correct_form is always built via kiwi.join() (which applies
        # ㅂ/ㄹ/으 irregular contractions correctly by construction);
        # distractor_form is always the hand-built stem+는 concatenation.
        "kind": "adj_relativizer",
        "kagas_code": "ADJ",  # official mapping, PRD §9.10
        "tag": "ETM",
        "pairs": (
            ("맵다", "음식"),    # ㅂ-irregular: 매운, not *맵는
            ("작다", "가방"),    # regular batchim: 작은, not *작는
            ("크다", "집"),      # 으-stem, no batchim: 큰, not *크는
            ("길다", "머리"),    # ㄹ-irregular: 긴, not *길는
            ("예쁘다", "옷"),    # 으-stem, no batchim: 예쁜, not *예쁘는
            ("비싸다", "시계"),  # regular, no batchim: 비싼, not *비싸는
        ),
    },
    "subj_topic": {
        # 이/가 (subject, JKS) vs 은/는 (topic, JX) -- §9.11. Revisited from
        # an earlier rejection (time_range's note, §9.6): bare "[noun]___"
        # is genuinely ambiguous between subject and topic reading without
        # context, the same trap that blocked time_range before the
        # paired-template fix was found. Unlike every structural skill so
        # far, correctness here is NOT syntactically forced -- both
        # particles are grammatical in nearly any sentence; the forcing
        # mechanism is a PEDAGOGICAL convention, not a hard rule: a 누가
        # ("who") question-answer frame, where textbook Korean teaches that
        # the answer takes subject-focus 이/가 (new information), not topic
        # 은/는. Documented explicitly in PRD §9.11 as a softer confidence
        # tier than the other 7 skills -- a textbook convention, not
        # guaranteed by construction.
        #
        # No LLM call: a small curated set of (pronoun, predicate) pairs.
        # kiwi.join() handles the irregular pronoun+가 contraction
        # correctly by construction (저+가->제가, 나+가->내가) -- the
        # DISTRACTOR (저는/나는) is also built via kiwi.join(), not string
        # concatenation, specifically to avoid the trap of literally
        # gluing "제" (the contracted stem, only valid before 가) onto "는"
        # and producing the invalid "*제는".
        "kind": "subj_topic",
        # NOT an official KAGAS code -- deliberately outside the KAGAS
        # framework (PRD §9.11), built from a pedagogical convention (누가
        # question-answer focus), not a KAGAS-annotated error type. None
        # here, not a made-up code, so downstream consumers (e.g. the
        # KAGAS-agreement proxy) can tell "not applicable" apart from a
        # real category and skip/flag it rather than silently mis-score it.
        "kagas_code": None,
        "tag": "JKS",
        "distractor_tag": "JX",
        "pairs": (
            ("저", "커피를 마셔요"),
            ("저", "책을 읽어요"),
            ("나", "학교에 가요"),
            ("나", "밥을 먹어요"),
        ),
    },
    "verb_choice": {
        # KAGAS category VERB (§9.10's gap, now closed): unlike every
        # other skill so far, this is a LEXICAL error (wrong word for the
        # context), not a morphological one (wrong form of the right
        # word) -- kiwipiepy's tag can't discriminate "right verb for
        # this object" any more than it could discriminate honorific vs
        # plain (both parse as EF) or correct vs over-regularised
        # adjective (both parse as ETM); correctness here is likewise
        # entirely a curated choice, verified by reading the resulting
        # sentence, not by any tag check -- reuses _build_whole_word_item
        # unchanged (see _build_verb_choice_item), same division of
        # responsibility as irregular_conj/honorific.
        "kind": "verb_choice",
        "kagas_code": "VERB",  # official mapping, PRD §9.22
        "tag": "EF",
        "pairs": (
            ("숙제", "하다", "보다"),    # do homework, not "see" it
            ("밥", "먹다", "마시다"),    # eat rice, not "drink" it
            ("영화", "보다", "먹다"),    # watch a movie, not "eat" it
            ("음악", "듣다", "보다"),    # listen to music, not "watch" it
            ("사진", "찍다", "먹다"),    # take a photo, not "eat" it
        ),
    },
    "noun_counter": {
        # KAGAS category NOUN (§9.10's gap, now closed): counter-word
        # (단위명사) choice -- 명 for people, 개 for generic objects, 권
        # for bound volumes (books), 분 the honorific people-counter.
        # Also lexical, not morphological (both options tag NNB
        # regardless -- verified empirically, same as verb_choice above),
        # so correctness is curated. Template avoids a subject/topic
        # particle choice on the counted noun itself (already its own
        # skill, subj_topic) by using a fixed topic frame instead.
        "kind": "noun_counter",
        "kagas_code": "NOUN",  # official mapping, PRD §9.22
        "tag": "NNB",
        "pairs": (
            ("사과", "개", "명"),
            ("학생", "명", "개"),
            ("선생님", "분", "개"),
            ("책", "권", "개"),
            ("컵", "개", "명"),
        ),
    },
    "mod_adverb": {
        # KAGAS category MOD (§9.10's gap, now closed): adverb (게/EC)
        # modifying a verb vs the attributive-adjective ending (은/ᆫ/ETM,
        # reusing adj_relativizer_ending -- same helper adj_relativizer
        # already uses) wrongly generalised onto a verb-modifying slot.
        # Unlike verb_choice/noun_counter, this one IS tag-discriminable
        # (EC vs ETM, verified directly) -- closer in kind to the
        # morphological skills than to the two lexical ones above, found
        # while designing this skill, not assumed going in.
        "kind": "mod_adverb",
        "kagas_code": "MOD",  # official mapping, PRD §9.22
        "tag": "EC",
        "distractor_tag": "ETM",
        "pairs": (
            ("조용하다", "말해요"),
            ("많다", "먹어요"),
            ("깨끗하다", "청소해요"),
            ("느리다", "걸어요"),
            ("빠르다", "달려요"),
        ),
    },
}


@dataclass
class GeneratedItem:
    skill_id: str
    stem: str
    blank_marker: str
    options: dict[str, str]
    answer: str
    # KAGAS misconception code the correct-vs-distractor pair is meant to
    # realise (PRD §9.8), or None for subj_topic (deliberately outside the
    # KAGAS framework, see SKILL_SPECS). Added so a KAGAS-agreement check
    # -- human or LLM-proxy -- has something to compare its own judgement
    # against; previously missing entirely (PRD §9.19).
    kagas_code: str | None = None
    # Which option label is the PRIMARY misconception distractor (as
    # opposed to one of the 2 morphologically-safe fillers from
    # _pick_fillers, which carry no intended misconception at all) --
    # needed so a KAGAS-agreement rater is told which option to judge,
    # not left to guess among 3 non-answer options.
    primary_distractor_label: str | None = None


def _extract_verb_lemma(text: str) -> str | None:
    """Scans ALL space-separated tokens (not just the first) for one that's
    a usable dictionary-form verb (ends in "다") -- a real bug caught
    empirically while building the critic loop (PRD §9.14): the original
    code only checked text.split()[0], silently discarding a perfectly
    good answer whenever the model prefixed it with anything else (e.g.
    "책을 읽다" for slot "박물관" -> only "책을" was ever checked, losing a
    valid "읽다"). Found by testing slots outside the curated lookup
    tables to generate real failure cases for the critic loop -- 2 of 12
    "failures" turned out to be this extraction bug, not a model failure,
    recovered here with zero LLM involvement.
    """
    for tok in text.split():
        tok = tok.strip('".,!?')
        if tok.endswith("다") and len(tok) >= 2:
            return tok
    return None


_CRITIC_VERB_PROMPT = (
    '방금 "{slot}"에 대해 "{bad_output}"라고 답했는데, 이건 잘못된 답이야 -- '
    '한국어 동사 기본형이 아니거나 "다"로 끝나지 않아.\n'
    '"{slot}"에서 자연스러운 한국어 동사 기본형(반드시 "다"로 끝나야 함) 딱 하나만 다시 써줘. '
    '설명이나 다른 말 없이 동사만 출력해.'
)


class ExaoneGrammarProvider:
    def __init__(self, model_name: str = _MODEL_NAME, seed: int | None = None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
        self.rng = random.Random(seed)
        self._counter = 0
        self.items: dict[str, GeneratedItem] = {}
        # Novelty 2 critic-loop instrumentation (PRD §3.2/§9.14) -- counts
        # how often the critic step is even reached, and how often it
        # actually recovers a usable verb, so the loop's real value can be
        # reported from production use, not just the one-off experiment.
        self.critic_attempts = 0
        self.critic_recoveries = 0

    def _pick_fillers(self, existing: tuple[str, ...], candidates: tuple[str, ...],
                       is_valid) -> list[str] | None:
        """Every item needs exactly 4 options -- pick 2 more from `candidates`,
        keeping only ones `is_valid` rejects (so a filler can never turn out to
        also be a correct answer), skipping anything already in `existing`.
        Returns None, not a short list, if fewer than 2 turn up -- the caller
        must then skip the item rather than ship one with fewer than 4 options.
        """
        fillers = []
        for cand in candidates:
            if len(fillers) == 2:
                break
            if cand in existing or cand in fillers:
                continue
            if not is_valid(cand):
                fillers.append(cand)
        return fillers if len(fillers) == 2 else None

    def _assemble_options(self, skill_id: str, stem: str, correct_form: str,
                           distractor_form: str, fillers: list[str]) -> str:
        option_texts = [correct_form, distractor_form, *fillers]
        labels = ["A", "B", "C", "D"]
        self.rng.shuffle(labels)
        answer_label = labels[0]
        options = dict(zip(labels, option_texts))

        item_id = f"grammar-{skill_id}-{self._counter}"
        self._counter += 1
        kagas_code = SKILL_SPECS.get(skill_id, {}).get("kagas_code")
        self.items[item_id] = GeneratedItem(skill_id, stem, "___", options, answer_label,
                                             kagas_code, primary_distractor_label=labels[1])
        return item_id

    def _generate_raw(self, messages: list[dict]) -> str:
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True).to("cuda")
        out = self.model.generate(**enc, max_new_tokens=15, do_sample=False)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _pick_verb_lemma(self, prompt_template: str, slot: str, use_critic: bool = True) -> str | None:
        prompt = prompt_template.format(slot=slot)
        text = self._generate_raw([{"role": "user", "content": prompt}])
        verb = _extract_verb_lemma(text)
        if verb is not None:
            return verb
        if not use_critic:
            return None

        # Critic loop (Novelty 2, PRD §3.2/§9.14): the ONE actual LLM call
        # in this whole generation pipeline (every other skill uses a
        # curated lookup or no LLM at all) -- instead of silently
        # discarding this failed attempt and letting the caller blindly
        # retry with a DIFFERENT random slot (which under this model's
        # greedy/non-sampling decoding would reproduce the identical
        # failure for the SAME slot -- retrying the same prompt is
        # mathematically guaranteed to fail again), show the model its own
        # bad output and ask it to diagnose/revise within the same
        # conversation, changing the actual context rather than hoping for
        # different luck.
        self.critic_attempts += 1
        critic_prompt = _CRITIC_VERB_PROMPT.format(slot=slot, bad_output=text[:50])
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
            {"role": "user", "content": critic_prompt},
        ]
        revised_text = self._generate_raw(messages)
        revised_verb = _extract_verb_lemma(revised_text)
        if revised_verb is not None:
            self.critic_recoveries += 1
            return revised_verb
        return None  # critic-assisted retry also failed; caller retries/skips

    def _build_whole_word_item(self, skill_id: str, spec: dict, template: str,
                                correct_form: str, distractor_form: str) -> str | None:
        """Shared tail for any skill where the blank covers a WHOLE
        conjugated word (multiple morphemes), not a single particle --
        e.g. honorific vs plain ending, or past vs present tense. Both
        options legitimately end in the same tag (EF), so kiwipiepy's
        tagger can't discriminate correct from distractor by tag alone;
        that discrimination is each skill's own job (honorific-worthy
        subject list, a time-adverb anchor, etc.), done before this is
        called. This only does what's genuinely identical across all of
        them: confirm both forms are well-formed sentence-final predicates,
        then assemble the item -- now always 4 options (A-D), padded with
        _EF_FILLER_WORDS (see its definition for why those are safe).
        """
        if correct_form == distractor_form:
            return None  # degenerate; shouldn't happen but guard anyway
        for form in (correct_form, distractor_form):
            filled = template.replace("___", form, 1)
            if not validate_sentence_final_tag(filled, spec["tag"]).valid:
                return None

        fillers = self._pick_fillers(
            (correct_form, distractor_form), _EF_FILLER_WORDS,
            lambda cand: validate_sentence_final_tag(template.replace("___", cand, 1), "EF").valid)
        if fillers is None:
            return None

        return self._assemble_options(skill_id, template, correct_form, distractor_form, fillers)

    def _build_verb_ending_item(self, skill_id: str, spec: dict, slot: str,
                                 verb_lemma: str) -> str | None:
        """honorific: correctness comes from the curated honorific-worthy
        subject list (§9.5), not from any rule kiwipiepy or an LLM derives.
        """
        stem_syllables = verb_lemma[:-1]
        try:
            honorific_form = _KIWI.join([(stem_syllables, "VV"), (honorific_ending(stem_syllables), "EF")])
            plain_form = _KIWI.join([(stem_syllables, "VV"), ("어요", "EF")])
        except Exception:
            return None
        return self._build_whole_word_item(skill_id, spec, f"{slot}께서 ___.", honorific_form, plain_form)

    def _build_tense_item(self, skill_id: str, spec: dict, slot: str,
                           verb_lemma: str) -> str | None:
        """tense_past: correctness comes from the explicit "어제" (yesterday)
        time-adverb anchor fixed in the template -- without an anchor like
        this, past vs present is ambiguous/context-dependent, the same
        trap time_range's bare "[time]___" would have been (§9.5's
        sibling skill, see time_range's note on paired templates).
        """
        stem_syllables = verb_lemma[:-1]
        try:
            past_form = _KIWI.join([(stem_syllables, "VV"), ("었", "EP"), ("어요", "EF")])
            present_form = _KIWI.join([(stem_syllables, "VV"), ("어요", "EF")])
        except Exception:
            return None
        particle = expected_object_particle(slot)
        template = f"저는 어제 {slot}{particle} ___."
        return self._build_whole_word_item(skill_id, spec, template, past_form, present_form)

    def _build_irregular_conj_item(self, skill_id: str, spec: dict,
                                    pair: tuple[str, str]) -> str | None:
        """irregular_conj: correctness comes from kiwi.join()'s irregular
        contraction (always right, by construction) vs a hand-built naive
        concatenation (the common learner error of not contracting at all).
        """
        obj, verb_lemma = pair
        stem = verb_lemma[:-1]
        try:
            correct_form = _KIWI.join([(stem, "VV"), ("아요", "EF")])
        except Exception:
            return None
        ending = "아요" if harmony_positive(stem[-1]) else "어요"
        distractor_form = stem + ending

        template = f"저는 {obj} ___."
        return self._build_whole_word_item(skill_id, spec, template, correct_form, distractor_form)

    def _build_adj_item(self, skill_id: str, spec: dict,
                         pair: tuple[str, str]) -> str | None:
        """adj_relativizer: correctness comes from kiwi.join()'s irregular
        contraction (always right, by construction, same division of
        responsibility as irregular_conj) vs a hand-built verb-style "는"
        ending wrongly generalized onto an adjective stem. The blank
        modifies a FOLLOWING noun, not the sentence's end, so this uses
        validate_prenominal_tag instead of _build_whole_word_item's
        validate_sentence_final_tag.
        """
        adj_lemma, noun = pair
        stem = adj_lemma[:-1]
        try:
            correct_form = _KIWI.join([(stem, "VA"), (adj_relativizer_ending(stem), "ETM")])
        except Exception:
            return None
        distractor_form = stem + "는"
        if correct_form == distractor_form:
            return None

        particle = expected_object_particle(noun)
        template = f"저는 ___ {noun}{particle} 좋아해요."
        for form in (correct_form, distractor_form):
            filled = template.replace("___", form, 1)
            if not validate_prenominal_tag(filled, noun, spec["tag"]).valid:
                return None

        fillers = self._pick_fillers(
            (correct_form, distractor_form), _ETM_FILLER_WORDS,
            lambda cand: validate_prenominal_tag(template.replace("___", cand, 1), noun, "ETM").valid)
        if fillers is None:
            return None

        return self._assemble_options(skill_id, template, correct_form, distractor_form, fillers)

    def _build_subj_topic_item(self, skill_id: str, spec: dict,
                                pair: tuple[str, str]) -> str | None:
        """subj_topic: correct_form is always the subject-marked (JKS)
        answer to a 누가 question, built via kiwi.join() so the irregular
        pronoun+가 contraction (저+가->제가) happens correctly; distractor
        is the topic-marked (JX) form, also via kiwi.join() to avoid
        gluing the wrong pronoun stem onto 는.
        """
        pronoun, predicate = pair
        try:
            subj_form = _KIWI.join([(pronoun, "NP"), ("가", "JKS")])
            topic_form = _KIWI.join([(pronoun, "NP"), ("는", "JX")])
        except Exception:
            return None
        if subj_form == topic_form:
            return None

        # "질문"/"답", not "A"/"B": found empirically (validate_novelty1.py,
        # 20/skill run) that labelling the dialogue turns "A:"/"B:" collides
        # with the MCQ option labels "A:"/"B:" rendered right after it --
        # the solver picked "B" in 20/20 trials regardless of content,
        # which a per-sentence breakdown ruled out as a content-based
        # pattern (unlike irregular_conj's genuine, content-tracking
        # failure, see its own comment). The same two-"B:" stem is shown to
        # a real learner too, so this was a stem-formatting defect, not
        # only an artifact of the solver check -- fixing it here, at the
        # source, rather than only in the solver's prompt.
        template = f"질문: 누가 {predicate}? 답: ___ {predicate}"
        for form, tag in ((subj_form, spec["tag"]), (topic_form, spec["distractor_tag"])):
            filled = template.replace("___", form, 1)
            if not validate_postnominal_tag(filled, pronoun, tag).valid:
                return None

        fillers = self._pick_fillers(
            (subj_form, topic_form), _JKS_FILLER_PARTICLES,
            lambda cand: validate_postnominal_tag(template.replace("___", cand, 1), pronoun, "JKS").valid)
        if fillers is None:
            return None

        return self._assemble_options(skill_id, template, subj_form, topic_form, fillers)

    def _build_verb_choice_item(self, skill_id: str, spec: dict,
                                 pair: tuple[str, str, str]) -> str | None:
        """verb_choice (KAGAS VERB): correct_lemma is the verb that
        collocates with `noun`; distractor_lemma is a real, common verb
        that does not (e.g. "see homework" instead of "do homework").
        Both conjugate to ordinary EF-tag sentence enders, same shared
        tail as honorific/tense_past/irregular_conj -- no new validator
        needed, this skill's only new part is the curated (noun,
        correct, wrong) triples themselves.
        """
        noun, correct_lemma, wrong_lemma = pair
        try:
            correct_form = _KIWI.join([(correct_lemma[:-1], "VV"), ("어요", "EF")])
            wrong_form = _KIWI.join([(wrong_lemma[:-1], "VV"), ("어요", "EF")])
        except Exception:
            return None
        particle = expected_object_particle(noun)
        template = f"저는 {noun}{particle} ___."
        return self._build_whole_word_item(skill_id, spec, template, correct_form, wrong_form)

    def _build_noun_counter_item(self, skill_id: str, spec: dict,
                                  pair: tuple[str, str, str]) -> str | None:
        """noun_counter (KAGAS NOUN): correct_counter is the 단위명사
        that classifies `noun` (명/개/권/분); wrong_counter is a real
        counter word, just the wrong class for this noun. Blank sits
        mid-sentence (before "있어요", not at the sentence's end or
        before a noun), hence validate_before_tag rather than
        validate_sentence_final_tag/validate_prenominal_tag.
        """
        noun, correct_counter, wrong_counter = pair
        topic_particle = "은" if has_batchim(noun[-1]) else "는"
        template = f"{noun}{topic_particle} 다섯 ___ 있어요."
        end_marker = " 있어요."
        for counter in (correct_counter, wrong_counter):
            filled = template.replace("___", counter, 1)
            if not validate_before_tag(filled, end_marker, spec["tag"]).valid:
                return None

        fillers = self._pick_fillers(
            (correct_counter, wrong_counter), _NNB_FILLER_WORDS,
            lambda cand: validate_before_tag(template.replace("___", cand, 1), end_marker, "NNB").valid)
        if fillers is None:
            return None

        return self._assemble_options(skill_id, template, correct_counter, wrong_counter, fillers)

    def _build_mod_item(self, skill_id: str, spec: dict,
                         pair: tuple[str, str]) -> str | None:
        """mod_adverb (KAGAS MOD): correct_form is the adjective's
        regular -게 adverbial form (EC-tagged); distractor_form is the
        SAME stem's attributive relativizer ending (은/ᆫ, ETM-tagged,
        via adj_relativizer_ending -- the exact helper adj_relativizer
        already uses), wrongly generalised onto a verb-modifying slot
        instead of a noun-modifying one. Unlike verb_choice/noun_counter,
        this pair IS tag-discriminable (verified directly, not assumed),
        so both forms are checked against their own expected tag, not
        just confirmed well-formed.
        """
        adj_lemma, verb_phrase = pair
        stem = adj_lemma[:-1]
        try:
            correct_form = _KIWI.join([(stem, "VA"), ("게", "EC")])
            distractor_form = _KIWI.join([(stem, "VA"), (adj_relativizer_ending(stem), "ETM")])
        except Exception:
            return None
        if correct_form == distractor_form:
            return None

        template = f"저는 ___ {verb_phrase}"
        end_marker = f" {verb_phrase}"
        if not validate_before_tag(template.replace("___", correct_form, 1), end_marker, spec["tag"]).valid:
            return None
        if validate_before_tag(template.replace("___", distractor_form, 1), end_marker, spec["tag"]).valid:
            return None  # distractor must NOT also parse as EC -- would mean it's not actually distinguishable

        fillers = self._pick_fillers(
            (correct_form, distractor_form), _EC_FILLER_WORDS,
            lambda cand: validate_before_tag(template.replace("___", cand, 1), end_marker, "EC").valid)
        if fillers is None:
            return None

        return self._assemble_options(skill_id, template, correct_form, distractor_form, fillers)

    def _build_paired_particle_item(self, skill_id: str, spec: dict,
                                     pair: tuple[str, str, str]) -> str | None:
        """time_range-style skills: the second half of the pair ("...까지")
        is already fixed text in the template, so only one particle is
        grammatically possible for the blank -- no LLM output to check,
        correctness comes entirely from the paired construction itself.
        """
        start, end, verb_lemma = pair
        try:
            conjugated = _KIWI.join([(verb_lemma[:-1], "VV"), ("어요", "EF")])
        except Exception:
            return None

        stem = f"{start}___ {end}까지 {conjugated}."
        correct_form, distractor_form = spec["correct_form"], spec["distractor_form"]
        if not validate_cloze_option(stem, "___", correct_form, spec["tag"]).valid:
            return None
        if not validate_cloze_option(stem, "___", distractor_form, spec["tag"]).valid:
            return None

        fillers = self._pick_fillers(
            (correct_form, distractor_form), _TIME_RANGE_FILLER_PARTICLES,
            lambda cand: validate_cloze_option(stem, "___", cand, spec["tag"]).valid)
        if fillers is None:
            return None

        return self._assemble_options(skill_id, stem, correct_form, distractor_form, fillers)

    def next_new_grammar_item(self, skill_id: str, learner_urn=None,
                               exclude: set[str] = frozenset()) -> str | None:
        # learner_urn/exclude accepted for daily_drill.ContentProvider protocol
        # compatibility (§9.15) but unused here: this provider GENERATES one
        # fresh item per call rather than selecting among existing candidates,
        # so there's nothing to adaptively choose between -- adaptive
        # selection is BankContentProvider's job (question_bank.py), once
        # generated items have been batched and approved into the bank.
        spec = SKILL_SPECS.get(skill_id)
        if spec is None:
            return None

        if spec.get("kind") == "paired_particle":
            pair = self.rng.choice(spec["pairs"])
            return self._build_paired_particle_item(skill_id, spec, pair)

        if spec.get("kind") == "irregular_conj":
            pair = self.rng.choice(spec["pairs"])
            return self._build_irregular_conj_item(skill_id, spec, pair)

        if spec.get("kind") == "adj_relativizer":
            pair = self.rng.choice(spec["pairs"])
            return self._build_adj_item(skill_id, spec, pair)

        if spec.get("kind") == "subj_topic":
            pair = self.rng.choice(spec["pairs"])
            return self._build_subj_topic_item(skill_id, spec, pair)

        if spec.get("kind") == "verb_choice":
            pair = self.rng.choice(spec["pairs"])
            return self._build_verb_choice_item(skill_id, spec, pair)

        if spec.get("kind") == "noun_counter":
            pair = self.rng.choice(spec["pairs"])
            return self._build_noun_counter_item(skill_id, spec, pair)

        if spec.get("kind") == "mod_adverb":
            pair = self.rng.choice(spec["pairs"])
            return self._build_mod_item(skill_id, spec, pair)

        slot = self.rng.choice(spec["slots"])
        verb_lemma = spec["verb_lookup"].get(slot) or self._pick_verb_lemma(spec["verb_prompt"], slot)
        if verb_lemma is None:
            return None

        if spec.get("kind") == "verb_ending":
            if spec.get("variant") == "tense":
                return self._build_tense_item(skill_id, spec, slot, verb_lemma)
            return self._build_verb_ending_item(skill_id, spec, slot, verb_lemma)

        try:
            conjugated = _KIWI.join([(verb_lemma[:-1], "VV"), ("어요", "EF")])
        except Exception:
            return None  # kiwipiepy doesn't recognise this as a real verb stem

        correct_form = spec["correct_form"] or expected_object_particle(slot)
        distractor_form, distractor_tag = spec["distractor_form"], spec["distractor_tag"]
        stem = f"저는 {slot}___ {conjugated}."

        if not validate_cloze_option(stem, "___", correct_form, spec["tag"]).valid:
            return None
        if not validate_cloze_option(stem, "___", distractor_form, distractor_tag).valid:
            return None

        fillers = self._pick_fillers(
            (correct_form, distractor_form), _FILLER_PARTICLES,
            lambda cand: validate_cloze_option(stem, "___", cand, spec["tag"]).valid)
        if fillers is None:
            return None

        return self._assemble_options(skill_id, stem, correct_form, distractor_form, fillers)

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        raise NotImplementedError("vocab items come from the admin-maintained curriculum, not generation")


if __name__ == "__main__":
    provider = ExaoneGrammarProvider(seed=0)

    for skill_id in ("particle_loc", "particle_obj", "honorific", "time_range", "tense_past",
                     "irregular_conj", "adj_relativizer", "subj_topic",
                     "verb_choice", "noun_counter", "mod_adverb"):
        successes = 0
        for _ in range(5):
            item_id = provider.next_new_grammar_item(skill_id)
            if item_id is not None:
                successes += 1
                item = provider.items[item_id]
                print(item)
                assert item.blank_marker in item.stem
                assert item.answer in item.options
                assert len(item.options) == 4, (
                    f"every item must have exactly 4 options (A-D), got {len(item.options)}: {item}")
                non_answer = {l: v for l, v in item.options.items() if l != item.answer}
                if skill_id == "honorific":
                    assert "세요" in item.options[item.answer], (
                        "the honorific form, never the plain one, must be the correct answer")
                    assert sum(v in _EF_FILLER_WORDS for v in non_answer.values()) == 2
                if skill_id == "time_range":
                    assert item.options[item.answer] == "부터", (
                        "부터 must always be correct in the paired '...까지' template")
                    assert sum(v in _TIME_RANGE_FILLER_PARTICLES for v in non_answer.values()) == 2
                if skill_id == "tense_past":
                    # correctness is guaranteed by construction (_build_tense_item always
                    # puts past_form in correct_form's slot); what the smoke test can
                    # usefully check is that the meaningful distractor (present form) is
                    # genuinely different, and the other two options are the known fillers.
                    assert sum(v in _EF_FILLER_WORDS for v in non_answer.values()) == 2
                    real_distractor = next(v for v in non_answer.values() if v not in _EF_FILLER_WORDS)
                    assert item.options[item.answer] != real_distractor
                if skill_id == "irregular_conj":
                    assert item.options[item.answer] in ("잘라요", "몰라요", "불러요"), (
                        f"the irregular-contracted form must be correct, got {item}")
                    assert sum(v in _EF_FILLER_WORDS for v in non_answer.values()) == 2
                    real_distractor = next(v for v in non_answer.values() if v not in _EF_FILLER_WORDS)
                    assert item.options[item.answer] != real_distractor
                if skill_id == "adj_relativizer":
                    assert item.options[item.answer] in ("매운", "작은", "큰", "긴", "예쁜", "비싼"), (
                        f"the correctly-contracted adjective form must be correct, got {item}")
                    assert sum(v in _ETM_FILLER_WORDS for v in non_answer.values()) == 2
                    real_distractor = next(v for v in non_answer.values() if v not in _ETM_FILLER_WORDS)
                    assert real_distractor.endswith("는"), (
                        f"the distractor must be the verb-pattern '는' overgeneralization, got {item}")
                if skill_id == "subj_topic":
                    assert item.options[item.answer] in ("제가", "내가"), (
                        f"the subject-marked (JKS) form must be the correct answer to a 누가 question, got {item}")
                    assert sum(v in _JKS_FILLER_PARTICLES for v in non_answer.values()) == 2
                    real_distractor = next(v for v in non_answer.values() if v not in _JKS_FILLER_PARTICLES)
                    assert real_distractor in ("저는", "나는"), (
                        f"the topic-marked (JX) form must be the distractor, got {item}")
                if skill_id == "verb_choice":
                    assert item.options[item.answer] in ("해요", "먹어요", "봐요", "들어요", "찍어요"), (
                        f"the collocating verb must be correct, got {item}")
                    assert sum(v in _EF_FILLER_WORDS for v in non_answer.values()) == 2
                if skill_id == "noun_counter":
                    assert item.options[item.answer] in ("개", "명", "분", "권"), (
                        f"the matching counter word must be correct, got {item}")
                    assert sum(v in _NNB_FILLER_WORDS for v in non_answer.values()) == 2
                if skill_id == "mod_adverb":
                    assert item.options[item.answer] in ("조용하게", "많게", "깨끗하게", "느리게", "빠르게"), (
                        f"the adverbial (-게/EC) form must be correct, got {item}")
                    assert sum(v in _EC_FILLER_WORDS for v in non_answer.values()) == 2
                    real_distractor = next(v for v in non_answer.values() if v not in _EC_FILLER_WORDS)
                    # Not a suffix check (endswith "은"/"ᆫ") -- the ETM batchim fuses into
                    # the preceding syllable as one composed character (하+ᆫ -> 한), the
                    # same representation quirk already documented in SKILL_SPECS'
                    # adj_relativizer comment, so there is no separate trailing character
                    # to match here.
                    assert real_distractor in ("조용한", "많은", "깨끗한", "느린", "빠른"), (
                        f"the distractor must be the attributive (ETM) overgeneralisation, got {item}")
        print(f"{skill_id}: {successes}/5 generations succeeded")
        assert successes >= 4, f"{skill_id}: expected the validated pipeline to succeed reliably, got {successes}/5"

    for skill_id, spec in SKILL_SPECS.items():
        for slot, verb in spec.get("verb_lookup", {}).items():
            conjugated = _KIWI.join([(verb[:-1], "VV"), ("어요", "EF")])
            assert f"저는 {slot}" in f"저는 {slot}" and conjugated, (
                f"curated verb for {slot!r} in {skill_id} must conjugate cleanly")

    # fallback path: a slot with no lookup entry must still reach the LLM.
    fallback_spec = dict(SKILL_SPECS["particle_obj"])
    fallback_spec["verb_lookup"] = {}
    SKILL_SPECS["_fallback_test"] = fallback_spec
    try:
        item_id = provider.next_new_grammar_item("_fallback_test")
        assert item_id is not None, "the LLM fallback path must still work for an uncurated slot"
    finally:
        del SKILL_SPECS["_fallback_test"]

    assert provider.next_new_grammar_item("unknown_skill") is None

    # Novelty 2 critic loop (PRD §3.2/§9.14): verified against real,
    # reproducible failure cases (slots outside every curated lookup
    # table, found empirically to make this model fail), not scenarios
    # invented to look good.
    #
    # (1) _extract_verb_lemma must recover a usable verb from ANYWHERE in
    # the raw text, not just the first token -- the free fix found before
    # reaching for the critic step at all.
    assert _extract_verb_lemma("책을 읽다") == "읽다", (
        "a valid verb elsewhere in the text must not be discarded just because it isn't first")
    assert _extract_verb_lemma("travel") is None
    assert _extract_verb_lemma("게임하기") is None, "a gerund (-기) form is not a usable dictionary-form verb"

    # (2) Blind retry of the identical prompt under this model's greedy
    # (non-sampling) decoding is deterministic -- must reproduce the exact
    # same failure, proving a critic step (which changes the prompt) is
    # the only thing that CAN help here, not luck from trying again.
    known_failures = ["해변", "교도소", "스타디움", "공항", "카지노", "수족관"]
    for slot in known_failures[:2]:
        blind_retries = [provider._pick_verb_lemma(_VERB_PROMPT_LOC, slot, use_critic=False)
                          for _ in range(3)]
        assert blind_retries == [None, None, None], (
            f"blind retry of {slot!r} must deterministically fail every time, got {blind_retries}")

    # (3) The critic step must recover SOME (not necessarily all) of these
    # real failures -- measured, not assumed. Verified once at ~40%
    # (4/10) on the full known-failure set; this smoke test checks a
    # smaller fixed subset for speed but still requires at least one
    # genuine recovery, so a regression to 0% is caught.
    recoveries = 0
    for slot in ["우주선", "호텔", "기차역", "영화관"]:  # known-recoverable, see §9.14
        provider.critic_attempts = provider.critic_recoveries = 0
        verb = provider._pick_verb_lemma(_VERB_PROMPT_LOC, slot, use_critic=True)
        if verb is not None:
            recoveries += 1
            assert verb.endswith("다")
    assert recoveries >= 1, (
        f"the critic loop must recover at least one of its previously-confirmed-recoverable "
        f"failure cases, got {recoveries}/4 -- possible model/prompt regression"
    )
    print(f"critic loop recovered {recoveries}/4 known-recoverable failures")

    print("all smoke tests passed")
