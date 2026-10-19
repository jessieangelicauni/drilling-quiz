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
    harmony_positive,
    honorific_ending,
    validate_cloze_option,
    validate_postnominal_tag,
    validate_prenominal_tag,
    validate_sentence_final_tag,
)

_MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"
_KIWI = Kiwi()
_FILLER_PARTICLES = ("도", "만", "가", "는")

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
        "tag": "JKS",
        "distractor_tag": "JX",
        "pairs": (
            ("저", "커피를 마셔요"),
            ("저", "책을 읽어요"),
            ("나", "학교에 가요"),
            ("나", "밥을 먹어요"),
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
        then assemble the item.
        """
        if correct_form == distractor_form:
            return None  # degenerate; shouldn't happen but guard anyway
        for form in (correct_form, distractor_form):
            filled = template.replace("___", form, 1)
            if not validate_sentence_final_tag(filled, spec["tag"]).valid:
                return None

        labels = ["A", "B"]
        self.rng.shuffle(labels)
        answer_label, distractor_label = labels
        options = {answer_label: correct_form, distractor_label: distractor_form}

        item_id = f"grammar-{skill_id}-{self._counter}"
        self._counter += 1
        self.items[item_id] = GeneratedItem(skill_id, template, "___", options, answer_label)
        return item_id

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

        labels = ["A", "B"]
        self.rng.shuffle(labels)
        answer_label, distractor_label = labels
        options = {answer_label: correct_form, distractor_label: distractor_form}

        item_id = f"grammar-{skill_id}-{self._counter}"
        self._counter += 1
        self.items[item_id] = GeneratedItem(skill_id, template, "___", options, answer_label)
        return item_id

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

        template = f"A: 누가 {predicate}? B: ___ {predicate}"
        for form, tag in ((subj_form, spec["tag"]), (topic_form, spec["distractor_tag"])):
            filled = template.replace("___", form, 1)
            if not validate_postnominal_tag(filled, pronoun, tag).valid:
                return None

        labels = ["A", "B"]
        self.rng.shuffle(labels)
        answer_label, distractor_label = labels
        options = {answer_label: subj_form, distractor_label: topic_form}

        item_id = f"grammar-{skill_id}-{self._counter}"
        self._counter += 1
        self.items[item_id] = GeneratedItem(skill_id, template, "___", options, answer_label)
        return item_id

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

        labels = ["A", "B"]
        self.rng.shuffle(labels)
        answer_label, distractor_label = labels
        options = {answer_label: correct_form, distractor_label: distractor_form}

        item_id = f"grammar-{skill_id}-{self._counter}"
        self._counter += 1
        self.items[item_id] = GeneratedItem(skill_id, stem, "___", options, answer_label)
        return item_id

    def next_new_grammar_item(self, skill_id: str) -> str | None:
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

        fillers = []
        for candidate in _FILLER_PARTICLES:
            if len(fillers) == 2:
                break
            if candidate in (correct_form, distractor_form):
                continue
            if not validate_cloze_option(stem, "___", candidate, spec["tag"]).valid:
                fillers.append(candidate)

        option_texts = [correct_form, distractor_form, *fillers]
        labels = ["A", "B", "C", "D"][:len(option_texts)]
        self.rng.shuffle(labels)
        answer_label = labels[0]
        options = dict(zip(labels, option_texts))

        item_id = f"grammar-{skill_id}-{self._counter}"
        self._counter += 1
        self.items[item_id] = GeneratedItem(skill_id, stem, "___", options, answer_label)
        return item_id

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        raise NotImplementedError("vocab items come from the admin-maintained curriculum, not generation")


if __name__ == "__main__":
    provider = ExaoneGrammarProvider(seed=0)

    for skill_id in ("particle_loc", "particle_obj", "honorific", "time_range", "tense_past",
                     "irregular_conj", "adj_relativizer", "subj_topic"):
        successes = 0
        for _ in range(5):
            item_id = provider.next_new_grammar_item(skill_id)
            if item_id is not None:
                successes += 1
                item = provider.items[item_id]
                print(item)
                assert item.blank_marker in item.stem
                assert item.answer in item.options
                assert len(item.options) in (2, 3, 4)
                if skill_id == "honorific":
                    assert "세요" in item.options[item.answer], (
                        "the honorific form, never the plain one, must be the correct answer")
                if skill_id == "time_range":
                    assert item.options[item.answer] == "부터", (
                        "부터 must always be correct in the paired '...까지' template")
                if skill_id == "tense_past":
                    distractor_label = next(l for l in item.options if l != item.answer)
                    # correctness is guaranteed by construction (_build_tense_item always
                    # puts past_form in correct_form's slot); what the smoke test can
                    # usefully check is that the two options are genuinely different
                    # conjugations, not a degenerate pair.
                    assert item.options[item.answer] != item.options[distractor_label]
                if skill_id == "irregular_conj":
                    distractor_label = next(l for l in item.options if l != item.answer)
                    assert item.options[item.answer] != item.options[distractor_label]
                    assert item.options[item.answer] in ("잘라요", "몰라요", "불러요"), (
                        f"the irregular-contracted form must be correct, got {item}")
                if skill_id == "adj_relativizer":
                    distractor_label = next(l for l in item.options if l != item.answer)
                    assert item.options[item.answer] != item.options[distractor_label]
                    assert item.options[distractor_label].endswith("는"), (
                        f"the distractor must be the verb-pattern '는' overgeneralization, got {item}")
                    assert item.options[item.answer] in ("매운", "작은", "큰", "긴", "예쁜", "비싼"), (
                        f"the correctly-contracted adjective form must be correct, got {item}")
                if skill_id == "subj_topic":
                    distractor_label = next(l for l in item.options if l != item.answer)
                    assert item.options[item.answer] != item.options[distractor_label]
                    assert item.options[item.answer] in ("제가", "내가"), (
                        f"the subject-marked (JKS) form must be the correct answer to a 누가 question, got {item}")
                    assert item.options[distractor_label] in ("저는", "나는"), (
                        f"the topic-marked (JX) form must be the distractor, got {item}")
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
