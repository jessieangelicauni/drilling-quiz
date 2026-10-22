"""
Deterministic Korean morphology helpers (kiwipiepy) for the item generator:
building the wrong options a skill needs directly from a correct answer's
own form, instead of asking an LLM to invent them -- validated at 77.3%
item acceptance vs. 28.8% when a model wrote distractors freely (it
reproduces the exact broken pattern it is shown as "do not do this",
verbatim). The one shared definition of what a valid wrong answer looks
like per skill, used by item_foundry.py (the production generator) and any
benchmark/eval script -- not two definitions that could silently disagree.
"""
from __future__ import annotations

import random
import re

from kiwipiepy import Kiwi

_KIWI = Kiwi()


def has_batchim(syllable: str) -> bool:
    """Does the final Hangul syllable block end in a consonant (batchim)?

    Deterministic Unicode decomposition, not a model call -- needed because
    an LLM cannot be trusted to pick the right 을/를 (or 이/가, 은/는, 과/와...)
    allomorph for a given noun. Found via a real failure: EXAONE-4.0-1.2B
    generated "라면를 먹어요" (should be "라면을" -- 라면 ends in the
    consonant ㄴ) while otherwise producing a structurally valid item that
    distractor_validator's tag-only check would not have caught.
    """
    code = ord(syllable[-1]) - 0xAC00
    if not (0 <= code < 11172):
        raise ValueError(f"{syllable!r} last character is not a Hangul syllable block")
    return code % 28 != 0


def harmony_positive(syllable: str) -> bool:
    """True if `syllable`'s vowel is ㅏ or ㅗ (takes the -아요 family of
    endings), False otherwise (-어요 family). Needed for KAGAS's CONJ
    category (irregular-stem errors, e.g. 르-irregular verbs 자르다/
    모르다/부르다): kiwi.join() always produces the CORRECT irregular
    contraction (자르+아요 -> 잘라요) regardless of what ending variant
    it's fed, so it can't be used to build the common learner error of
    skipping the irregular contraction (literal "자르아요"). That
    distractor has to be constructed by hand, which means doing the vowel
    harmony kiwi.join() would otherwise have handled automatically.
    """
    code = ord(syllable[-1]) - 0xAC00
    if not (0 <= code < 11172):
        raise ValueError(f"{syllable!r} last character is not a Hangul syllable block")
    vowel_index = (code % (21 * 28)) // 28
    return vowel_index in (0, 8)  # 0=ㅏ, 8=ㅗ


def adj_relativizer_ending(adj_stem: str) -> str:
    """은 after a consonant-final (batchim) adjective stem, the combining
    jongseong NIEUN (U+11AB, NOT the standalone letter ㄴ U+3134 -- a real
    bug caught empirically: kiwi.join() silently fails to merge the
    standalone letter into the preceding syllable, producing garbage like
    "크ㄴ" instead of "큰") after a vowel-final one. kiwi.join() then
    applies irregular-stem contractions (ㅂ-irregular 맵다->매운, ㄹ-elision
    길다->긴, 으-drop 예쁘다->예쁜) correctly regardless of whether the
    stem's tag is passed as VA or VA-I -- verified empirically, so callers
    don't need to detect irregularity themselves, same simplification
    already relied on for verb tags elsewhere in this codebase.
    """
    return "은" if has_batchim(adj_stem) else "ᆫ"


_HANGUL_SPAN = re.compile(r"[가-힣](?:[가-힣\s]*[가-힣])?[.!?]?")
_VERB_TAGS = ("VV", "VV-I", "VX", "VA", "XSV")


def correct_korean_conjugation(text: str) -> str:
    """Auto-corrects a specific, recurring LLM conjugation error: a bare
    "요" ending attached directly to a verb stem instead of the proper
    linking-vowel contraction (마시요 -> 마셔요, 보요 -> 봐요, 듣어요 ->
    들어요, 공부하요 -> 공부해요).

    Works on free-form mixed-language text (Korean example sentences
    embedded in an English/Indonesian explanation) by finding Hangul spans
    and correcting each one independently. Deliberately narrow: this fixes
    ONE identified error pattern, not a general grammar checker --
    kiwipiepy's analyzer accepts "stem+요" as one valid (if non-standard)
    parse, which is exactly why a plain analyze-then-join round trip does
    NOT catch it on its own; forcing the EF morph to "어요" before
    rejoining is what makes the correction happen. Verified idempotent on
    already-correct text (공부해요, 가요, 만나요, ...) -- kiwipiepy already
    represents those internally as stem+"어요", never bare stem+"요", so
    this never "fixes" something that wasn't broken.
    """
    def _fix_span(match: re.Match) -> str:
        span = match.group(0)
        try:
            tokens = _KIWI.analyze(span)[0][0]
        except Exception:
            return span
        morphs = []
        for t in tokens:
            if t.tag in ("EF", "EC") and t.form == "요" and morphs and morphs[-1][1] in _VERB_TAGS:
                morphs.append(("어요", "EF"))
            else:
                morphs.append((t.form, t.tag))
        try:
            return _KIWI.join(morphs)
        except Exception:
            return span

    return _HANGUL_SPAN.sub(_fix_span, text)


_NOUN_TAGS = {"NNG", "NNP", "NNB"}
# An ending whose form already closes the sentence on its own -- a further trailing 요 after one
# of these is not the "-는데요/-거든요" polite-softener pattern (which kiwi also tokenizes as
# EF + separate 요), it is a plain duplicate (봅니다요, 확인했어요요).
_SELF_CLOSING_EF_SUFFIXES = ("요", "니다", "니까", "시오", "시다", "다")  # "다" last: plain/written-style
# endings (는다/있다/었다/겠다/다 itself) are always complete sentence-final forms in their own
# register and never take a polite 요 after them, same as the explicit 니다/시다 cases -- found
# empirically ("읽는다요" passed as if it were the "-는데요" pattern before this was added).


def morphologically_broken(text: str) -> bool:
    """True if kiwipiepy's own tokenization shows `text` is not one coherent
    word -- found by inspecting the actual tokenization of a recurring LLM
    bug (answer + extra particle/ending stuck on top of it: 봅니다요,
    배터리를로, 정확한는), not guessed. Deliberately narrow, same spirit as
    correct_korean_conjugation() above: three real Korean patterns are
    explicitly exempted because they look identical in tag shape to the bug
    but are not errors --
      - EF + 요 is fine when the EF itself is a bare connective-style ending
        (-는데/-거든/-잖아 etc.) that conventionally takes a polite 요; it is
        only wrong when the EF already closes the sentence by itself
        (ends in 요/니다/니까/시오/시다 already -- 효율적인데요 ok,
        봅니다요/확인했어요요 not).
      - ETM followed by a noun is a normal adnominal phrase (복잡한것
        = "the complicated thing"), only ETM followed by anything else
        (another ETM, a particle, a verb) is broken.
      - two particles in a row is fine for [case particle] + topic marker
        는/은 (에서는, 에게는, 로는 are everyday Korean), only other
        combinations (two case particles, topic+case, two generic
        delimiters) are broken.
    Checked against the real failures that motivated each rule plus the
    real false positives found by testing (효율적인데요, 복잡한것,
    팩토리에서는) before relying on it -- see benchmark_v2.py's
    root-cause notebook. Used by both benchmark_v2.py (word-level
    distractors) and item_foundry.py (whole-sentence distractors, applied
    to just the differing span)."""
    toks = _KIWI.tokenize(text)
    tags, forms = [t.tag for t in toks], [t.form for t in toks]
    while tags and tags[-1] == "SF":  # trailing sentence punctuation (.?!) is never part of the bug
        tags, forms = tags[:-1], forms[:-1]
    if not tags:
        return False
    for i in range(len(tags) - 1):
        if tags[i] == "ETM" and tags[i + 1] not in _NOUN_TAGS:
            return True
        if tags[i] == "EF":
            is_softener_yo = tags[i + 1] == "JX" and forms[i + 1] == "요"
            if not is_softener_yo or forms[i].endswith(_SELF_CLOSING_EF_SUFFIXES):
                return True
    particles = [(j, tags[j], f) for j, f in enumerate(forms) if tags[j].startswith("J")]
    for (j1, tag1, _), (_, tag2, form2) in zip(particles, particles[1:]):
        if not (tag1 != "JX" and tag2 == "JX" and form2 in ("는", "은")):
            return True
    return False


# ---------------------------------------------------------------- deterministic distractors
#
# Builds the wrong options for a skill directly from a correct answer's own
# morphology/lexical identity, instead of asking an LLM to invent them --
# validated at 77.3% item acceptance vs. 28.8% when EXAONE wrote distractors
# freely (EXAONE reproduces the exact broken pattern it is shown as "do not
# do this", verbatim). The one shared definition of what a valid wrong
# answer looks like per skill, used by item_foundry.py (the production
# generator) and any benchmark/eval script -- not two definitions that could
# silently disagree.

# Padding for a skill whose real minimal pair has only ONE natural wrong-form
# contrast (honorific presence/absence, past/present, adjective/verb
# relativizer): a bare, unrelated noun fails the slot's own grammatical role
# (not a conjugated predicate, not an adnominal ending) by construction, so
# it can never accidentally also be correct.
_FILLER_WORDS = ("학교", "사과", "저기")

# Honorific-worthy by social convention, not by any rule kiwipiepy or an LLM
# could derive -- common/frequent enough that a log-prob scorer reliably
# treats the honorific form as more natural than the plain one for them
# (unlike domain jargon a model invents freely, e.g. 공장장님/데이터분석가님).
_HONORIFIC_SUBJECTS = ("선생님", "할머니", "교수님", "사장님", "의사선생님")

# Regular -게 adjectives with no competing irregular adverb form -- 깨끗하다
# is deliberately excluded: its idiomatic "깨끗이" outcompetes the regular
# "깨끗하게" this skill tests, on frequency alone.
_MOD_ADVERB_ADJECTIVES = ("조용하다", "많다", "늦다", "느리다", "빠르다")

# 르-irregular verbs, each paired with the object its collocation needs --
# the irregular stem itself is the fixed closed set being tested here, not a
# vocabulary choice left open.
_IRREGULAR_CONJ_VERBS = (("머리를", "자르다"), ("답을", "모르다"), ("노래를", "부르다"))

# verb_choice: a LEXICAL error (wrong word for the context), not a
# morphological one. Each wrong verb is deliberately a nonsense collocation
# for the noun (not just "a different but still plausible" verb).
_VERB_CHOICE_PAIRS = (
    ("데이터", "분석하다", "마시다"),
    ("센서", "확인하다", "먹다"),
    ("배터리", "충전하다", "듣다"),
    ("부품", "조립하다", "찍다"),
    ("에너지", "절약하다", "입다"),
)
_VERB_CHOICE_WRONG = {correct: wrong for _, correct, wrong in _VERB_CHOICE_PAIRS}

# noun_counter: also lexical, same reasoning as verb_choice. 선생님/분/명 is a
# known, disclosed exception (분's honorific register is invisible to a
# frequency-based score) -- kept anyway since testing something other than
# 분's honorific-ness would defeat the point of including it.
_NOUN_COUNTER_PAIRS = (("사과", "개", "명"), ("학생", "명", "개"), ("선생님", "분", "명"),
                        ("책", "권", "장"), ("컵", "개", "명"))
_NOUN_COUNTERS = ("개", "명", "분", "권", "장")

# Every skill with a deterministic-distractor function -- everything but
# subj_topic, a pedagogical convention (focus-of-the-question), not a hard
# grammar rule, so no deterministic builder applies to it.
DETERMINISTIC_SKILLS = frozenset({
    "particle_loc", "particle_obj", "time_range",
    "tense_past", "honorific", "adj_relativizer", "mod_adverb", "irregular_conj",
    "verb_choice", "noun_counter",
})

_PARTICLE_CANDIDATES = {
    "particle_loc": {"에서": ["에", "_dir", "까지"], "에": ["에서", "_dir", "까지"]},
    "time_range": {"부터": ["까지", "에서", "에"], "까지": ["부터", "에서", "에"]},
}


def _direction_particle(stem: str) -> str:
    """으로 after a consonant-final (batchim) stem, 로 after a vowel-final
    one -- same allomorph rule as has_batchim's 을/를 case, applied to
    locative/directional 로/으로 instead."""
    return "으로" if has_batchim(stem) else "로"


def _subject_particle(stem: str) -> str:
    """가 after a vowel-final stem, 이 after a consonant-final one -- the
    same batchim rule, applied to 이/가."""
    return "이" if has_batchim(stem) else "가"


def _deterministic_particle_distractors(skill_id: str, answer: str) -> list[dict] | None:
    """Builds the wrong options for a particle-swap skill directly from
    `answer`'s own suffix, instead of asking an LLM. Returns None when
    `skill_id`/`answer` is not covered, so the caller tries the next skill
    family or gives up on this candidate."""
    if skill_id == "particle_obj":
        for particle in ("을", "를"):
            if answer.endswith(particle):
                stem = answer[: -len(particle)]
                if not stem:  # a bare particle with no noun before it
                    return None
                wrongs = ("에", _subject_particle(stem), _direction_particle(stem))
                return [{"mistake": f"uses the particle '{c}' where the object particle '{particle}' belongs",
                          "text": stem + c} for c in wrongs]
        return None
    table = _PARTICLE_CANDIDATES.get(skill_id)
    if table is None:
        return None
    for particle, wrongs in table.items():
        if answer.endswith(particle):
            stem = answer[: -len(particle)]
            if not stem:
                return None
            resolved = [_direction_particle(stem) if c == "_dir" else c for c in wrongs]
            return [{"mistake": f"uses '{c}' where '{particle}' belongs", "text": stem + c} for c in resolved]
    return None


def _replace_last_morph(answer: str, new_form: str, new_tag: str) -> str | None:
    """Re-tokenizes `answer` and swaps only its LAST morph for (new_form,
    new_tag), then rejoins -- letting kiwi.join() apply Korean's own
    irregular contractions rather than hand-rolling them."""
    toks = _KIWI.tokenize(answer)
    if not toks:
        return None
    morphs = [(t.form, t.tag) for t in toks[:-1]] + [(new_form, new_tag)]
    try:
        result = _KIWI.join(morphs)
    except Exception:
        return None
    return result if result != answer else None


def _drop_ep_morph(answer: str, ep_forms: tuple[str, ...]) -> str | None:
    """Re-tokenizes `answer` and removes whichever EP (pre-final ending)
    morph matches `ep_forms`, then rejoins the rest. None if absent."""
    toks = _KIWI.tokenize(answer)
    morphs = [(t.form, t.tag) for t in toks if not (t.tag == "EP" and t.form in ep_forms)]
    if len(morphs) == len(toks):
        return None
    try:
        result = _KIWI.join(morphs)
    except Exception:
        return None
    return result if result != answer else None


def _deterministic_ending_distractors(skill_id: str, answer: str,
                                       rng: random.Random = random) -> list[dict] | None:
    """honorific/tense_past/adj_relativizer/mod_adverb/irregular_conj/
    verb_choice/noun_counter: one real morphological or lexical contrast,
    padded with _FILLER_WORDS to reach 3 options where only one real
    contrast exists. Returns None when `answer` does not show the expected
    form for this skill (the caller then falls back or retries)."""
    if skill_id == "tense_past":
        wrong = _drop_ep_morph(answer, ("았", "었", "였"))
        mistake = "uses the present tense instead of the past tense"
    elif skill_id == "honorific":
        toks = _KIWI.tokenize(answer)
        if not toks or toks[-1].tag not in ("EF", "EP"):
            return None
        wrong = _drop_ep_morph(answer, ("시",)) or _replace_last_morph(answer, "어요", "EF")
        mistake = "drops the subject honorific -(으)시-"
    elif skill_id == "adj_relativizer":
        toks = _KIWI.tokenize(answer)
        if not toks or toks[-1].tag != "ETM":
            return None
        wrong = _replace_last_morph(answer, "는", "ETM")
        mistake = "uses the verb-style relativizer -는 on an adjective stem"
    elif skill_id == "mod_adverb":
        toks = _KIWI.tokenize(answer)
        if len(toks) < 2 or toks[-1].tag != "EC" or toks[-1].form != "게":
            return None
        wrong = _replace_last_morph(answer, adj_relativizer_ending(toks[-2].form), "ETM")
        mistake = "uses the adjective-attributive ending instead of the adverbial -게"
    elif skill_id == "irregular_conj":
        toks = _KIWI.tokenize(answer)
        # kiwi lemmatises any surface conjugation back to the dictionary
        # stem (자르/모르/부르, tag VV) regardless of how it was conjugated.
        if not toks or toks[0].tag != "VV" or toks[0].form not in ("자르", "모르", "부르"):
            return None
        stem = toks[0].form
        rest = "".join(t.form for t in toks[1:])
        if not rest.startswith(("아", "어")):
            return None  # a register this simple vowel-initial-ending swap doesn't cover
        glide = "아" if harmony_positive(stem[0]) else "어"
        wrong = stem + glide + rest[1:]  # literal concatenation -- no kiwi.join(), which would
                                         # auto-apply the correct ㄹㄹ contraction and erase the mistake
        mistake = "writes the 르-irregular stem's ending without the ㄹㄹ contraction"
    elif skill_id == "verb_choice":
        # 3 genuinely different wrong VERBS exist in the curated set (not
        # just one real contrast padded with filler nouns) -- re-conjugated
        # to match the answer's own tense/register via the same tail-morph-
        # preserving join() used for the single-contrast skills above.
        toks = _KIWI.tokenize(answer)
        tail = [(t.form, t.tag) for t in toks if t.tag in ("EP", "EF")]
        if not tail:
            return None
        lead = "".join(t.form for t in toks if t.tag not in ("EP", "EF")) + "다"
        if lead not in _VERB_CHOICE_WRONG:
            return None  # the sentence didn't use one of the curated verbs
        other_wrongs = rng.sample([w for c, w in _VERB_CHOICE_WRONG.items() if c != lead], 2)
        wrong_verbs = [_VERB_CHOICE_WRONG[lead], *other_wrongs]
        try:
            texts = [_KIWI.join([(w[:-1], "VV"), *tail]) for w in wrong_verbs]
        except Exception:
            return None
        return [{"mistake": f"uses an implausible verb for the object (not '{lead}')", "text": t}
                for t in texts]
    elif skill_id == "noun_counter":
        # Also lexical, no conjugation involved -- a literal counter swap,
        # keeping whatever suffix (bare, or "의") the sentence used.
        for counter in _NOUN_COUNTERS:
            if answer in (counter, counter + "의"):
                suffix = answer[len(counter):]
                others = rng.sample([c for c in _NOUN_COUNTERS if c != counter], 3)
                return [{"mistake": f"uses the counter '{c}' instead of '{counter}'", "text": c + suffix}
                        for c in others]
        return None
    else:
        return None
    if wrong is None:
        return None
    return [{"mistake": mistake, "text": wrong}] + \
           [{"mistake": "an unrelated filler word, not a real grammar mistake", "text": w}
            for w in _FILLER_WORDS[:2]]


def deterministic_distractors(skill_id: str, answer: str, rng: random.Random = random) -> list[dict] | None:
    """The single dispatch point: tries the particle-swap builder, then the
    ending/lexical builder. None means no rule exists for `skill_id` (only
    subj_topic) or `answer` doesn't show the form either builder needs."""
    return (_deterministic_particle_distractors(skill_id, answer)
            or _deterministic_ending_distractors(skill_id, answer, rng))


def deterministic_distractors_for_sentence(skill_id: str, sentence: str, rng: random.Random = random):
    """Finds which word of a FULL SENTENCE is `skill_id`'s answer and builds
    its distractors, without trusting the model that wrote the sentence to
    self-report a span. Tries every space-separated word as a candidate
    answer, in order -- the same builder function that constructs the wrong
    forms also decides whether a word is a valid answer, so there is
    exactly one definition of "valid answer" for a skill, not a second one
    (e.g. a tag-matching heuristic) that could silently disagree with it.
    A word that recurs in the sentence is skipped even if it would
    otherwise match -- its position can't be located unambiguously.

    Returns (start, end, wrong_texts, mistakes), or None if no word works.
    Verified empirically against real kiwipiepy tokenization for all ten
    skills this covers before relying on it (see this module's __main__)."""
    body = sentence.rstrip(".?!")
    for word in dict.fromkeys(body.split()):
        result = deterministic_distractors(skill_id, word, rng)
        if result is None:
            continue
        if sentence.count(word) > 1:
            continue
        start = sentence.find(word)
        return start, start + len(word), [d["text"] for d in result], [d["mistake"] for d in result]
    return None


def sentence_constraint(skill_id: str, description: str, rng: random.Random = random) -> str:
    """Appends a skill-specific constraint to `description` (the skill's
    base grammar-point text) so a freely-written sentence is anchored to a
    case the deterministic builder above can actually use -- found
    empirically necessary for these five skills (free sentence choice picks
    an unusable construction far too often otherwise, e.g. honorific: 1/6
    accepted vs 6/6 once the subject is anchored to a common
    honorific-worthy title). Returns `description` unchanged for every
    other skill."""
    if skill_id == "honorific":
        subject = rng.choice(_HONORIFIC_SUBJECTS)
        return (f"{description} The subject performing the action must be exactly '{subject}'. "
                f"The verb must end in -(으)세요 or -(으)십니다/-(스)ㅂ니다, not -죠 or any other ending.")
    if skill_id == "mod_adverb":
        adj_stem = rng.choice(_MOD_ADVERB_ADJECTIVES)[:-1]  # drop the dictionary-form "다"
        return f"{description} The adverb must be exactly '{adj_stem}게'."
    if skill_id == "irregular_conj":
        _, verb = rng.choice(_IRREGULAR_CONJ_VERBS)
        conjugated = _KIWI.join([(verb[:-1], "VV"), ("어요", "EF")])
        return (f"The sentence must use the verb '{verb}', conjugated as '{conjugated}' "
                f"(polite present tense), not left in dictionary form.")
    if skill_id == "verb_choice":
        noun, verb, _ = rng.choice(_VERB_CHOICE_PAIRS)
        return f"The sentence must use '{noun}' as the object of the verb '{verb}' (conjugated appropriately)."
    if skill_id == "noun_counter":
        noun, counter, _ = rng.choice(_NOUN_COUNTER_PAIRS)
        return f"The sentence must count '{noun}' using the counter word '{counter}' (e.g. '다섯 {counter}')."
    return description


if __name__ == "__main__":
    assert has_batchim("책"), "책 ends in the consonant ㄱ"
    assert not has_batchim("사과"), "사과 ends in a vowel"
    assert has_batchim("라면"), "라면 ends in the consonant ㄴ"

    assert correct_korean_conjugation("저는 커피를 마시요.") == "저는 커피를 마셔요."
    assert correct_korean_conjugation("저는 영화를 보요.") == "저는 영화를 봐요."
    assert correct_korean_conjugation("저는 학교에서 수업을 듣어요.") == "저는 학교에서 수업을 들어요."
    assert correct_korean_conjugation("저는 공부하요.") == "저는 공부해요."
    for already_ok in ("저는 책을 읽어요.", "저는 학교에 가요.", "내일 만나요.", "저는 공부해요."):
        assert correct_korean_conjugation(already_ok) == already_ok, (
            f"must be idempotent on already-correct text, got a change for {already_ok!r}")
    mixed = '**"저는 학교에서 수업을 듣어요"** (I have class at the school).'
    assert "들어요" in correct_korean_conjugation(mixed), "must fix Korean embedded inside English/markdown prose"

    assert harmony_positive("자"), "자 has vowel ㅏ"
    assert harmony_positive("모"), "모 has vowel ㅗ"
    assert not harmony_positive("부"), "부 has vowel ㅜ"
    assert _KIWI.join([("자르", "VV"), ("아요", "EF")]) == "잘라요"
    assert _KIWI.join([("부르", "VV"), ("아요", "EF")]) == "불러요", (
        "kiwi.join() must normalize harmony itself even when fed the 'wrong' variant"
    )

    assert adj_relativizer_ending("작") == "은", "작 ends in a consonant batchim"
    assert adj_relativizer_ending("크") == "ᆫ", "크 ends in a vowel"
    assert _KIWI.join([("맵", "VA"), (adj_relativizer_ending("맵"), "ETM")]) == "매운", (
        "ㅂ-irregular: must contract, not just concatenate"
    )
    assert _KIWI.join([("길", "VA"), (adj_relativizer_ending("길"), "ETM")]) == "긴", (
        "ㄹ-irregular: must elide ㄹ before ETM, not just concatenate"
    )
    assert _KIWI.join([("예쁘", "VA"), (adj_relativizer_ending("예쁘"), "ETM")]) == "예쁜", (
        "으-stem: must drop 으, not just concatenate"
    )
    assert _KIWI.join([("작", "VA"), (adj_relativizer_ending("작"), "ETM")]) == "작은"
    # The real bug this guards against: using the standalone letter ㄴ (U+3134)
    # instead of the combining jongseong NIEUN (U+11AB) silently fails to merge.
    broken = _KIWI.join([("크", "VA"), ("ㄴ", "ETM")])
    assert broken != "큰", "sanity check: confirms the wrong codepoint really does produce garbage, not 큰"

    assert _KIWI.join([("저", "NP"), ("가", "JKS")]) == "제가", "irregular pronoun+subject contraction"
    assert _KIWI.join([("나", "NP"), ("가", "JKS")]) == "내가", "irregular pronoun+subject contraction"
    assert _KIWI.join([("저", "NP"), ("는", "JX")]) == "저는", "no contraction for the topic particle"

    # morphologically_broken (benchmark_v2.py root-cause notebook, 2026-10-22): real broken
    # distractors must be caught, real clean ones and the 3 explicit exemptions must not.
    for bad in ("봅니다요", "읽어요를", "테스트해요니", "높은는", "정확한는", "확인했어요요", "배터리를로"):
        assert morphologically_broken(bad), f"{bad!r} must be flagged broken"
    for good in ("확인해요", "학교로", "효율적인데요", "복잡한것", "팩토리에서는", "높는",
                 "먹어요.", "먹어요?", "먹어요!"):
        assert not morphologically_broken(good), f"{good!r} must NOT be flagged broken"
    for bad in ("읽어요어어요", "읽어요요요", "읽어요어요"):
        assert morphologically_broken(bad), f"{bad!r} (stem + duplicated fragment) must be flagged broken"
    for bad in ("읽는다요", "했다요", "하겠다요"):
        assert morphologically_broken(bad), f"{bad!r} (plain -다 ending + 요) must be flagged broken"
    assert not morphologically_broken("읽는다"), "plain -는다 ending alone (no trailing 요) is valid"

    # deterministic_distractors_for_sentence: verified against real kiwipiepy
    # tokenization (not guessed) on one full sentence per skill, picking the
    # right word out of several candidates every time.
    located = {
        "particle_loc": ("저는 공장에서 결함을 검사해요.", "공장에서"),
        "particle_obj": ("저는 공장에서 결함을 검사해요.", "결함을"),
        "time_range": ("저는 학교부터 공원까지 걸어요.", "학교부터"),
        "honorific": ("선생님께서 가르치세요.", "가르치세요"),
        "tense_past": ("저는 어제 학교에 갔어요.", "갔어요"),
        "adj_relativizer": ("저는 매운 음식을 좋아해요.", "매운"),
        "mod_adverb": ("저는 조용하게 말해요.", "조용하게"),
        "irregular_conj": ("저는 머리를 잘라요.", "잘라요"),
        "noun_counter": ("사과가 다섯 개 있어요.", "개"),
        "verb_choice": ("저는 데이터를 분석해요.", "분석해요"),
    }
    for skill_id, (sentence, expected_word) in located.items():
        result = deterministic_distractors_for_sentence(skill_id, sentence, random.Random(0))
        assert result is not None, f"{skill_id}: expected to locate {expected_word!r} in {sentence!r}, got None"
        start, end, wrongs, mistakes = result
        assert sentence[start:end] == expected_word, (
            f"{skill_id}: located {sentence[start:end]!r}, expected {expected_word!r}")
        assert len(wrongs) == 3 and len(mistakes) == 3, f"{skill_id}: expected 3 distractors, got {wrongs}"
        assert len(set(wrongs)) == 3 and expected_word not in wrongs, (
            f"{skill_id}: distractors must be 3 distinct forms, none equal to the answer, got {wrongs}")
    assert deterministic_distractors_for_sentence("subj_topic", "저는 학교에 가요.", random.Random(0)) is None, (
        "subj_topic has no deterministic builder by design -- must always return None")

    assert sentence_constraint("honorific", "base") != "base", "honorific must anchor a subject"
    assert sentence_constraint("particle_loc", "base") == "base", "a skill with no constraint must pass through"

    print("all smoke tests passed")
