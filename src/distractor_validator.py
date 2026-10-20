"""
KAGAS-aligned distractor validator (PRD §3.2): uses Korean morphological
analysis (kiwipiepy) to check that an option actually parses as the
part-of-speech/particle it's supposed to be testing -- instead of trusting
the generator's own label for what grammar point a distractor represents.

Built after a real failure observed from EXAONE-4.0-1.2B generation (see
docs/PRD.md §3.2 audit note, 2026-10-19): asked to produce a distractor
confusing "에" with "에서", it silently produced a sentence using "와" (a
conjunctive particle, unrelated grammar point) instead, and its own
self-written explanation additionally mislabelled "에" as an accusative
particle. Neither error would be caught by asking the LLM to check itself.

This validator works on a cloze (stem + blank + options) representation,
not whole regenerated sentences -- constraining the format this way is
itself part of the fix: it structurally prevents the kind of off-target
drift seen in that failure, by construction, rather than detecting it after
the fact over free-form text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from kiwipiepy import Kiwi

_KIWI = Kiwi()

PARTICLE_TAGS = ("JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC")


@dataclass
class ValidationResult:
    valid: bool
    reason: str


def particle_tags(sentence: str) -> list[tuple[str, str]]:
    """[(surface_form, tag), ...] for every particle morpheme in the sentence."""
    analysis = _KIWI.analyze(sentence)[0][0]
    return [(t.form, t.tag) for t in analysis if t.tag in PARTICLE_TAGS]


def all_morph_tags(sentence: str) -> list[tuple[str, str]]:
    """[(surface_form, tag), ...] for every morpheme, not just particles --
    needed for skills whose blank is a verb ending (EF/EP), not a particle.
    """
    analysis = _KIWI.analyze(sentence)[0][0]
    return [(t.form, t.tag) for t in analysis]


def validate_cloze_option(stem_with_blank: str, blank_marker: str, option: str,
                           expected_tag: str) -> ValidationResult:
    """Does substituting `option` into the blank parse as `expected_tag`
    (e.g. "JKB", or "EF" for a verb-ending skill) at that position? Catches
    an option that isn't even the right kind of morpheme for the grammar
    point being tested.
    """
    if blank_marker not in stem_with_blank:
        raise ValueError(f"{blank_marker!r} not found in stem")
    filled = stem_with_blank.replace(blank_marker, option, 1)
    tags_at_option = [tag for form, tag in all_morph_tags(filled) if form == option]
    if expected_tag in tags_at_option:
        return ValidationResult(True, f"{option!r} parses as {expected_tag}, as expected")
    found = set(tags_at_option) or {"not analysed as a particle at all"}
    return ValidationResult(False, f"{option!r} parses as {found}, not {expected_tag}")


def validate_options(stem_with_blank: str, blank_marker: str,
                      options: dict[str, str], expected_tag: str) -> dict[str, ValidationResult]:
    """Validate a {option_id: option_text} map in one call."""
    return {oid: validate_cloze_option(stem_with_blank, blank_marker, text, expected_tag)
            for oid, text in options.items()}


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


def expected_object_particle(preceding_text: str) -> str:
    """을 after a consonant-final noun, 를 after a vowel-final one."""
    return "을" if has_batchim(preceding_text.strip()[-1]) else "를"


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


def validate_sentence_final_tag(sentence: str, expected_tag: str) -> ValidationResult:
    """Does the sentence's final predicate morpheme (the one right before
    sentence-final punctuation, if any) carry `expected_tag`?

    For options that are a WHOLE conjugated word (e.g. an honorific vs
    plain verb ending), not a single particle: validate_cloze_option can't
    be reused as-is, because it looks for a token whose FORM exactly
    equals the option string, and kiwipiepy decomposes a conjugated verb
    into multiple morphemes (stem + ending) -- no single token's form ever
    equals the whole word. Checking the sentence-final morpheme's tag
    instead works regardless of how many syllables the option spans.
    """
    tokens = all_morph_tags(sentence)
    content = [t for t in tokens if t[1] != "SF"]
    if not content:
        return ValidationResult(False, "sentence has no content morphemes to check")
    form, tag = content[-1]
    if tag == expected_tag:
        return ValidationResult(True, f"sentence-final morpheme {form!r} parses as {expected_tag}, as expected")
    return ValidationResult(False, f"sentence-final morpheme {form!r} parses as {tag}, not {expected_tag}")


def validate_prenominal_tag(sentence: str, following_word: str, expected_tag: str) -> ValidationResult:
    """Does the morpheme immediately before `following_word` (a noun the
    option is modifying) carry `expected_tag`?

    Needed for KAGAS's ADJ category (§9.10): the adjective-relativizer
    blank ("___ 음식을 좋아해요") isn't at the sentence's end like every
    other whole-word skill so far (honorific, tense_past) -- it's in the
    middle, modifying a following noun -- so validate_sentence_final_tag
    doesn't apply. Mirrors that function's approach (check a tag at a
    structural position, not a literal string match, since the option
    spans multiple morphemes) but anchored to the following noun instead
    of the end of the sentence.

    Note: both the correct form (은/ᆫ) and the common learner error (는,
    the verb-style ending wrongly generalized onto an adjective stem) tag
    as ETM here -- kiwipiepy's tagger is permissive about which stem class
    an ETM attaches to (same permissiveness documented in
    correct_korean_conjugation's docstring). This validator only catches a
    structurally broken option (e.g. not a relativizer at all); it cannot
    and is not meant to distinguish adjective-correct from verb-pattern-
    wrong by tag alone -- that correctness comes from which bucket the
    caller puts a form in (built via kiwi.join() vs hand-built), the same
    division of responsibility as irregular_conj's correct/distractor pair.
    """
    tokens = all_morph_tags(sentence)
    idx = next((i for i, (form, tag) in enumerate(tokens)
                if form == following_word and tag.startswith("NN")), None)
    if idx is None or idx == 0:
        return ValidationResult(False, f"{following_word!r} not found as a noun token with a preceding morpheme")
    form, tag = tokens[idx - 1]
    if tag == expected_tag:
        return ValidationResult(True, f"morpheme {form!r} right before {following_word!r} parses as {expected_tag}")
    return ValidationResult(False, f"morpheme {form!r} right before {following_word!r} parses as {tag}, not {expected_tag}")


def validate_postnominal_tag(sentence: str, preceding_word: str, expected_tag: str) -> ValidationResult:
    """Does the morpheme immediately AFTER `preceding_word` carry
    `expected_tag`? The mirror image of validate_prenominal_tag.

    Needed for `subj_topic` (이/가 vs 은/는, §9.11): the blank is a whole
    pronoun+particle unit (제가/저는, 내가/나는) where an irregular
    contraction changes the PRONOUN's own surface form depending on which
    particle attaches (저+가 -> 제가, not "저가") -- kiwipiepy still
    represents the underlying lemma as "저" regardless of the surface
    contraction, so `preceding_word` is the lemma (e.g. "저"), not the
    surface string, and this checks the particle token right after it.
    """
    tokens = all_morph_tags(sentence)
    idx = next((i for i, (form, tag) in enumerate(tokens) if form == preceding_word), None)
    if idx is None or idx + 1 >= len(tokens):
        return ValidationResult(False, f"{preceding_word!r} not found as a token with a following morpheme")
    form, tag = tokens[idx + 1]
    if tag == expected_tag:
        return ValidationResult(True, f"morpheme {form!r} right after {preceding_word!r} parses as {expected_tag}")
    return ValidationResult(False, f"morpheme {form!r} right after {preceding_word!r} parses as {tag}, not {expected_tag}")


def validate_before_tag(sentence: str, end_marker: str, expected_tag: str) -> ValidationResult:
    """Does the morpheme ending exactly where `end_marker` (a fixed string
    that follows the option in the template, e.g. " 말해요." or " 있어요.")
    begins carry `expected_tag`?

    For a whole-word option that sits mid-sentence, followed by more text,
    but where neither existing anchor fits: not validate_sentence_final_tag
    (the option isn't at the sentence's end), and not validate_prenominal_tag
    (what follows isn't a NOUN -- a counter word is followed by a verb/copula,
    an adverb by a verb -- and a verb's own surface form varies too much to
    anchor on reliably: a bare VV/VA token for a native verb vs an NNG+XSV
    split for a 하다-compound, e.g. "말하다" tokenises as 말(NNG)+하(XSV), not
    one "말하" token).
    """
    idx = sentence.find(end_marker)
    if idx < 0:
        return ValidationResult(False, f"{end_marker!r} not found in sentence")
    tok = next((t for t in _KIWI.analyze(sentence)[0][0] if t.start + t.len == idx), None)
    if tok is None:
        return ValidationResult(False, f"no morpheme ends exactly before {end_marker!r}")
    if tok.tag == expected_tag:
        return ValidationResult(True, f"morpheme {tok.form!r} right before {end_marker!r} parses as {expected_tag}")
    return ValidationResult(False,
        f"morpheme {tok.form!r} right before {end_marker!r} parses as {tok.tag}, not {expected_tag}")


def adj_relativizer_ending(adj_stem: str) -> str:
    """은 after a consonant-final (batchim) adjective stem, the combining
    jongseong NIEUN (U+11AB, NOT the standalone letter ㄴ U+3134 -- a real
    bug caught empirically: kiwi.join() silently fails to merge the
    standalone letter into the preceding syllable, producing garbage like
    "크ㄴ" instead of "큰") after a vowel-final one. Same batchim rule as
    expected_object_particle/honorific_ending, applied to the -(으)ㄴ
    adjective-relativizer ending. kiwi.join() then applies irregular-stem
    contractions (ㅂ-irregular 맵다->매운, ㄹ-elision 길다->긴, 으-drop
    예쁘다->예쁜) correctly regardless of whether the stem's tag is passed
    as VA or VA-I -- verified empirically, so callers don't need to detect
    irregularity themselves, same simplification already relied on for
    verb tags elsewhere in this codebase.
    """
    return "은" if has_batchim(adj_stem) else "ᆫ"


def honorific_ending(verb_stem: str) -> str:
    """으세요 after a consonant-final verb stem, 세요 after a vowel-final
    one -- same batchim rule as expected_object_particle, applied to a
    verb stem instead of a noun. kiwipiepy represents these as single
    fused EF morphemes (not stem+EP"시"+EF"어요"), which is also why this
    is the right pair to build with kiwi.join(), not the EP-insertion
    route: join([(stem,"VV"),("시","EP"),("어요","EF")]) produces the
    archaic/formal "셔요" contraction instead of the modern-standard
    "세요" actually taught in TOPIK materials.
    """
    return "으세요" if has_batchim(verb_stem) else "세요"


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


# Skills where kiwipiepy's parse score is an empirically confirmed signal
# for "correct form more plausible than distractor" (PRD §9.34) -- NOT a
# general-purpose semantic checker, scoped exactly to where it was tested
# and shown reliable. Deliberately excludes honorific/tense_past/
# noun_counter: tested on real curated pairs and found the SAME bias
# direction every time (plain/more-frequent form outscores the
# contextually-correct one) because correctness there depends on a social
# or temporal cue external to local n-gram frequency (honorific register,
# a time adverb elsewhere in the sentence, an arbitrary noun-class
# convention) -- exactly the kind of context a frequency-based parse score
# cannot see. Do not add a skill to this set without first checking it
# empirically the same way (see __main__), not by assuming it transfers.
LM_PLAUSIBILITY_RELIABLE_SKILLS = frozenset({
    "particle_loc", "time_range", "irregular_conj", "adj_relativizer", "verb_choice",
})


def lm_plausibility_gap(correct_sentence: str, distractor_sentence: str) -> float:
    """kiwipiepy's own analyze() score (log-probability-like, from its
    internal language model -- a genuine corpus-frequency signal, not an
    LLM judge) for the correct-filled sentence minus the distractor-filled
    one. Positive means the correct form is more plausible per the corpus
    LM; this is the raw signal validate_lm_plausibility() thresholds at 0.

    Tried after two LLM-judge framings for "is this distractor also
    correct" both failed empirically (PRD §9.31, Qwen3-4B: yes-biased
    under direct judgment, below-chance under pairwise comparison) --
    this is a structurally different approach (no LLM call at all) that
    was tested, not assumed, before being added: see PRD §9.34 for the
    N=20/skill validation run this threshold is based on.
    """
    correct_score = _KIWI.analyze(correct_sentence, top_n=1)[0][1]
    distractor_score = _KIWI.analyze(distractor_sentence, top_n=1)[0][1]
    return correct_score - distractor_score


def validate_lm_plausibility(skill_id: str, correct_sentence: str,
                              distractor_sentence: str) -> ValidationResult:
    """Only meaningful for skills in LM_PLAUSIBILITY_RELIABLE_SKILLS --
    returns valid=True (no opinion, not a false pass) for any other skill,
    so a caller that checks every skill uniformly does not silently start
    trusting a signal known to point the wrong way for honorific/
    tense_past/noun_counter.
    """
    if skill_id not in LM_PLAUSIBILITY_RELIABLE_SKILLS:
        return ValidationResult(True, f"lm_plausibility not validated for skill '{skill_id}', skipped")
    gap = lm_plausibility_gap(correct_sentence, distractor_sentence)
    if gap > 0:
        return ValidationResult(True, f"correct form scores {gap:.2f} higher, as expected")
    return ValidationResult(False, f"distractor scores {-gap:.2f} higher than the correct form -- "
                                    f"suspect pair, flag for review rather than ship silently")


if __name__ == "__main__":
    stem = "나는 학교___ 갔다."

    # The real EXAONE-4.0-1.2B failure, reduced to cloze form: asked for an
    # 에 vs 에서 item, it produced a distractor built on "와" (conjunction)
    # instead of a locative particle -- an unrelated grammar point.
    result = validate_cloze_option(stem, "___", "와", expected_tag="JKB")
    assert not result.valid, "a conjunctive particle must not pass as a locative (JKB) option"

    ok_particle = validate_cloze_option(stem, "___", "에서", expected_tag="JKB")
    assert ok_particle.valid, "에서 is a genuine JKB particle and must pass"

    wrong_case = validate_cloze_option(stem, "___", "을", expected_tag="JKB")
    assert not wrong_case.valid, "the accusative marker 을 must not pass as locative JKB"

    options = {"A": "에", "B": "에서", "C": "와", "D": "을"}
    results = validate_options(stem, "___", options, expected_tag="JKB")
    on_target = [oid for oid, r in results.items() if r.valid]
    assert set(on_target) == {"A", "B"}, f"only the two genuine locative particles should pass, got {on_target}"

    assert has_batchim("책"), "책 ends in the consonant ㄱ"
    assert not has_batchim("사과"), "사과 ends in a vowel"
    assert has_batchim("라면"), "라면 ends in the consonant ㄴ"
    assert expected_object_particle("라면") == "을", "the real EXAONE failure this guards against: 라면를 is wrong"
    assert expected_object_particle("사과") == "를"

    assert correct_korean_conjugation("저는 커피를 마시요.") == "저는 커피를 마셔요."
    assert correct_korean_conjugation("저는 영화를 보요.") == "저는 영화를 봐요."
    assert correct_korean_conjugation("저는 학교에서 수업을 듣어요.") == "저는 학교에서 수업을 들어요."
    assert correct_korean_conjugation("저는 공부하요.") == "저는 공부해요."
    for already_ok in ("저는 책을 읽어요.", "저는 학교에 가요.", "내일 만나요.", "저는 공부해요."):
        assert correct_korean_conjugation(already_ok) == already_ok, (
            f"must be idempotent on already-correct text, got a change for {already_ok!r}")
    mixed = '**"저는 학교에서 수업을 듣어요"** (I have class at the school).'
    assert "들어요" in correct_korean_conjugation(mixed), "must fix Korean embedded inside English/markdown prose"

    hon_check = validate_sentence_final_tag("선생님께서 가르치세요.", "EF")
    assert hon_check.valid, "a whole conjugated verb ending in EF must validate via sentence-final tag"
    bad_check = validate_sentence_final_tag("선생님께서 가르치는.", "EF")
    assert not bad_check.valid, "a non-EF sentence-final morpheme must fail"

    assert harmony_positive("자"), "자 has vowel ㅏ"
    assert harmony_positive("모"), "모 has vowel ㅗ"
    assert not harmony_positive("부"), "부 has vowel ㅜ"
    assert _KIWI.join([("자르", "VV"), ("아요", "EF")]) == "잘라요"
    assert _KIWI.join([("부르", "VV"), ("아요", "EF")]) == "불러요", (
        "kiwi.join() must normalize harmony itself even when fed the 'wrong' variant"
    )

    assert honorific_ending("읽") == "으세요", "읽 ends in a consonant batchim"
    assert honorific_ending("가") == "세요", "가 ends in a vowel"
    assert _KIWI.join([("읽", "VV"), (honorific_ending("읽"), "EF")]) == "읽으세요"
    assert _KIWI.join([("가", "VV"), (honorific_ending("가"), "EF")]) == "가세요"

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

    pren_ok = validate_prenominal_tag("저는 매운 음식을 좋아해요.", "음식", "ETM")
    assert pren_ok.valid, "은/ᆫ relativizer right before a noun must validate structurally"
    pren_bad = validate_prenominal_tag("저는 매운 음식을 좋아해요.", "없는단어", "ETM")
    assert not pren_bad.valid, "a noun that doesn't appear in the sentence must fail, not silently pass"

    assert _KIWI.join([("저", "NP"), ("가", "JKS")]) == "제가", "irregular pronoun+subject contraction"
    assert _KIWI.join([("나", "NP"), ("가", "JKS")]) == "내가", "irregular pronoun+subject contraction"
    assert _KIWI.join([("저", "NP"), ("는", "JX")]) == "저는", "no contraction for the topic particle"
    postn_subj = validate_postnominal_tag("제가 커피를 마셔요?", "저", "JKS")
    assert postn_subj.valid, "the particle right after 저 (surfaced as 제가) must validate as JKS"
    postn_topic = validate_postnominal_tag("저는 커피를 마셔요?", "저", "JX")
    assert postn_topic.valid, "the particle right after 저 (저는, no contraction) must validate as JX"
    postn_bad = validate_postnominal_tag("제가 커피를 마셔요?", "저", "JX")
    assert not postn_bad.valid, "제가's particle is JKS, must not pass a JX check"

    # lm_plausibility (PRD §9.34): must correctly favor the correct form
    # for the 5 confirmed-reliable skills, and must stay neutral (not
    # silently wrong) for the 3 known-unreliable ones instead of being
    # applied there anyway.
    lm_particle = validate_lm_plausibility("particle_loc", "저는 학교에서 공부해요.", "저는 학교에 공부해요.")
    assert lm_particle.valid, f"particle_loc: correct form (에서) must score higher, got {lm_particle.reason}"
    lm_irregular = validate_lm_plausibility("irregular_conj", "저는 머리를 잘라요.", "저는 머리를 자르어요.")
    assert lm_irregular.valid, f"irregular_conj: contracted form must score higher, got {lm_irregular.reason}"
    lm_adj = validate_lm_plausibility("adj_relativizer", "저는 매운 음식을 좋아해요.", "저는 맵는 음식을 좋아해요.")
    assert lm_adj.valid, f"adj_relativizer: correct relativizer must score higher, got {lm_adj.reason}"
    lm_honorific = validate_lm_plausibility("honorific", "할머니께서 쉬세요.", "할머니께서 쉬어요.")
    assert lm_honorific.valid and "skipped" in lm_honorific.reason, (
        "honorific is a known-unreliable skill -- must return valid=True via the explicit "
        f"skip path, not by coincidentally scoring correctly, got {lm_honorific.reason}")

    print("all smoke tests passed")
