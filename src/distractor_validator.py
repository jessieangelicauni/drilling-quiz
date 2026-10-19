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


def validate_cloze_option(stem_with_blank: str, blank_marker: str, option: str,
                           expected_tag: str) -> ValidationResult:
    """Does substituting `option` into the blank parse as `expected_tag`
    (e.g. "JKB") at that position? Catches an option that isn't even the
    right kind of particle for the grammar point being tested.
    """
    if blank_marker not in stem_with_blank:
        raise ValueError(f"{blank_marker!r} not found in stem")
    filled = stem_with_blank.replace(blank_marker, option, 1)
    tags_at_option = [tag for form, tag in particle_tags(filled) if form == option]
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

    print("all smoke tests passed")
