"""
EV/smart-factory vocabulary substitution layer (user request, 2026-10-20,
clarified via AskUserQuestion: "Kosakata/konteks manufaktur, grammar
tetap standar" -- vocabulary/context substitution ON TOP OF standard
TOPIK grammar, NOT a separate vocational curriculum).

IMPORTANT correction (user, same session): the first version of this
module hand-coded a fixed noun->noun mapping table (학생->작업자,
영화->로봇, etc.) written by me directly. User explicitly rejected this:
"saya ingin semuanya itu llm yang buat bukan agent claude" -- the actual
domain-vocabulary CONTENT decision (which word becomes what) must come
from the project's own LLM pipeline (Qwen3-4B, already used throughout
this codebase), not be authored by the coding agent. Rewritten around
that: the LLM is asked, per sentence, which noun to swap and what to
swap it for; this module's own code does ONLY the deterministic,
mechanical part -- applying that one substitution and fixing the
following particle's batchim allomorph (이/가, 을/를, 은/는, 과/와) --
the same "LLM proposes content, code guarantees grammatical correctness"
division of labour this entire codebase already uses everywhere else
(distractors, items, feedback). No hand-written word-pair table exists
here any more.

Design: substitution runs on a CurriculumEntry's `example` sentence
BEFORE it reaches curriculum_item_generator.py, producing a new entry
with the same grammar/meaning/level/unit but a domain-flavoured example.
curriculum_item_generator.generate() then runs completely UNCHANGED on
whatever example sentence it's given -- the grammar-point-finding and
distractor-generation logic has no idea the vocabulary is different, by
design, since the whole point is grammar stays standard.

Model reuse, not a 4th separate load: PRD §9.49 already found that
loading 3 independent models at once (EXAONE + 2x separate Qwen3-4B
instances) OOMs a 15GB GPU. DomainVocabSubstitutor takes a `generate_fn`
callable instead of loading its own model -- a caller wires it to an
already-loaded CurriculumItemGenerator's `_generate` method, so this
adds zero extra GPU memory.
"""
from __future__ import annotations

from dataclasses import replace

from kiwipiepy import Kiwi

from curriculum_provider import CurriculumEntry
from distractor_validator import has_batchim

_KIWI = Kiwi()

_DOMAIN_PROMPT = (
    '아래 한국어 문장을 "스마트 팩토리 / 전기차(EV) 제조" 맥락으로 바꾸고 싶어. '
    '문법과 문장 구조는 완전히 그대로 두고, 그 문맥에 안 맞는 일반 명사 딱 하나만 골라서 '
    '네가 알고 있는 스마트 팩토리/전기차 제조 현장 어휘 중에서 가장 자연스러운 단어로 바꿔줘.\n'
    '문장: "{example}"\n\n'
    '다음 형식으로만 답해줘, 설명 없이:\n'
    '원래단어: 문장에 있는 그대로의 명사\n'
    '바뀐단어: 스마트 팩토리/EV 제조 맥락에 맞는 새 단어'
)
_STRIP_CHARS = ' \t\n\r\'"“”[]()-'

# Particle allomorph pairs: (batchim-final form, vowel-final form). When
# the substituted noun's batchim differs from the original word's, the
# following particle must be re-picked -- same batchim rule already
# established elsewhere in this codebase (distractor_validator.py).
_PARTICLE_ALLOMORPHS = [("이", "가"), ("을", "를"), ("은", "는"), ("과", "와")]
_SUBSTITUTABLE_TAGS = ("NNG", "NNP")
_PARTICLE_TAGS = ("JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC")


def _corrected_particle(noun: str, current_form: str) -> str:
    for batchim_form, vowel_form in _PARTICLE_ALLOMORPHS:
        if current_form in (batchim_form, vowel_form):
            return batchim_form if has_batchim(noun) else vowel_form
    return current_form  # not a batchim-sensitive particle, leave unchanged


def _clean(text: str) -> str:
    return text.strip(_STRIP_CHARS)


def _parse_response(text: str) -> tuple[str, str] | None:
    if "원래단어:" not in text or "바뀐단어:" not in text:
        return None
    original = _clean(text.split("원래단어:")[1].split("바뀐단어:")[0])
    new_word = _clean(text.split("바뀐단어:")[1].splitlines()[0])
    if not original or not new_word:
        return None
    return original, new_word


def _apply_substitution(entry: CurriculumEntry, original: str, new_word: str) -> CurriculumEntry | None:
    """The deterministic half: swap exactly the ONE token the LLM named
    (must be a real noun token kiwipiepy recognises, not a hallucinated
    span) and fix the following particle's batchim allomorph if needed.
    """
    tokens = list(_KIWI.analyze(entry.example, top_n=1)[0][0])
    morphs = [(t.form, t.tag) for t in tokens]
    idx = next((i for i, (form, tag) in enumerate(morphs)
                if form == original and tag in _SUBSTITUTABLE_TAGS), None)
    if idx is None:
        return None  # not a real standalone noun token -- discard rather than guess
    morphs[idx] = (new_word, morphs[idx][1])
    if idx + 1 < len(morphs) and morphs[idx + 1][1] in _PARTICLE_TAGS:
        part_form, part_tag = morphs[idx + 1]
        morphs[idx + 1] = (_corrected_particle(new_word, part_form), part_tag)
    try:
        new_example = _KIWI.join(morphs)
    except Exception:
        return None
    return replace(entry, example=new_example)


class DomainVocabSubstitutor:
    def __init__(self, generate_fn):
        self._generate = generate_fn

    def substitute(self, entry: CurriculumEntry) -> CurriculumEntry | None:
        prompt = _DOMAIN_PROMPT.format(example=entry.example)
        text = self._generate(prompt)
        parsed = _parse_response(text)
        if parsed is None:
            return None
        original, new_word = parsed
        if original not in entry.example or new_word == original:
            return None  # check (a)/(b): named word must be real, replacement must differ
        return _apply_substitution(entry, original, new_word)


if __name__ == "__main__":
    from curriculum_item_generator import CurriculumItemGenerator
    from curriculum_provider import parse_curriculum_docx

    entries = parse_curriculum_docx("Korean_Grammar_Curriculum.docx")
    by_num = {e.number: e for e in entries}
    generator = CurriculumItemGenerator(seed=0)
    substitutor = DomainVocabSubstitutor(generator._generate)

    sample = [by_num[n] for n in ["1", "18", "40", "11", "4"]]
    produced = 0
    for entry in sample:
        result = substitutor.substitute(entry)
        status = "SKIP" if result is None else "OK"
        print(f"[{entry.number}] {entry.example!r} -> {status}")
        if result is not None:
            produced += 1
            print(f"    {result.example!r}")
    assert produced > 0, "must produce at least one domain-substituted item on this real sample"

    print(f"\nproduced {produced}/{len(sample)} on hand-picked sample")
    print("all smoke tests passed (mechanism verified; see counts above for actual coverage)")
