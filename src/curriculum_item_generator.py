"""
Generates MCQ items from an arbitrary CurriculumEntry (curriculum_provider.py),
for the ~247/524 grammar points (PRD §9.46) whose surface form can't be
found as a literal substring in the curriculum's own example sentence --
i.e. the ones that need real morphological/grammatical understanding to
locate and vary, not just string matching. User's explicit choice for
this case (2026-10-20): allow free LLM generation with a LOOSER
validator than the 11 hand-coded skills' deterministic construction,
rather than hand-build a verified template for each of 524 points.

Model choice tested, not assumed (PRD §9.46): EXAONE-4.0-1.2B (the
historical default for generation elsewhere in this codebase) was tried
first on this exact task and performed poorly -- on a 7-case spot check it
mostly failed to locate a real substring of the example sentence at all,
sometimes answering with a description of the grammar point instead of a
located span. This matches §9.36's independent finding (EXAONE scores
48.2% vs Qwen3's 87.5% on a different grammaticality-scoring task) rather
than contradicting it. Qwen3-4B is used here instead, consistent with
that finding, not out of habit.

Validation, explicitly looser than exaone_provider.py's skills, and
disclosed as such rather than presented with equal confidence:
  (a) the "original span" the model names must actually occur in the
      curriculum's own example sentence (sanity check against
      hallucination -- the ONE check applied unconditionally); if it
      doesn't, the item is discarded rather than shipped with a guessed
      blank position.
  (b) the distractor must differ from the correct span and be non-empty.
  (c) solver_check.py's round-trip check (blind key-acceptance) is run
      and recorded, but -- as with the 11 hand-coded skills -- is a
      signal to review, not a gate that silently discards items.
  (d) gibberish filter: on a 12-sample spot check, 2/7 produced items had
      a distractor that was not a real learner error at all but a
      malformed/non-word string (e.g. "합니다니다", "바르요") -- the kind
      of defect check (a)/(b) cannot catch since they only check for a
      real, non-empty, different string, not a real WORD. Found by
      reusing distractor_validator.py's lm_plausibility_gap() (PRD §9.34)
      on the whole correct- vs distractor-filled sentence: both known-bad
      cases scored a gap of 18.2/9.8, while all 5 known-good (real-word,
      merely wrong) distractors scored 0.1-4.1 -- a clean separation
      confirmed on this sample, not assumed. Gated at gap > 8.0 (the
      midpoint of that separation). This is a coarse "is this even real
      Korean" filter, not a grammatical-category check like the 11
      hand-coded skills have -- it will not catch a distractor that is a
      real word but the wrong grammar point.
  (e) empty-context filter: on the larger, representative 41-entry sweep
      (every 13th of 524, PRD §9.46), several intermediate/advanced
      grammar points (e.g. "-(으)며", "-(으)ㄹ 듯 말 듯 하다", "-(으)ㄹ
      리(가) 만무하다") name a span covering nearly the WHOLE example
      sentence (these points connect two full clauses, unlike beginner
      particle/ending points), so blanking it left a stem of just "___"
      with zero surrounding context -- useless as a cloze item for a
      learner or for solver_check's round-trip. Rejected when nothing but
      the blank marker and punctuation would remain in the stem.
There is no per-item-type morphological tag check here, unlike the 11
hand-coded skills -- 524 grammar points span far too many structural
types to hand-verify each one the way irregular_conj/adj_relativizer/etc.
were. kagas_code is left None for every item this module produces: the
official 7-code KAGAS mapping (PRD §9.8) was derived by hand for the 11
existing skills and does not generalise automatically to the other ~500
curriculum points without the same care, so guessing one here would be a
false precision this module does not claim.

Known remaining quality limitation, NOT fixed by any check above (PRD
§9.46): at intermediate/advanced TOPIK levels, many grammar points are
near-synonyms of each other (e.g. "-(으)ㄹ 리(가) 만무하다" vs "-(으)ㄹ
리가 없다", "N에 비하면" vs "N에 비해"), so a distractor built from a
genuinely different grammar point is often ALSO an acceptable native
phrasing, not a true error -- the pair ends up testing "which synonym did
the textbook pick" rather than a right/wrong distinction. This is why the
representative 41-sample run's solver round-trip rate (7/16, see
__main__) is far below the hand-picked 12-sample's 7/7: that sample was
unintentionally all beginner-level, unambiguous points. No automatic
check here distinguishes "wrong" from "a different correct synonym" --
flagged for human review, not silently shipped as equal-confidence to the
11 hand-coded skills.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from curriculum_provider import CurriculumEntry
from distractor_validator import lm_plausibility_gap

_MODEL_NAME = "Qwen/Qwen3-4B"  # tested against EXAONE-4.0-1.2B first, see module docstring
_GIBBERISH_GAP_THRESHOLD = 8.0  # see module docstring, check (d)

_GEN_PROMPT = (
    '한국어 문법 학습 문제를 만들어줘.\n'
    '문법 포인트: {grammar} ({meaning})\n'
    '예문: "{example}"\n\n'
    '이 예문에서 "{grammar}" 문법을 실현하는 정확한 부분(어절 또는 조사/어미)을 찾아줘.\n'
    '그 다음, 학습자가 흔히 저지르는 실수로 그 부분 대신 쓸 법한 틀린 표현 하나를 만들어줘.\n\n'
    '다음 형식으로만 답해줘, 따옴표나 괄호 없이:\n'
    '원래표현: 예문에서 찾은 정확한 부분\n'
    '틀린표현: 학습자가 실수로 쓸 법한 틀린 표현'
)
_STRIP_CHARS = ' \t\n\r\'"“”[]()-'


@dataclass
class CurriculumItem:
    entry: CurriculumEntry
    stem: str                 # the example sentence with the span blanked out
    blank_marker: str
    correct_span: str
    distractor_span: str
    kagas_code: None = None   # deliberately None, see module docstring


def _clean(text: str) -> str:
    return text.strip(_STRIP_CHARS)


def _parse_response(text: str) -> tuple[str, str] | None:
    if "원래표현:" not in text or "틀린표현:" not in text:
        return None
    original = _clean(text.split("원래표현:")[1].split("틀린표현:")[0])
    distractor = _clean(text.split("틀린표현:")[1].splitlines()[0])
    if not original or not distractor:
        return None
    return original, distractor


class CurriculumItemGenerator:
    def __init__(self, model_name: str = _MODEL_NAME):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
        self.model.eval()

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        messages = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True).to("cuda")
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate(self, entry: CurriculumEntry) -> CurriculumItem | None:
        prompt = _GEN_PROMPT.format(grammar=entry.grammar, meaning=entry.meaning, example=entry.example)
        text = self._generate(prompt)
        parsed = _parse_response(text)
        if parsed is None:
            return None
        original, distractor = parsed
        if original not in entry.example:
            return None  # check (a): the named span must really be in the example, not hallucinated
        if distractor == original or not distractor:
            return None  # check (b)
        distractor_sentence = entry.example.replace(original, distractor, 1)
        gap = lm_plausibility_gap(entry.example, distractor_sentence)
        if gap > _GIBBERISH_GAP_THRESHOLD:
            return None  # check (d): distractor is likely gibberish, not a real learner error
        stem = entry.example.replace(original, "___", 1)
        if stem.replace("___", "").strip(" .!?,") == "":
            return None  # check (e): the span ate the whole sentence, no context left
        return CurriculumItem(entry=entry, stem=stem, blank_marker="___",
                               correct_span=original, distractor_span=distractor)


if __name__ == "__main__":
    from curriculum_provider import parse_curriculum_docx
    from solver_check import SolverCheckProvider

    entries = parse_curriculum_docx("Korean_Grammar_Curriculum.docx")
    # Evenly spaced across the whole 524-entry document (every 13th entry,
    # ~40 points) rather than the earlier 12-entry hand-picked sample, so
    # the production/quality rate reported below is representative of the
    # full curriculum, not just the beginner-level rows checked first.
    sample = entries[::13]

    generator = CurriculumItemGenerator()
    solver = SolverCheckProvider()

    produced, solver_ok = 0, 0
    for entry in sample:
        item = generator.generate(entry)
        status = "SKIP (failed check a/b/d)" if item is None else "OK"
        print(f"[{entry.number}] {entry.grammar}: {status}")
        if item is None:
            continue
        produced += 1
        print(f"    stem={item.stem!r}  correct={item.correct_span!r}  distractor={item.distractor_span!r}")
        options = {"A": item.correct_span, "B": item.distractor_span}
        result = solver.check(item.stem, item.blank_marker, options, "A")
        solver_ok += int(result.key_accepted)
        print(f"    solver blind pick: {result.picked}  (key_accepted={result.key_accepted})")

    print(f"\nproduced {produced}/{len(sample)} items; solver round-trip accepted "
          f"{solver_ok}/{produced if produced else 1}")
    assert produced > 0, "the generator must produce at least one item on this real sample"
    print("\nall smoke tests passed (mechanism verified; see counts above for actual quality)")
