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
  (f) generic fillers, to reach 4 options (PRD §9.49): daily_drill's
      existing ContentProvider protocol and the already-validated
      Urnings/IRT math assume a fixed 25% guess-rate (PRD §9.21 -- every
      hand-coded skill was deliberately forced from 2 to 4 options for
      this reason), so a curriculum item cannot stay a bare 2-way
      correct/distractor pair if it is to be scheduled through the same
      pipeline. Reuses exaone_provider.py's own proven pattern (bare,
      generic filler words like "학교"/"사과", individually checked to be
      definitely wrong for THIS slot, not independently diagnostic) --
      but there is no per-skill tag validator to check against here like
      the 11 hand-coded skills have. First attempt reused
      lm_plausibility_gap() (check (d)'s raw-SUM score) directly and
      failed badly: production rate collapsed from 14/41 to 5/41,
      because kiwipiepy's raw summed score favors SHORTER sentences
      regardless of grammaticality -- whenever the correct span was a
      longer multi-word phrase (e.g. entry 144's "하기가 어려워요"), every
      short bare-noun filler scored HIGHER (gap negative) than the
      correct answer even though the resulting sentence is nonsense,
      because it's shorter, not because it's better. Found by directly
      inspecting the gap values for known-good cases, not assumed.
      Fixed with a SEPARATE, length-normalised gap (_filler_gap() below,
      average score per morpheme, not raw sum) used ONLY for this check
      -- check (d)'s gibberish filter keeps the original raw-sum signal
      unchanged, since it was independently validated in that form and
      compares two FULL sentences of similar length to begin with.
      Threshold calibration, revised TWICE after each pass was checked
      against more data rather than trusted on a small sample (PRD
      §9.49): pass 1 tested 3 entries, saw gaps of 2.2-3.9, and set the
      threshold at 1.5 -- re-measuring on 2 more real entries ("하지만"
      vs "그러나", "-아/어 두다") showed every genuine filler gap
      clustering at 0.4-1.4, all below that threshold, so production
      collapsed (7/41) for the wrong reason: correctly-directional
      fillers rejected on margin, not actually unsafe. Lowered to 0.3 --
      still comfortably above 0 (the only sign that actually matters:
      the filler must be LESS plausible than the correct answer, not by
      a specific margin) and below every genuinely-safe case observed
      across both calibration rounds. Re-measured on the full 41-entry
      sweep at this final threshold: 13/41 produced, 9/13 (69%) solver
      agreement -- back in line with the pre-filler-check baseline
      (14/41, 64%, PRD §9.48), confirming the filler mechanism itself
      doesn't cost production rate once correctly calibrated. The real
      safety guarantee here is mostly the curated candidate pool itself
      (bare content nouns that don't fit a grammar slot almost by
      construction, same logic as exaone_provider.py's own fillers) --
      this numeric check is a secondary guard against a candidate that
      somehow comes out MORE plausible than the correct answer, not the
      primary source of safety, so a small positive margin is enough. If
      fewer than 2 of the 4 candidates clear that bar for a given
      sentence, the item is discarded rather than shipped with only 2
      or 3 options.
      One anomaly seen in the 41-sweep, disclosed rather than hidden:
      entry "-느니" (352)'s two fillers both got flagged
      also_acceptable=True by solver_check's check (b) -- but check (b)
      is ALREADY documented (solver_check.py's own module docstring) as
      having a persistent yes-bias on this solver model, independent of
      this module, so this is read as that known limitation recurring,
      not new evidence the filler itself is unsafe (the lm_plausibility-
      based filler check that actually gates production is a SEPARATE,
      already-validated signal from check (b)'s judgment prompt).
There is no per-item-type morphological tag check here, unlike the 11
hand-coded skills -- 524 grammar points span far too many structural
types to hand-verify each one the way irregular_conj/adj_relativizer/etc.
were. kagas_code is left None for every item this module produces: the
official 7-code KAGAS mapping (PRD §9.8) was derived by hand for the 11
existing skills and does not generalise automatically to the other ~500
curriculum points without the same care, so guessing one here would be a
false precision this module does not claim.

skill_id is set to f"curriculum_{entry.number}" -- PER GRAMMAR POINT,
not one shared bucket (user's explicit choice, PRD §9.49), so each of
the 524 points gets its own independent Urnings mastery track through
the existing per-skill scheduling machinery, same as the 11 hand-coded
skills. This has NOT been validated at this scale (§9.12-9.38's cold-
start/convergence results were all measured at ~11 skills) -- flagged
as an open validation gap, not assumed to transfer.

Known remaining quality limitation, partially (not fully) mitigated by
prompting (PRD §9.48): at intermediate/advanced TOPIK levels, many
grammar points are near-synonyms of each other (e.g. "-(으)ㄹ 리(가)
만무하다" vs "-(으)ㄹ 리가 없다", "N에 비하면" vs "N에 비해"), so a
distractor built from a genuinely different grammar point is often ALSO
an acceptable native phrasing, not a true error -- the pair ends up
testing "which synonym did the textbook pick" rather than a right/wrong
distinction. The representative 41-sample run first measured this at
13/41 produced, 6/13 (46%) solver round-trip agreement (PRD §9.46).
_GEN_PROMPT was then revised to explicitly forbid swapping in a
different-but-also-correct expression, with concrete examples of what
NOT to do -- re-measured at 14/41 produced, 9/14 (64%) agreement (PRD
§9.48): a real improvement (and several previously-bad items, e.g.
"-아/어 봤자" and "-(으)ㄹ 리(가) 만무하다", now self-reject via check
(a)/(b)/(d) instead of shipping a flawed pair), but NOT a full fix --
"N에 비하면" vs "에 비해", "N(이)란" vs "은", and "하지만" vs its own
near-synonyms still slip through for the same underlying reason: some
TOPIK grammar points genuinely don't have a "wrong" near-neighbor to
contrast against, only other correct ones. No automatic check here
distinguishes "wrong" from "a different correct synonym" in the general
case -- flagged for human review, not silently shipped as
equal-confidence to the 11 hand-coded skills.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import torch
from kiwipiepy import Kiwi
from transformers import AutoModelForCausalLM, AutoTokenizer

from curriculum_provider import CurriculumEntry
from distractor_validator import lm_plausibility_gap

_MODEL_NAME = "Qwen/Qwen3-4B"  # tested against EXAONE-4.0-1.2B first, see module docstring
_GIBBERISH_GAP_THRESHOLD = 8.0  # see module docstring, check (d)
# Reuses exaone_provider.py's own proven-safe bare-noun filler pool
# (_EF_FILLER_WORDS) rather than inventing a new one -- see check (f).
_FILLER_CANDIDATES = ("학교", "사과", "저기", "안녕")
_FILLER_GAP_THRESHOLD = 0.3  # length-normalised, see check (f) in module docstring
_KIWI = Kiwi()


def _filler_gap(correct_sentence: str, candidate_sentence: str) -> float:
    """Per-morpheme average score gap -- deliberately NOT the same raw-sum
    signal as lm_plausibility_gap(), see check (f) in the module docstring
    for why a raw sum is the wrong comparison when the correct span is a
    much longer phrase than a short filler candidate.
    """
    def avg_score(sentence: str) -> float:
        tokens, score = _KIWI.analyze(sentence, top_n=1)[0]
        return score / max(len(tokens), 1)
    return avg_score(correct_sentence) - avg_score(candidate_sentence)

_GEN_PROMPT = (
    '한국어 문법 학습 문제를 만들어줘.\n'
    '문법 포인트: {grammar} ({meaning})\n'
    '예문: "{example}"\n\n'
    '1단계: 이 예문에서 "{grammar}" 문법을 실현하는 정확한 부분(어절 또는 조사/어미)을 찾아줘.\n'
    '2단계: 그 부분 대신 쓰면 이 문맥에서 실제로 비문(문법적으로 틀린 문장)이 되는 표현을 하나 만들어줘.\n\n'
    '중요-- "틀린표현"은 반드시 이 문맥에서 비문이어야 해. 절대로 하면 안 되는 것:\n'
    '- 의미가 같거나 비슷한 다른 정식 문법/어미로 바꾸는 것 (예: "-(으)ㄹ 리가 만무하다"를 "-(으)ㄹ 리가 없다"로, '
    '"N에 비하면"을 "N에 비해"로 바꾸는 식 -- 둘 다 자연스러운 정답이라서 안 됨)\n'
    '- 문맥상 똑같이 자연스럽게 들리는 다른 표현으로 바꾸는 것\n'
    '허용되는 흔한 학습자 실수: 틀린 시제, 틀린 존댓말 레벨, 조사 생략/오용, 불규칙 활용을 틀리게 적용, '
    '비슷하게 생겼지만 뜻이 다른 문법과 혼동.\n\n'
    '다음 형식으로만 답해줘, 따옴표나 괄호 없이:\n'
    '원래표현: 예문에서 찾은 정확한 부분\n'
    '틀린표현: 이 문맥에서 비문이 되는, 학습자가 실수로 쓸 법한 표현'
)
_STRIP_CHARS = ' \t\n\r\'"“”[]()-'


@dataclass
class CurriculumItem:
    # Duck-typed to match exaone_provider.GeneratedItem's interface
    # (skill_id, stem, blank_marker, options, answer, kagas_code,
    # primary_distractor_label) so answer_processing.submit_grammar_answer
    # works unmodified -- see module docstring, check (f).
    skill_id: str              # f"curriculum_{entry.number}" -- one Urnings track per grammar point
    entry: CurriculumEntry
    stem: str                  # the example sentence with the span blanked out
    blank_marker: str
    options: dict[str, str]    # 4 options, labels A-D
    answer: str                # which label (A-D) is correct
    primary_distractor_label: str
    correct_span: str          # kept for introspection/debugging, duplicates options[answer]
    distractor_span: str       # kept for introspection/debugging, duplicates options[primary_distractor_label]
    kagas_code: None = None    # deliberately None, see module docstring


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
    def __init__(self, model_name: str = _MODEL_NAME, seed: int | None = None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
        self.model.eval()
        self.rng = random.Random(seed)

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        messages = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True).to("cuda")
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _pick_fillers(self, entry: CurriculumEntry, stem: str, blank_marker: str,
                       correct: str, distractor: str, n: int = 2) -> list[str] | None:
        """check (f): generic fillers, accepted only if clearly WORSE than
        the correct answer when substituted in -- see module docstring.
        """
        fillers: list[str] = []
        for cand in _FILLER_CANDIDATES:
            if cand in (correct, distractor) or cand in fillers:
                continue
            candidate_sentence = stem.replace(blank_marker, cand, 1)
            gap = _filler_gap(entry.example, candidate_sentence)
            if gap > _FILLER_GAP_THRESHOLD:
                fillers.append(cand)
            if len(fillers) == n:
                break
        return fillers if len(fillers) == n else None

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

        fillers = self._pick_fillers(entry, stem, "___", original, distractor)
        if fillers is None:
            return None  # check (f): couldn't build 2 safely-wrong generic fillers for this sentence

        option_texts = [original, distractor, *fillers]
        labels = ["A", "B", "C", "D"]
        self.rng.shuffle(labels)
        options = dict(zip(labels, option_texts))
        answer_label, distractor_label = labels[0], labels[1]

        return CurriculumItem(skill_id=f"curriculum_{entry.number}", entry=entry, stem=stem,
                               blank_marker="___", options=options, answer=answer_label,
                               primary_distractor_label=distractor_label,
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

    generator = CurriculumItemGenerator(seed=0)
    solver = SolverCheckProvider()

    produced, solver_ok = 0, 0
    for entry in sample:
        item = generator.generate(entry)
        status = "SKIP (failed check a/b/d/e/f)" if item is None else "OK"
        print(f"[{entry.number}] {entry.grammar}: {status}")
        if item is None:
            continue
        produced += 1
        assert item.skill_id == f"curriculum_{entry.number}"
        assert len(item.options) == 4 and set(item.options) == {"A", "B", "C", "D"}
        assert item.answer in item.options and item.primary_distractor_label in item.options
        assert item.answer != item.primary_distractor_label
        print(f"    stem={item.stem!r}  options={item.options!r}  answer={item.answer!r}")
        result = solver.check(item.stem, item.blank_marker, item.options, item.answer)
        solver_ok += int(result.key_accepted)
        print(f"    solver blind pick: {result.picked}  (key_accepted={result.key_accepted})  "
              f"also_acceptable={result.also_acceptable}")

    print(f"\nproduced {produced}/{len(sample)} items; solver round-trip accepted "
          f"{solver_ok}/{produced if produced else 1}")
    assert produced > 0, "the generator must produce at least one item on this real sample"
    print("\nall smoke tests passed (mechanism verified; see counts above for actual quality)")
