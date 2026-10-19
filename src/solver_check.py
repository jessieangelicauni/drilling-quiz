"""
Verifier gate checks (b) and (c) from PRD §3.2/Fig. 3, implemented as a
GENERAL check usable on any GeneratedItem -- the two checks §9.12 found
missing (only check (a), morphological realization, existed before this):

  (c) a solver model, blind to the key, still picks the intended answer
  (b) the distractor is not itself also a correct answer

For the 8 skills already built (§9.1-§9.11), this risk is avoided BY
CONSTRUCTION -- each distractor is hand-built to be wrong, not generated
freely and checked after the fact, so (b)/(c) were never strictly
necessary for them. This module exists to test whether that "guaranteed
by construction" assumption actually holds, using an independent judge
(Qwen3-4B, reused from qwen_feedback.py's model but in a new role: not
explaining, but solving blind) rather than assuming it does.

History (found empirically, not assumed -- §9.16):
  - First attempt: one prompt asking the solver to list ALL acceptable
    options at once (testing (b) and (c) together). Result: 4/24 clean
    on a real 8-skill sample, including the solver rejecting
    grammatically well-formed answers and accepting nonsensical ones
    (e.g. accepting "도"/"만" as if they completed a sentence that needs
    a location particle). A one-shot worked example (the fix that
    rescued exaone_provider's OWN generation reliability) made it WORSE,
    not better -- the model started echoing meta-commentary instead of
    answering. Multi-select is apparently too hard a task for this
    model's non-reasoning mode, the same "too much prompt complexity for
    a small model" lesson already learned once for EXAONE-1.2B, now
    found again for Qwen3-4B on a different task.
  - Fix: split into two SEPARATE, narrower tasks instead -- (c) is a
    single best-answer pick (standard MCQ format, with one worked
    example), (b) is a per-option yes/no ("is this ALSO acceptable")
    rather than a combined multi-select. Verified directly: re-parsing
    the first attempt's raw output properly (the quick diagnostic script
    that found this had ITS OWN bug -- checking only the first character
    of the response, which was always "정" from "정답:", not the actual
    answer letter) showed single-best-answer was already working on 3/5
    spot-checked cases where multi-select had looked like a near-total
    failure. The two genuine misses were both `irregular_conj` items,
    where the solver consistently preferred the wrong uncontracted form
    -- a real, specific, narrower finding, not a blanket "distractors are
    bad" conclusion.
  - Check (b) specifically ("is this sentence also acceptable?") turned
    out to have a persistent YES-BIAS: tested on 6 known cases (3
    genuinely acceptable, 3 genuinely wrong, including an ungrammatical
    uncontracted irregular-stem form), the model said "yes, acceptable"
    for the wrong ones too, in BOTH non-reasoning and reasoning mode,
    and with two different prompt framings ("자연스러운가? 예/아니오" and
    the more blunt "틀림/맞음"). Reasoning mode with an 800-token budget
    didn't fix it either. This is NOT a configuration problem fixable by
    more prompt iteration -- treated as a real, tested capability limit
    and reported as such, not silently patched over. Check (b)'s result
    is kept in SolverResult for visibility but EXCLUDED from the
    `unambiguous` verdict (see below) -- a check with a known yes-bias
    would make every item look falsely safe, which is worse than not
    running it at all.

This is a judge, not ground truth -- a solver disagreement is a signal to
investigate (same spirit as every other validator in this codebase),
not an automatic verdict that flips an item's status. Check (c) is
usable as implemented; check (b) is reported for reference only -- its
known yes-bias means a "no also-acceptable options found" result here
must NOT be read as "this item is confirmed unambiguous."
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen3-4B"

_PICK_PROMPT = (
    '다음 한국어 문제의 빈칸(___)에 가장 알맞은 답 하나를 골라줘.\n\n'
    '예시:\n저는 학교___ 공부해요.\nA: 에서\nB: 에\n정답: A\n\n'
    '이제 새 문제:\n{stem}\n{options}\n정답:'
)
_ALSO_ACCEPTABLE_PROMPT = (
    '다음 한국어 문장이 자연스러운 문법 문장인지 판단해줘.\n\n'
    '예시:\n문장: "저는 학교에서 공부해요."\n자연스러운 문장인가? 예\n\n'
    '이제 새 문장:\n문장: "{filled_sentence}"\n자연스러운 문장인가?'
)
_ANSWER_LETTER_RE = re.compile(r"([A-Z])")


@dataclass
class SolverResult:
    picked: str | None        # the solver's single best-answer pick, check (c)
    key_accepted: bool         # check (c) verdict -- trustworthy, use this
    also_acceptable: dict[str, bool]  # check (b), per non-key option -- NOT trustworthy,
                                        # known yes-bias (see module docstring), reference only


class SolverCheckProvider:
    def __init__(self, model_name: str = _MODEL_NAME):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True).to("cuda")
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _pick_best(self, stem: str, options: dict[str, str]) -> str | None:
        option_lines = "\n".join(f"{label}: {text}" for label, text in options.items())
        # 40, not 10 -- found empirically (same pattern as _is_also_acceptable):
        # this model often echoes the whole question back before answering
        # in non-reasoning mode, so a short budget returns None (not
        # "no answer exists", just "cut off before reaching it").
        text = self._generate(_PICK_PROMPT.format(stem=stem, options=option_lines), max_new_tokens=40)
        answer_part = text.split("정답:")[-1]  # ignore the echoed question/options when searching
        match = _ANSWER_LETTER_RE.search(answer_part)
        picked = match.group(1) if match else None
        return picked if picked in options else None

    def _is_also_acceptable(self, stem: str, blank_marker: str, option_text: str) -> bool:
        filled = stem.replace(blank_marker, option_text, 1)
        # 30, not 5 -- found empirically: this model echoes the whole
        # prompt back before answering in non-reasoning mode, so a short
        # budget cuts it off before the actual 예/아니오 ever appears.
        text = self._generate(_ALSO_ACCEPTABLE_PROMPT.format(filled_sentence=filled), max_new_tokens=30)
        answer_part = text.split("인가?")[-1]  # ignore the echoed question when checking for 아니
        return "예" in answer_part and "아니" not in answer_part

    def check(self, stem: str, blank_marker: str, options: dict[str, str], answer: str) -> SolverResult:
        picked = self._pick_best(stem, options)
        key_accepted = picked == answer

        also_acceptable = {}
        for label, text in options.items():
            if label == answer:
                continue
            also_acceptable[label] = self._is_also_acceptable(stem, blank_marker, text)

        return SolverResult(picked=picked, key_accepted=key_accepted, also_acceptable=also_acceptable)


if __name__ == "__main__":
    from exaone_provider import ExaoneGrammarProvider

    solver = SolverCheckProvider()
    provider = ExaoneGrammarProvider(seed=0)

    skills = ["particle_loc", "particle_obj", "honorific", "time_range", "tense_past",
              "irregular_conj", "adj_relativizer", "subj_topic"]

    results: dict[str, list[SolverResult]] = {}
    for skill_id in skills:
        skill_results = []
        for _ in range(3):
            item_id = provider.next_new_grammar_item(skill_id)
            if item_id is None:
                continue
            item = provider.items[item_id]
            result = solver.check(item.stem, item.blank_marker, item.options, item.answer)
            skill_results.append(result)
            status = "OK" if result.key_accepted else "KEY-MISS"
            print(f"[{status}] {skill_id}: {item.stem} opts={item.options} key={item.answer} "
                  f"picked={result.picked} also_acceptable(unreliable, FYI only)={result.also_acceptable}")
        results[skill_id] = skill_results

    print("\n--- summary (check (c), the trustworthy one) ---")
    total, key_ok = 0, 0
    for skill_id, skill_results in results.items():
        n = len(skill_results)
        n_key_ok = sum(r.key_accepted for r in skill_results)
        total += n
        key_ok += n_key_ok
        print(f"{skill_id:20s} key-accepted {n_key_ok}/{n}")
    print(f"\nOVERALL: key-accepted {key_ok}/{total}")
    print("(check (b) 'also_acceptable' printed above for reference only -- known yes-bias, see module docstring)")

    assert total > 0, "the solver check must have actually run against real generated items"
    print("\nall smoke tests passed (mechanism verified; see summary above for actual pass rates)")
