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
  - Two further JUDGMENT framings tried later (PRD §9.31) also failed:
    context-anchored forced-choice (worse than the original -- wrong in
    both directions now, not just consistently yes-biased) and pairwise
    comparison (below chance, and reasoning mode never finished within
    300 tokens). Three judgment framings, three different failure modes
    -- treated as confirmation this model cannot reliably JUDGE this
    distinction when asked directly, not as "the right prompt hasn't
    been found yet."
  - What DOES work (PRD §9.34/§9.35), found by changing the TASK rather
    than the wording: SCORING instead of judging. sentence_logprob()
    below asks nothing -- it teacher-forces the model over the filled
    sentence and reads off its own log-probability, which cannot be
    acquiescence-biased because there is no question to agree with. "Is
    the correct form more probable than the distractor under the
    model's own distribution" turned out to be answerable even though
    "is the distractor also acceptable, yes or no" was not. Tested at
    N=20/skill across all 11 skills: 100% for 9 skills, explainably
    weaker for noun_counter (the generic counter 개 sometimes
    outscores a correct but rarer counter like 권/분 on raw frequency),
    and a clean 0% for subj_topic -- which is not a failure of the
    method, it independently reconfirms PRD §9.11's own honest framing
    that subj_topic's correctness is a pedagogical convention, not a
    hard rule a trained model's natural distribution would reflect.

This is a judge, not ground truth -- a solver disagreement is a signal to
investigate (same spirit as every other validator in this codebase),
not an automatic verdict that flips an item's status. Check (c) is
usable as implemented; check (b) is reported for reference only -- its
known yes-bias means a "no also-acceptable options found" result here
must NOT be read as "this item is confirmed unambiguous." log_likelihood_check
is the one to actually use for "is this distractor also correct",
scoped to LOGPROB_RELIABLE_SKILLS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"  # swapped from Qwen3-4B 2026-10-20, per
# explicit user instruction ("pakai exaone 7.8b" -> "ketiganya" -> "lanjut"). THIS IS THE
# HIGHEST-RISK OF THE THREE SWAPS: Qwen3-4B's judgment numbers for this exact file are already
# PUBLISHED in the paper (Section IV-B / PRD Sec.9.36: EXAONE-4.0-1.2B scored 48.2% vs Qwen3-4B's
# 87.5% on grammaticality-judgment; PRD Sec.9.50: Qwen3-4B scored 57.7% on the separate
# semantic-plausibility task). Those numbers are for EXAONE-4.0-1.2B, NOT EXAONE-3.5-7.8B, and a
# "65.4% vs 57.7%" EXAONE-3.5-7.8B-beats-Qwen3 result was discussed earlier in this project's
# chat history but could NOT be located in docs/PRD.md, any committed results JSON, or git
# history when this swap was made -- treat that number as UNCONFIRMED, not as settled evidence.
# Before this swap is final: re-run validate_log_likelihood.py and validate_novelty1.py on a real
# GPU with this model and compare against the committed Qwen3-4B results
# (log_likelihood_results.json, 1_validation_results.json). If EXAONE-3.5-7.8B regresses either
# check(c) key-acceptance or the log-likelihood gap, revert _MODEL_NAME to "Qwen/Qwen3-4B" and/or
# rewrite the affected paper section -- do not leave both the code and the paper's cited numbers
# silently mismatched. Also requires trust_remote_code=True and
# src/patch_exaone35_remote_code.py (see that script's docstring).

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

# PRD §9.34/§9.35: skills where "correct form scores higher log-prob than
# the distractor" was empirically confirmed reliable, at N~=20/skill
# against every currently curated pair. subj_topic is deliberately
# excluded -- its 0% isn't a method failure, it independently confirms
# PRD §9.11 (no hard rule for the model's own distribution to reflect).
# noun_counter is included despite being weaker than the other 9 (the
# generic counter 개 can outscore a correct but rarer counter on raw
# frequency) because it is still far better than chance and the failure
# mode is understood, not mysterious -- callers that need the stronger
# guarantee can filter it out themselves using LOGPROB_SKILL_NOTES.
LOGPROB_RELIABLE_SKILLS = frozenset({
    "particle_loc", "particle_obj", "honorific", "time_range", "tense_past",
    "irregular_conj", "adj_relativizer", "verb_choice", "mod_adverb", "noun_counter",
})
LOGPROB_SKILL_NOTES = {
    "noun_counter": "weaker than the other 9 skills: the generic counter 개 "
                     "sometimes outscores a correct but rarer counter (권, 분) "
                     "on raw frequency alone -- real risk, not noise.",
}


@dataclass
class SolverResult:
    picked: str | None        # the solver's single best-answer pick, check (c)
    key_accepted: bool         # check (c) verdict -- trustworthy, use this
    also_acceptable: dict[str, bool]  # check (b), per non-key option -- NOT trustworthy,
                                        # known yes-bias (see module docstring), reference only


@dataclass
class LogLikelihoodResult:
    valid: bool        # True if correct form scored higher, OR skill is out of scope (skipped)
    gap: float | None  # correct_logprob - distractor_logprob; None if skipped
    reason: str


class SolverCheckProvider:
    def __init__(self, model_name: str = _MODEL_NAME):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
        self.model.eval()

    @torch.no_grad()
    def sentence_logprob(self, sentence: str) -> float:
        """Total log-probability of `sentence` under the model's own next-
        token distribution, teacher-forced -- a pure scoring call, not a
        judgment prompt, so acquiescence/yes-bias (see module docstring)
        cannot apply. Reuses the same already-loaded model as check (c),
        no extra model dependency.
        """
        enc = self.tok(sentence, return_tensors="pt").to("cuda")
        ids = enc["input_ids"]
        n_tokens = ids.shape[1] - 1
        out = self.model(ids, labels=ids)
        return -out.loss.item() * n_tokens

    def log_likelihood_check(self, skill_id: str, stem: str, blank_marker: str,
                              correct_text: str, distractor_text: str) -> LogLikelihoodResult:
        if skill_id not in LOGPROB_RELIABLE_SKILLS:
            return LogLikelihoodResult(True, None,
                                        f"log-likelihood check not validated for skill '{skill_id}', skipped")
        filled_correct = stem.replace(blank_marker, correct_text, 1)
        filled_distractor = stem.replace(blank_marker, distractor_text, 1)
        gap = self.sentence_logprob(filled_correct) - self.sentence_logprob(filled_distractor)
        if gap > 0:
            return LogLikelihoodResult(True, gap, f"correct form scores {gap:.2f} higher, as expected")
        note = f" ({LOGPROB_SKILL_NOTES[skill_id]})" if skill_id in LOGPROB_SKILL_NOTES else ""
        return LogLikelihoodResult(False, gap, f"distractor scores {-gap:.2f} higher than the correct "
                                                f"form -- suspect pair, flag for review{note}")

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

    print("\n--- log_likelihood_check (PRD §9.35, the working replacement for check (b)) ---")
    ll_honorific = solver.log_likelihood_check("honorific", "할머니께서 ___.", "___", "쉬세요", "쉬어요")
    assert ll_honorific.valid and ll_honorific.gap is not None and ll_honorific.gap > 0, (
        f"honorific is a confirmed-reliable skill, correct form must score higher, got {ll_honorific.reason}")
    print(f"[OK] honorific: {ll_honorific.reason}")
    ll_subj = solver.log_likelihood_check("subj_topic", "질문: 누가 커피를 마셔요? 답: ___ 마셔요.", "___", "제가", "저는")
    assert ll_subj.valid and ll_subj.gap is None, (
        "subj_topic must be valid=True via the explicit SKIP path (gap=None), not by coincidentally "
        f"scoring correctly -- got {ll_subj.reason}")
    print(f"[OK] subj_topic correctly skipped (not scored): {ll_subj.reason}")

    print("\nall smoke tests passed (mechanism verified; see summary above for actual pass rates)")
