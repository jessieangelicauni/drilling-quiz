"""
Feedback generation (PRD §2.3/§3.3): Qwen3-4B, two paths gated by
urnings_tracker.should_escalate() -- the uncertainty-gated escalation
Novelty 3 is built on. Supports English and Indonesian.

History of getting here (2026-10-19), because the first two attempts
failed and the fix was neither obvious nor the first thing tried:
  - Qwen3-1.7B, asked to answer in Indonesian directly, ignored the
    instruction and answered in Korean instead.
  - A one-shot Indonesian worked example fixed language compliance (4/4
    answered in Indonesian) but introduced a worse failure: 3/4 trials
    leaked the EXAMPLE's topic into the answer (explaining a
    restaurant-eating sentence as if it were about studying, because the
    one-shot example was about studying) -- fluent, convincing, and wrong.
  - Generate-in-English-then-translate (a different fix attempt) was
    tried next and failed differently and more dangerously: translation
    swapped which particle was "correct" in one case, mistranslated an
    idiom ("see a doctor" -> "menyaksikan dokter" = "witness a doctor"),
    and silently dropped half the explanation in another.
  - Qwen3-1.7B was judged too small for reliable Indonesian output and
    this was nearly recorded as a permanent limitation (English-only).
  - That conclusion was challenged, correctly: the untested variable was
    model SIZE, not the task's feasibility. Qwen3-4B, given the SAME
    one-shot-example technique but with the example diversified to two
    contrastive sentences (reducing overfitting to one topic), got 4/4
    correct and leak-free on both the fast and the detailed (reasoning
    mode) path. ~8GB VRAM, comfortably inside the 16GB budget.

Lesson that generalises beyond this one fix: an "eliminated, not deferred"
conclusion from a single model size is still a conclusion scoped to that
model size, not to the capability itself -- re-test the next size up
before calling something infeasible.

Residual fixed separately (2026-10-19, audit follow-up): Qwen's own
illustrative example sentences sometimes contained a conjugation error
(들어요 written as 듣어요). Both fast_feedback() and detailed_feedback() now
post-process their output through
correct_korean_conjugation() below, a deterministic fix for this one
error pattern, applied to free-form mixed-language text.
"""
from __future__ import annotations

import re
from typing import Literal

import torch
from kiwipiepy import Kiwi
from transformers import AutoModelForCausalLM, AutoTokenizer

_KIWI = Kiwi()

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


# Qwen3-4B again (2026-10-21, PRD §9.71), after a day on EXAONE-3.5-7.8B
# (2026-10-20, swapped on request without re-measuring this task):
#   - detailed_feedback() is the escalation path and relies on reasoning mode
#     (enable_thinking=True); EXAONE-3.5's chat template has no reasoning
#     mode, so on EXAONE that flag was a silent no-op;
#   - Qwen3-4B is the model this module was validated on (4/4 correct and
#     leak-free in both languages, docstring above), and it was clean in
#     Indonesian (PRD §9.67);
#   - it is a different family from the item generator (EXAONE), and the
#     same weights as the key-uniqueness checker (solver_check.py), so the
#     drill holds two models on the GPU, not three.
# bf16 (~8GB), as validated; no trust_remote_code. compare_feedback_models.py
# is the side-by-side reading pack against EXAONE.
_MODEL_NAME = "Qwen/Qwen3-4B"
_DETAILED_MIN_TOKENS = 1200  # empirically the minimum that reliably finishes; see qwen_feedback history

Language = Literal["en", "id"]

# Added 2026-10-20 (PRD §9.64): the English path had NO one-shot example at
# all, unlike Indonesian below -- a bare "Answer ONLY in English" instruction,
# which is exactly the pattern that failed for Qwen3-1.7B on Indonesian until
# a worked example was added (see module docstring). Qwen3-4B apparently
# never needed this for English (high-resource language, never reported
# failing). EXAONE is Korean-first trained (paper Section III-C) and, per a
# live user report, answered in Korean despite this instruction when this
# file was swapped to EXAONE-3.5-7.8B -- untested until now, not assumed
# fine. Same diversified-topic technique as the Indonesian example, applied
# to the path that turned out to need it too.
_EN_FAST_EXAMPLE = (
    'Example:\n'
    'Item: "저는 학교___ 공부해요." Correct answer: "에서". The learner picked "에" (wrong).\n'
    'Answer: "에" is wrong because it marks a destination, not the place where an activity '
    'happens. "에서" is correct because it marks the place where studying takes place. Another '
    'example: "식당에서 밥을 먹어요" (eating at a restaurant), not "식당에 밥을 먹어요".\n\n'
)
_EN_DETAILED_EXAMPLE = (
    'Example:\n'
    'Item: "저는 학교___ 공부해요." Correct answer: "에서". The learner picked "에" (wrong), and has '
    'made this same type of mistake before.\n'
    'Answer: In general, "에" marks the goal/destination of an action, while "에서" marks the '
    'place where the action happens. Here "에" is wrong because studying is not about a goal, '
    'but an activity taking place at school. Another example: "저는 식당에서 밥을 먹어요" (eating at '
    'a restaurant) vs "저는 식당에 가요" (going to a restaurant).\n\n'
)

# Two different topics in ONE example, not one -- a single-topic example is
# what caused the topic-leak failure on the smaller model; diversifying the
# example reduces the model's tendency to copy its specific content.
_ID_FAST_EXAMPLE = (
    'Contoh:\n'
    'Soal: "저는 학교___ 공부해요." Jawaban benar: "에서". Siswa memilih "에" (salah).\n'
    'Jawaban: "에" salah karena menandai tujuan, bukan tempat terjadinya aktivitas. "에서" benar '
    'karena menunjukkan tempat belajar berlangsung. Contoh lain: "식당에서 밥을 먹어요" (makan di '
    'restoran, bukan "식당에 밥을 먹어요").\n\n'
)
_ID_DETAILED_EXAMPLE = (
    'Contoh:\n'
    'Soal: "저는 학교___ 공부해요." Jawaban benar: "에서". Siswa memilih "에" (salah), dan sudah '
    'pernah membuat kesalahan serupa.\n'
    'Jawaban: Secara umum, "에" menandai tujuan suatu tindakan, sedangkan "에서" menandai tempat '
    'tindakan itu berlangsung. Di sini "에" salah karena belajar bukan soal tujuan, melainkan '
    'aktivitas yang terjadi di sekolah. Contoh lain: "저는 식당에서 밥을 먹어요" (makan di restoran) '
    'vs "저는 식당에 가요" (pergi ke restoran).\n\n'
)


class QwenFeedbackProvider:
    def __init__(self, model_name: str = _MODEL_NAME):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
        self.model.eval()

    def fast_feedback(self, stem: str, correct: str, wrong: str, language: Language = "en") -> str:
        """Default path: short, cheap. No reasoning mode."""
        if language == "id":
            prompt = (
                f'{_ID_FAST_EXAMPLE}'
                f'Sekarang soal baru:\n'
                f'Soal: "{stem}" Jawaban benar: "{correct}". Siswa memilih "{wrong}" (salah).\n'
                f'Jelaskan dalam 1-2 kalimat singkat, HANYA dalam Bahasa Indonesia, kenapa "{wrong}" '
                f'salah dan "{correct}" benar.\nJawaban:'
            )
        else:
            prompt = (
                f'{_EN_FAST_EXAMPLE}'
                f'Now a new item:\n'
                f'Korean grammar item: "{stem}" The correct answer is "{correct}".\n'
                f'A learner picked "{wrong}" instead, which is wrong.\n'
                f'Explain in English, in 1-2 short sentences, why "{wrong}" is wrong here and why '
                f'"{correct}" is correct.\nAnswer ONLY in English.'
            )
        return correct_korean_conjugation(self._generate(prompt, enable_thinking=False, max_new_tokens=150))

    def detailed_feedback(self, stem: str, correct: str, wrong: str, language: Language = "en") -> str:
        """Escalation path (urnings_tracker.should_escalate() gates this):
        thorough, contrastive, reasoning mode. Needs a large token budget
        -- see _DETAILED_MIN_TOKENS.
        """
        if language == "id":
            prompt = (
                f'{_ID_DETAILED_EXAMPLE}'
                f'Sekarang soal baru:\n'
                f'Soal: "{stem}" Jawaban benar: "{correct}". Siswa memilih "{wrong}" (salah), dan '
                f'sudah pernah membuat kesalahan serupa.\n'
                f'Jelaskan secara menyeluruh dan kontrastif, HANYA dalam Bahasa Indonesia: aturan '
                f'umum "{correct}" vs "{wrong}", kenapa "{wrong}" salah di kasus ini secara '
                f'spesifik, dan satu contoh kalimat lain yang mengontraskan keduanya.\nJawaban:'
            )
        else:
            prompt = (
                f'{_EN_DETAILED_EXAMPLE}'
                f'Now a new item:\n'
                f'Korean grammar item: "{stem}" The correct answer is "{correct}".\n'
                f'A learner picked "{wrong}" instead, which is wrong, and has made this same type '
                f'of mistake before.\n'
                f'Give a thorough, contrastive explanation in English: the general rule distinguishing '
                f'"{correct}" vs "{wrong}", why "{wrong}" fails here specifically, and one more example '
                f'sentence contrasting both uses.\nAnswer ONLY in English.'
            )
        full = self._generate(prompt, enable_thinking=True, max_new_tokens=_DETAILED_MIN_TOKENS)
        return correct_korean_conjugation(full.split("</think>")[-1].strip())

    def _generate(self, prompt: str, enable_thinking: bool, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                            enable_thinking=enable_thinking,
                                            return_tensors="pt", return_dict=True).to("cuda")
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _hangul_char_count(text: str) -> int:
    return sum(1 for ch in text if "가" <= ch <= "힣")


if __name__ == "__main__":
    provider = QwenFeedbackProvider()
    stem, correct, wrong = "저는 카페___ 마셔요.", "에서", "에"

    # Explicit language-compliance check added 2026-10-20 (PRD §9.64), on top
    # of the pre-existing topic-leak check below -- the two are different
    # failure modes (leaking the example's CONTENT vs. ignoring the
    # requested LANGUAGE entirely) and the leak check alone does not catch
    # a response that's correctly on-topic but wholly in Korean, which is
    # the actual bug a live user hit after the EXAONE-3.5-7.8B swap.
    # Thresholds calibrated 2026-10-20 from a real run (PRD §9.65), not
    # guessed: a genuinely compliant, high-quality English detailed_feedback
    # answer legitimately hit 89 Hangul chars (4 embedded Korean example
    # sentences + repeated quoted-particle contrasts, exactly what "one more
    # example sentence contrasting both uses" asks for) -- the original
    # guess of 40 was simply too strict and flagged a correct answer as a
    # false positive. Set with real headroom above the observed 89, not
    # padded arbitrarily; still low enough to catch a response that's
    # actually substantially Korean prose (which would run into the
    # hundreds of characters for a multi-sentence explanation).
    _FAST_HANGUL_LIMIT = 15
    _DETAILED_HANGUL_LIMIT = 150

    for language in ("en", "id"):
        fast = provider.fast_feedback(stem, correct, wrong, language=language)
        print(f"--- fast [{language}] ---\n{fast}\n")
        assert fast, "fast feedback must not be empty"
        assert "에서" in fast or "에" in fast, "explanation should reference the actual particles"
        assert "학교" not in fast and "공부" not in fast, (
            "must not leak the few-shot example's topic (school/studying) into an unrelated answer")
        assert _hangul_char_count(fast) <= _FAST_HANGUL_LIMIT, (
            f"fast_feedback[{language}] looks like it answered substantially in Korean "
            f"instead of the requested language ({_hangul_char_count(fast)} Hangul chars): {fast!r}")

        detailed = provider.detailed_feedback(stem, correct, wrong, language=language)
        print(f"--- detailed [{language}] ---\n{detailed}\n")
        assert detailed, "detailed feedback must not be empty -- _DETAILED_MIN_TOKENS set too low again"
        assert "<think>" not in detailed, "the reasoning scratchpad must be stripped before showing the learner"
        assert _hangul_char_count(detailed) <= _DETAILED_HANGUL_LIMIT, (
            f"detailed_feedback[{language}] looks like it answered substantially in Korean "
            f"instead of the requested language ({_hangul_char_count(detailed)} Hangul chars): {detailed!r}")

    print("all smoke tests passed")
