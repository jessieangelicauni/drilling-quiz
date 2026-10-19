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
"""
from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen3-4B"
_DETAILED_MIN_TOKENS = 1200  # empirically the minimum that reliably finishes; see qwen_feedback history

Language = Literal["en", "id"]

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
                f'Korean grammar item: "{stem}" The correct answer is "{correct}".\n'
                f'A learner picked "{wrong}" instead, which is wrong.\n'
                f'Explain in English, in 1-2 short sentences, why "{wrong}" is wrong here and why '
                f'"{correct}" is correct.\nAnswer ONLY in English.'
            )
        return self._generate(prompt, enable_thinking=False, max_new_tokens=150)

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
                f'Korean grammar item: "{stem}" The correct answer is "{correct}".\n'
                f'A learner picked "{wrong}" instead, which is wrong, and has made this same type '
                f'of mistake before.\n'
                f'Give a thorough, contrastive explanation in English: the general rule distinguishing '
                f'"{correct}" vs "{wrong}", why "{wrong}" fails here specifically, and one more example '
                f'sentence contrasting both uses.\nAnswer ONLY in English.'
            )
        full = self._generate(prompt, enable_thinking=True, max_new_tokens=_DETAILED_MIN_TOKENS)
        return full.split("</think>")[-1].strip()

    def _generate(self, prompt: str, enable_thinking: bool, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                            enable_thinking=enable_thinking,
                                            return_tensors="pt", return_dict=True).to("cuda")
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    provider = QwenFeedbackProvider()
    stem, correct, wrong = "저는 카페___ 마셔요.", "에서", "에"

    for language in ("en", "id"):
        fast = provider.fast_feedback(stem, correct, wrong, language=language)
        print(f"--- fast [{language}] ---\n{fast}\n")
        assert fast, "fast feedback must not be empty"
        assert "에서" in fast or "에" in fast, "explanation should reference the actual particles"
        assert "학교" not in fast and "공부" not in fast, (
            "must not leak the few-shot example's topic (school/studying) into an unrelated answer")

        detailed = provider.detailed_feedback(stem, correct, wrong, language=language)
        print(f"--- detailed [{language}] ---\n{detailed}\n")
        assert detailed, "detailed feedback must not be empty -- _DETAILED_MIN_TOKENS set too low again"
        assert "<think>" not in detailed, "the reasoning scratchpad must be stripped before showing the learner"

    print("all smoke tests passed")
