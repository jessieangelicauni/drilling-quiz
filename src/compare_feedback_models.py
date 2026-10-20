"""
Head-to-head comparison: EXAONE-3.5-7.8B (current qwen_feedback.py config)
vs Qwen/Qwen3-4B (the originally-validated model) on the SAME test items,
same prompts, same generation settings -- to answer the open question in
PRD §9.65: EXAONE is confirmed to WORK (language-compliant, looks
high-quality) but was never formally compared against Qwen3 to back a
claim that it's actually BETTER.

Deliberately does NOT reuse QwenFeedbackProvider's __init__ for the Qwen3
side: that constructor now hardcodes trust_remote_code=True and 4-bit
quantization (added for EXAONE, PRD §9.53/§9.55), which would quantize
Qwen3-4B too -- a different, never-validated configuration for it (Qwen3
was only ever validated in plain bf16, ~8GB VRAM, see qwen_feedback.py's
own module docstring). Loads each model with its OWN correct, previously-
validated configuration, and reuses QwenFeedbackProvider's existing
fast_feedback/detailed_feedback/_generate methods via object.__new__ (skips
__init__, manually sets .tok/.model) so the exact same prompt text is used
for both -- no duplicated prompt logic to drift out of sync.

This produces a reading pack, not an automatic verdict: feedback-text
quality is free-form prose, which this project has no automated grader
for (unlike the other components' solver-check/log-likelihood scoring).
Print both models' output side by side per test case and judge by reading,
the same way instructor review is used elsewhere in this project for
content a deterministic check can't score.

Could not be run in the cloud sandbox this was written in (no GPU). Run
on a machine with both models' weights reachable (EXAONE-3.5-7.8B needs
src/patch_exaone35_remote_code.py applied after first load, same as
elsewhere in this project).

Usage: python3 src/compare_feedback_models.py
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qwen_feedback import QwenFeedbackProvider

_EXAONE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
_QWEN_MODEL = "Qwen/Qwen3-4B"

# A handful of test cases spanning different skills/particles, not just the
# one case qwen_feedback.py's own smoke test uses -- a comparison on a
# single item is too thin to support a "better" claim either way.
_TEST_CASES = [
    ("저는 카페___ 마셔요.", "에서", "에"),
    ("할머니께서 ___.", "쉬세요", "쉬어요"),
    ("저는 어제 밥을 ___.", "먹었어요", "먹어요"),
    ("이것은 제가 ___ 책이에요.", "읽은", "읽는"),
]


def _load_exaone() -> QwenFeedbackProvider:
    provider = object.__new__(QwenFeedbackProvider)
    provider.tok = AutoTokenizer.from_pretrained(_EXAONE_MODEL, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    provider.model = AutoModelForCausalLM.from_pretrained(
        _EXAONE_MODEL, quantization_config=bnb_config, trust_remote_code=True, device_map="cuda")
    return provider


def _load_qwen3() -> QwenFeedbackProvider:
    provider = object.__new__(QwenFeedbackProvider)
    provider.tok = AutoTokenizer.from_pretrained(_QWEN_MODEL)
    provider.model = AutoModelForCausalLM.from_pretrained(
        _QWEN_MODEL, dtype=torch.bfloat16).to("cuda")
    return provider


def main() -> None:
    print("Loading EXAONE-3.5-7.8B (current production config)...")
    exaone = _load_exaone()
    print("Loading Qwen3-4B (originally-validated config)...")
    qwen = _load_qwen3()

    for stem, correct, wrong in _TEST_CASES:
        print(f"\n{'=' * 70}\nITEM: {stem}  correct={correct!r}  wrong={wrong!r}\n{'=' * 70}")
        for language in ("en", "id"):
            print(f"\n--- fast_feedback [{language}] ---")
            print(f"EXAONE: {exaone.fast_feedback(stem, correct, wrong, language=language)}")
            print(f"Qwen3:  {qwen.fast_feedback(stem, correct, wrong, language=language)}")

            print(f"\n--- detailed_feedback [{language}] ---")
            print(f"EXAONE: {exaone.detailed_feedback(stem, correct, wrong, language=language)}")
            print(f"Qwen3:  {qwen.detailed_feedback(stem, correct, wrong, language=language)}")

    print(f"\n{'=' * 70}")
    print("Done. This is a reading pack, not an automatic verdict -- read both "
          "columns per item and judge: factual correctness, clarity, whether the "
          "explanation would actually help a learner, and whether either model "
          "produced a wrong/misleading explanation (the one failure mode that "
          "would matter most). Report back which (if either) reads better, and "
          "whether there's a clear pattern across languages/items or it's mixed.")


if __name__ == "__main__":
    main()
