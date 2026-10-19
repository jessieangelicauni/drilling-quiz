"""
Real ContentProvider for grammar items (PRD §2.3/§3.1): wires
EXAONE-4.0-1.2B generation to the KAGAS distractor validator
(distractor_validator.py), implementing daily_drill.ContentProvider.

Built from direct experimentation (2026-10-19, see docs/PRD.md), through
two rounds of fixes, not from the vendor's documented usage pattern:

Round 1 -- getting the model to produce anything usable at all:
  - Zero-shot JSON-structured requests produced syntactically valid JSON
    with semantically wrong content (meta-commentary about the task
    instead of an actual item), 3/3 trials, even with a one-shot example.
  - An open-ended non-reasoning request ("write a sentence using X")
    failed near-instantly (echoed one token, stopped).
  - Fix: a narrowly SCAFFOLDED prompt (explicit template + one worked
    example + a single slot to fill) made non-reasoning mode work
    reliably (5/5 across five topics).

Round 2 -- the model still got things wrong that no prompt fix could
reliably prevent, because they're not language-modelling tasks:
  - Wrong 을/를 allomorph for the noun's batchim ("라면를", should be
    "라면을") -- caught, but only after adding a dedicated deterministic
    check (distractor_validator.expected_object_particle); a tag-only
    check couldn't see it.
  - Wrong verb conjugation ("마시요" instead of "마셔요", "보요" instead of
    "봐요") -- NOT caught by kiwipiepy's analyzer, which is deliberately
    permissive (built for noisy/informal text) and happily re-parses a
    malformed surface form as if it were well-formed. A round-trip
    analyze-then-join check is therefore not a conjugation validator.

The real fix for round 2 was to stop asking the LLM to produce the parts
that have a deterministic answer at all. The LLM is now asked only for a
dictionary-form verb lemma (a vocabulary-selection task, still genuinely
open-ended -- this is where the "infinite item pool" generative value
actually lives). Everything else is built by code: the particle's
correctness (fixed by skill for particle_loc; by Hangul batchim for
particle_obj) and the verb's conjugation (kiwi.join(), which applies
Korean's irregular contractions correctly by construction, e.g.
마시다->마셔요, 보다->봐요, 하다->해요). The LLM never writes a full sentence
or marks a blank; it can no longer get conjugation or particle correctness
wrong because it's no longer asked to produce either.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from kiwipiepy import Kiwi
from transformers import AutoModelForCausalLM, AutoTokenizer

from distractor_validator import expected_object_particle, validate_cloze_option

_MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"
_KIWI = Kiwi()
_FILLER_PARTICLES = ("도", "만", "가", "는")

_VERB_PROMPT_LOC = (
    '"{slot}"에서 보통 하는 자연스러운 한국어 동사(기본형, "다"로 끝나는 형태) 딱 하나만 써줘.\n'
    '예시: 입력 "학교" -> 출력 "공부하다"\n'
    "다른 설명 없이 동사 기본형만 출력해."
)
_VERB_PROMPT_OBJ = (
    '"{slot}"와 자연스럽게 어울리는 한국어 동사(기본형, "다"로 끝나는 형태) 딱 하나만 써줘.\n'
    '예시: 입력 "밥" -> 출력 "먹다"\n'
    "다른 설명 없이 동사 기본형만 출력해."
)

SKILL_SPECS = {
    "particle_loc": {
        "tag": "JKB",
        # Fixed by construction: "저는 [place] ___ [activity verb]." is an
        # activity-at-a-location frame, which always wants 에서 -- this
        # isn't detected from LLM output, it's true by the template itself.
        "correct_form": "에서",
        "distractor_form": "에",
        "distractor_tag": "JKB",
        "slots": ("도서관", "회사", "공원", "카페", "병원", "학교", "식당"),
        "verb_prompt": _VERB_PROMPT_LOC,
    },
    "particle_obj": {
        "tag": "JKO",
        # Correct form is computed per-noun (batchim rule), not fixed.
        "correct_form": None,
        "distractor_form": "에",
        "distractor_tag": "JKB",
        "slots": ("사과", "책", "커피", "영화", "빵", "라면"),
        "verb_prompt": _VERB_PROMPT_OBJ,
    },
}


@dataclass
class GeneratedItem:
    skill_id: str
    stem: str
    blank_marker: str
    options: dict[str, str]
    answer: str


class ExaoneGrammarProvider:
    def __init__(self, model_name: str = _MODEL_NAME, seed: int | None = None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
        self.rng = random.Random(seed)
        self._counter = 0
        self.items: dict[str, GeneratedItem] = {}

    def _pick_verb_lemma(self, prompt_template: str, slot: str) -> str | None:
        prompt = prompt_template.format(slot=slot)
        messages = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True).to("cuda")
        out = self.model.generate(**enc, max_new_tokens=15, do_sample=False)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        verb = text.split()[0].strip('".,!?') if text else ""
        if verb.endswith("다") and len(verb) >= 2:
            return verb
        return None  # not a usable dictionary-form verb; caller retries/skips

    def next_new_grammar_item(self, skill_id: str) -> str | None:
        spec = SKILL_SPECS.get(skill_id)
        if spec is None:
            return None

        slot = self.rng.choice(spec["slots"])
        verb_lemma = self._pick_verb_lemma(spec["verb_prompt"], slot)
        if verb_lemma is None:
            return None

        try:
            conjugated = _KIWI.join([(verb_lemma[:-1], "VV"), ("어요", "EF")])
        except Exception:
            return None  # kiwipiepy doesn't recognise this as a real verb stem

        correct_form = spec["correct_form"] or expected_object_particle(slot)
        distractor_form, distractor_tag = spec["distractor_form"], spec["distractor_tag"]
        stem = f"저는 {slot}___ {conjugated}."

        if not validate_cloze_option(stem, "___", correct_form, spec["tag"]).valid:
            return None
        if not validate_cloze_option(stem, "___", distractor_form, distractor_tag).valid:
            return None

        fillers = []
        for candidate in _FILLER_PARTICLES:
            if len(fillers) == 2:
                break
            if candidate in (correct_form, distractor_form):
                continue
            if not validate_cloze_option(stem, "___", candidate, spec["tag"]).valid:
                fillers.append(candidate)

        option_texts = [correct_form, distractor_form, *fillers]
        labels = ["A", "B", "C", "D"][:len(option_texts)]
        self.rng.shuffle(labels)
        answer_label = labels[0]
        options = dict(zip(labels, option_texts))

        item_id = f"grammar-{skill_id}-{self._counter}"
        self._counter += 1
        self.items[item_id] = GeneratedItem(skill_id, stem, "___", options, answer_label)
        return item_id

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        raise NotImplementedError("vocab items come from the admin-maintained curriculum, not generation")


if __name__ == "__main__":
    provider = ExaoneGrammarProvider(seed=0)

    for skill_id in ("particle_loc", "particle_obj"):
        successes = 0
        for _ in range(5):
            item_id = provider.next_new_grammar_item(skill_id)
            if item_id is not None:
                successes += 1
                item = provider.items[item_id]
                print(item)
                assert item.blank_marker in item.stem
                assert item.answer in item.options
                assert len(item.options) in (2, 3, 4)
        print(f"{skill_id}: {successes}/5 generations succeeded")
        assert successes >= 4, f"{skill_id}: expected the validated pipeline to succeed reliably, got {successes}/5"

    assert provider.next_new_grammar_item("unknown_skill") is None

    print("all smoke tests passed")
