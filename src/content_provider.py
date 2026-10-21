"""
Combines VocabContentProvider and item_foundry.MisconceptionItemGenerator
(or a question_bank.BankContentProvider) into one object
satisfying daily_drill.ContentProvider's full protocol (both methods) --
assemble_daily_drill() calls both next_new_vocab_item and
next_new_grammar_item on the SAME provider argument, so neither half can
be passed alone once vocab_ratio is strictly between 0 and 1.
"""
from __future__ import annotations

from vocab_provider import VocabContentProvider


class CombinedContentProvider:
    def __init__(self, vocab: VocabContentProvider, grammar):
        self.vocab = vocab
        self.grammar = grammar

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        return self.vocab.next_new_vocab_item(user_id, exclude)

    def next_new_grammar_item(self, skill_id: str, learner_urn=None,
                               exclude: set[str] = frozenset(),
                               target_misconception: str | None = None) -> str | None:
        return self.grammar.next_new_grammar_item(skill_id, learner_urn=learner_urn, exclude=exclude,
                                                  target_misconception=target_misconception)


if __name__ == "__main__":
    import json

    from daily_drill import DailyDrillConfig, UserDrillState, assemble_daily_drill
    from item_foundry import MisconceptionItemGenerator, build_codebook
    from vocab_provider import VocabCurriculum

    # CPU stand-ins for EXAONE and Qwen3-4B, as in question_bank.py's smoke test.
    book = build_codebook("particle_loc", lambda m: json.dumps([
        {"name": n, "explanation": n, "correct": "저는 학교에서 공부해요.", "wrong": f"저는 학교{p} 공부해요."}
        for n, p in (("에 everywhere", "에"), ("object marking", "를"), ("subject marking", "가"))],
        ensure_ascii=False))
    reply = json.dumps({"sentence": "친구가 공원에서 축구를 해요.", "target": "에서",
                        "distractors": [{"text": "에", "misconception": 1}, {"text": "을", "misconception": 2},
                                        {"text": "이", "misconception": 3}]}, ensure_ascii=False)
    grammar = MisconceptionItemGenerator(lambda m: reply, lambda s: 0.0 if "에서" in s else -5.0,
                                         {"particle_loc": book}, seed=0)

    curriculum = VocabCurriculum()
    curriculum.add("w1", "도서관", "library", topic="places")
    curriculum.add("w2", "병원", "hospital", topic="places")
    curriculum.add("w3", "공원", "park", topic="places")
    curriculum.add("w4", "사과", "apple", topic="food")

    provider = CombinedContentProvider(
        vocab=VocabContentProvider(curriculum, n_options=3, seed=0),
        grammar=grammar,
    )

    state = UserDrillState(user_id="u1", skill_ids=["particle_loc"])
    drill = assemble_daily_drill(state, provider, DailyDrillConfig(n_questions=4, vocab_ratio=0.5))
    assert len(drill) == 4, (drill, grammar.stats)

    vocab_ids = [i for i in drill if i in provider.vocab.questions]
    grammar_ids = [i for i in drill if i in provider.grammar.items]
    assert len(vocab_ids) == 2, f"expected 2 vocab items, got {vocab_ids}"
    assert len(grammar_ids) == 2, f"expected 2 grammar items, got {grammar_ids}"

    print("drill:", drill)
    print("all smoke tests passed")
