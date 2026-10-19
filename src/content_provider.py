"""
Combines VocabContentProvider and ExaoneGrammarProvider into one object
satisfying daily_drill.ContentProvider's full protocol (both methods) --
assemble_daily_drill() calls both next_new_vocab_item and
next_new_grammar_item on the SAME provider argument, so neither half can
be passed alone once vocab_ratio is strictly between 0 and 1.
"""
from __future__ import annotations

from exaone_provider import ExaoneGrammarProvider
from vocab_provider import VocabContentProvider


class CombinedContentProvider:
    def __init__(self, vocab: VocabContentProvider, grammar: ExaoneGrammarProvider):
        self.vocab = vocab
        self.grammar = grammar

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        return self.vocab.next_new_vocab_item(user_id, exclude)

    def next_new_grammar_item(self, skill_id: str) -> str | None:
        return self.grammar.next_new_grammar_item(skill_id)


if __name__ == "__main__":
    from daily_drill import DailyDrillConfig, UserDrillState, assemble_daily_drill
    from vocab_provider import VocabCurriculum

    curriculum = VocabCurriculum()
    curriculum.add("w1", "도서관", "library", topic="places")
    curriculum.add("w2", "병원", "hospital", topic="places")
    curriculum.add("w3", "공원", "park", topic="places")
    curriculum.add("w4", "사과", "apple", topic="food")

    provider = CombinedContentProvider(
        vocab=VocabContentProvider(curriculum, n_options=3, seed=0),
        grammar=ExaoneGrammarProvider(seed=0),
    )

    state = UserDrillState(user_id="u1", skill_ids=["particle_loc"])
    drill = assemble_daily_drill(state, provider, DailyDrillConfig(n_questions=4, vocab_ratio=0.5))
    assert len(drill) == 4

    vocab_ids = [i for i in drill if i in provider.vocab.questions]
    grammar_ids = [i for i in drill if i in provider.grammar.items]
    assert len(vocab_ids) == 2, f"expected 2 vocab items, got {vocab_ids}"
    assert len(grammar_ids) == 2, f"expected 2 grammar items, got {grammar_ids}"

    print("drill:", drill)
    print("all smoke tests passed")
