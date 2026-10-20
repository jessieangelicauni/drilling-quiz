"""
Combines VocabContentProvider and ExaoneGrammarProvider into one object
satisfying daily_drill.ContentProvider's full protocol (both methods) --
assemble_daily_drill() calls both next_new_vocab_item and
next_new_grammar_item on the SAME provider argument, so neither half can
be passed alone once vocab_ratio is strictly between 0 and 1.
"""
from __future__ import annotations

from collections import ChainMap

from exaone_provider import ExaoneGrammarProvider
from vocab_provider import VocabContentProvider


class DispatchingGrammarProvider:
    """Routes next_new_grammar_item by skill_id prefix to one of several
    underlying grammar providers (PRD §9.49) -- added so curriculum-
    generated skills (skill_id="curriculum_<number>",
    curriculum_grammar_provider.py) and the 11 hand-coded skills
    (exaone_provider.py) can be scheduled through the SAME
    ContentProvider without either provider needing to know the other
    exists. daily_drill.py/answer_processing.py only ever call
    next_new_grammar_item and read back provider.items[item_id] -- both
    duck-typed, so no change was needed to either of those modules.
    """
    def __init__(self, exaone: ExaoneGrammarProvider, curriculum):
        self.exaone = exaone
        self.curriculum = curriculum

    @property
    def items(self):
        # A read-only merged view, not a copy -- items keep accumulating
        # in each sub-provider's own dict as the drill runs, and this
        # property must see that without re-merging by hand each time.
        return ChainMap(self.curriculum.items, self.exaone.items)

    def next_new_grammar_item(self, skill_id: str, learner_urn=None,
                               exclude: set[str] = frozenset()) -> str | None:
        provider = self.curriculum if skill_id.startswith("curriculum_") else self.exaone
        return provider.next_new_grammar_item(skill_id, learner_urn=learner_urn, exclude=exclude)


class CombinedContentProvider:
    def __init__(self, vocab: VocabContentProvider, grammar):
        self.vocab = vocab
        self.grammar = grammar

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        return self.vocab.next_new_vocab_item(user_id, exclude)

    def next_new_grammar_item(self, skill_id: str, learner_urn=None,
                               exclude: set[str] = frozenset()) -> str | None:
        return self.grammar.next_new_grammar_item(skill_id, learner_urn=learner_urn, exclude=exclude)


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

    # End-to-end mixed-source smoke test (PRD §9.49): a drill whose
    # skill_ids span BOTH a hand-coded skill and a curriculum-generated
    # one, through the SAME DispatchingGrammarProvider -- proves the
    # dispatch-by-prefix wiring and the shared ChainMap `.items` view
    # actually work together, not just each half in isolation.
    import numpy as np

    from answer_processing import submit_grammar_answer
    from curriculum_grammar_provider import CurriculumGrammarProvider
    from urnings_tracker import UrnState

    class _FakeFeedbackProvider:
        """Stand-in for QwenFeedbackProvider -- this smoke test already
        loads EXAONE + Qwen3-4B (via CurriculumGrammarProvider) at once;
        a 3rd, separate Qwen3-4B instance for feedback alone OOMs the
        15GB GPU this was run on (found empirically, PRD §9.49).
        QwenFeedbackProvider's OWN interface is already exercised by
        answer_processing.py's own __main__ -- what THIS test needs to
        prove is that submit_grammar_answer accepts a CurriculumItem via
        duck typing and updates skill_profile correctly, not that
        feedback generation itself works (unrelated to this change).
        """
        def fast_feedback(self, stem, correct, wrong, language="en"):
            return f"fake fast feedback: {correct} vs {wrong}"

        def detailed_feedback(self, stem, correct, wrong, language="en"):
            return f"fake detailed feedback: {correct} vs {wrong}"

    mixed_provider = DispatchingGrammarProvider(
        exaone=ExaoneGrammarProvider(seed=0),
        curriculum=CurriculumGrammarProvider(seed=0),
    )
    mixed_state = UserDrillState(user_id="u2", skill_ids=["particle_loc", "curriculum_40"])
    mixed_state.skill_profile.set("particle_loc", UrnState(k=10, n=32))
    mixed_state.skill_profile.set("curriculum_40", UrnState(k=10, n=32))

    exaone_item_id = mixed_provider.next_new_grammar_item("particle_loc")
    assert exaone_item_id is not None
    # "curriculum_40" (-았/었/했-) is one of the entries confirmed reliable
    # at §9.49's final calibration -- used here so this smoke test's
    # result doesn't depend on the ~32% miss rate the rest of the 524
    # points still have.
    curriculum_item_id = mixed_provider.next_new_grammar_item("curriculum_40")
    assert curriculum_item_id is not None, "entry 40 is a known-reliable point, see PRD §9.49"

    assert exaone_item_id in mixed_provider.items
    assert curriculum_item_id in mixed_provider.items
    assert mixed_provider.items[curriculum_item_id].skill_id == "curriculum_40"

    feedback_provider = _FakeFeedbackProvider()
    rng = np.random.default_rng(0)
    curriculum_item = mixed_provider.items[curriculum_item_id]
    wrong_option = next(oid for oid in curriculum_item.options if oid != curriculum_item.answer)
    result = submit_grammar_answer(mixed_state, curriculum_item, wrong_option,
                                    feedback_provider, rng)
    assert not result.correct
    assert result.feedback, "a wrong answer on a curriculum item must still produce feedback"
    assert mixed_state.skill_profile.get("curriculum_40").k <= 10, (
        "a wrong answer must not increase k, same as any other skill"
    )
    print(f"\nmixed-source drill OK: exaone item {exaone_item_id!r} + "
          f"curriculum item {curriculum_item_id!r} both served and answerable "
          f"through the same DispatchingGrammarProvider")

    print("all smoke tests passed")
