"""
Answer-processing use case (PRD §2.1/§2.2/§3.3): the other half of the
daily drill lifecycle. daily_drill.py assembles what to show the learner;
this module processes what the learner answers back -- updating the
tracker, deciding escalation, generating feedback, updating the vocab
scheduler, and recording a streak/recheck outcome. Nothing here existed
before this was pointed out as a gap: without it, item.answer was compared
to nothing and no state ever changed.

Deliberately application-logic only, no persistence/API/serving layer --
those are separate, deferred decisions. State objects (UserDrillState,
StreakState, grammar/vocab item stores) are passed in and returned/mutated
the same way the rest of this codebase works; a persistence layer would
load state before calling these functions and save it after, not change
their signatures.

Item-level Urnings calibration is intentionally NOT persisted here: items
from exaone_provider are uniquely generated per occasion, not reused
across users or even across occasions for the same user, so there is no
repeated exposure for an item urn to accumulate evidence over -- unlike
the LEARNER's skill urn, which does accumulate across every item of that
skill. A neutral, throwaway item urn is used only so update()'s symmetric
ball-exchange math has two sides to operate on; only the learner side is
kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from daily_drill import UserDrillState
from exaone_provider import GeneratedItem
from qwen_feedback import QwenFeedbackProvider
from streak import StreakState
from urnings_tracker import UrnState, is_mastered, should_escalate
from urnings_tracker import update as urnings_update
from vocab_scheduler import new_vocab_card
from vocab_scheduler import review as vocab_review

_NEUTRAL_ITEM_URN = UrnState(k=16, n=32)


@dataclass
class AnswerResult:
    correct: bool
    feedback: str | None = None
    newly_mastered: bool = False


def submit_grammar_answer(state: UserDrillState, item: GeneratedItem, chosen_option: str,
                           feedback_provider: QwenFeedbackProvider,
                           rng: np.random.Generator, now: datetime | None = None) -> AnswerResult:
    correct = chosen_option == item.answer
    learner_urn = state.skill_profile.get(item.skill_id)
    was_mastered = is_mastered(learner_urn)

    new_learner_urn, _ = urnings_update(learner_urn, _NEUTRAL_ITEM_URN, correct, rng)
    state.skill_profile.set(item.skill_id, new_learner_urn)

    feedback = None
    if not correct:
        correct_text = item.options[item.answer]
        wrong_text = item.options[chosen_option]
        if should_escalate(new_learner_urn):
            feedback = feedback_provider.detailed_feedback(item.stem, correct_text, wrong_text)
        else:
            feedback = feedback_provider.fast_feedback(item.stem, correct_text, wrong_text)

    today = (now or datetime.now()).date()
    state.practice_log.record(item.skill_id, today)
    newly_mastered = (not was_mastered) and is_mastered(new_learner_urn)

    return AnswerResult(correct=correct, feedback=feedback, newly_mastered=newly_mastered)


def submit_vocab_answer(state: UserDrillState, word_id: str, correct: bool,
                         now: datetime | None = None) -> AnswerResult:
    card = state.vocab_cards.get(word_id) or new_vocab_card(hash(word_id) & 0xFFFFFFFF)
    state.vocab_cards[word_id], _ = vocab_review(card, correct, now=now)
    return AnswerResult(correct=correct)


def complete_drill_session(streak: StreakState, today) -> StreakState:
    """Call once per day, after the day's drill is finished -- not per answer."""
    return streak.record_completion(today)


if __name__ == "__main__":
    from datetime import timezone

    from exaone_provider import ExaoneGrammarProvider
    from qwen_feedback import QwenFeedbackProvider
    from urnings_tracker import UrnState

    rng = np.random.default_rng(0)
    now = datetime.now(timezone.utc)

    state = UserDrillState(user_id="u1", skill_ids=["particle_loc"])
    state.skill_profile.set("particle_loc", UrnState(k=10, n=32))

    provider = ExaoneGrammarProvider(seed=0)
    item_id = provider.next_new_grammar_item("particle_loc")
    assert item_id is not None
    item = provider.items[item_id]

    feedback_provider = QwenFeedbackProvider()

    wrong_option = next(oid for oid, text in item.options.items() if oid != item.answer)
    result = submit_grammar_answer(state, item, wrong_option, feedback_provider, rng, now=now)
    assert not result.correct
    assert result.feedback, "a wrong answer must produce feedback text"
    assert item.options[wrong_option] in result.feedback, (
        "feedback must reference the actual wrong particle text, not its option label (A/B/C/D)")
    print("wrong-answer feedback:", result.feedback)

    urn_after_wrong = state.skill_profile.get("particle_loc")
    assert urn_after_wrong.k <= 10, "a wrong answer must not increase k"

    result2 = submit_grammar_answer(state, item, item.answer, feedback_provider, rng, now=now)
    assert result2.correct
    assert result2.feedback is None, "a correct answer must not produce feedback"

    vocab_state = UserDrillState(user_id="u1", skill_ids=[])
    vr = submit_vocab_answer(vocab_state, "word-1", correct=True, now=now)
    assert vr.correct
    assert "word-1" in vocab_state.vocab_cards

    streak = StreakState()
    streak = complete_drill_session(streak, now.date())
    assert streak.count == 1

    print("all smoke tests passed")
