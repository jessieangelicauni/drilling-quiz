from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .daily_drill import UserDrillState
from ..urnings.difficulty_heuristic import estimate_difficulty
from ..generation.item_foundry import GeneratedItem
from extras.streak import StreakState
from ..urnings.urnings_tracker import UrnState, is_mastered, seed_item_from_encoder
from ..urnings.urnings_tracker import update as urnings_update
from extras.vocab_scheduler import new_vocab_card
from extras.vocab_scheduler import review as vocab_review


@dataclass
class AnswerResult:
    correct: bool
    newly_mastered: bool = False
    updated_item_urn: UrnState | None = None


def submit_grammar_answer(state: UserDrillState, item: GeneratedItem, chosen_option: str,
                           rng: np.random.Generator, now: datetime | None = None,
                           item_urn: UrnState | None = None,
                           selection_prob=None) -> AnswerResult:
    correct = chosen_option == item.answer
    learner_urn = state.skill_profile.get(item.skill_id)
    was_mastered = is_mastered(learner_urn)

    if item_urn is None:
        item_urn = seed_item_from_encoder(*estimate_difficulty(item.stem))

    new_learner_urn, new_item_urn = urnings_update(learner_urn, item_urn, correct, rng,
                                                   selection_prob=selection_prob)
    state.skill_profile.set(item.skill_id, new_learner_urn)
    state.skill_profile.record_outcome(item.skill_id, correct)

    today = (now or datetime.now()).date()
    state.practice_log.record(item.skill_id, today)
    newly_mastered = (not was_mastered) and is_mastered(new_learner_urn)

    return AnswerResult(correct=correct, newly_mastered=newly_mastered, updated_item_urn=new_item_urn)


def submit_vocab_answer(state: UserDrillState, word_id: str, correct: bool,
                         now: datetime | None = None) -> AnswerResult:
    card = state.vocab_cards.get(word_id) or new_vocab_card(hash(word_id) & 0xFFFFFFFF)
    state.vocab_cards[word_id], _ = vocab_review(card, correct, now=now)
    return AnswerResult(correct=correct)


def complete_drill_session(streak: StreakState, today) -> StreakState:
    return streak.record_completion(today)


if __name__ == "__main__":
    from datetime import timezone

    from ..urnings.urnings_tracker import UrnState

    rng = np.random.default_rng(0)
    now = datetime.now(timezone.utc)

    state = UserDrillState(user_id="u1", skill_ids=["particle_loc"])
    state.skill_profile.set("particle_loc", UrnState(k=10, n=32))
    item = GeneratedItem("particle_loc", "동생이 ___ 축구를 해요.", "___",
                         {"A": "공원에", "B": "공원에서", "C": "공원을", "D": "공원이"}, "B",
                         reasoning={"A": "'공원에' is incorrect here.", "B": "'공원에서' is correct here.",
                                    "C": "'공원을' is incorrect here.", "D": "'공원이' is incorrect here."})

    result = submit_grammar_answer(state, item, "A", rng, now=now)
    assert not result.correct
    urn_after_wrong = state.skill_profile.get("particle_loc")
    assert urn_after_wrong.k <= 10

    assert state.skill_profile.consecutive_wrong("particle_loc") == 1
    submit_grammar_answer(state, item, "A", rng, now=now)
    assert state.skill_profile.consecutive_wrong("particle_loc") == 2

    result2 = submit_grammar_answer(state, item, item.answer, rng, now=now)
    assert result2.correct
    assert state.skill_profile.consecutive_wrong("particle_loc") == 0

    assert result.updated_item_urn is not None
    assert result2.updated_item_urn is not None

    persisted_item_urn = seed_item_from_encoder(mu_hat=0.0, sigma_hat2=1.5)
    result3 = submit_grammar_answer(state, item, item.answer, rng, now=now,
                                     item_urn=persisted_item_urn)
    assert result3.updated_item_urn.n == persisted_item_urn.n
    assert 0 <= result3.updated_item_urn.k <= result3.updated_item_urn.n

    from ..urnings.urnings_tracker import UrnState as _UrnState
    from ..urnings.urnings_tracker import update as _urnings_update
    hard_item_urn = seed_item_from_encoder(mu_hat=2.5, sigma_hat2=1.5)
    easy_item_urn = seed_item_from_encoder(mu_hat=-2.5, sigma_hat2=1.5)
    start = _UrnState(k=16, n=32)
    trial_rng = np.random.default_rng(7)
    n_trials = 3000
    hard_ks = [_urnings_update(start, hard_item_urn, correct=True, rng=trial_rng)[0].k
               for _ in range(n_trials)]
    easy_ks = [_urnings_update(start, easy_item_urn, correct=True, rng=trial_rng)[0].k
               for _ in range(n_trials)]
    avg_hard, avg_easy = np.mean(hard_ks), np.mean(easy_ks)
    assert avg_hard > avg_easy, (
        f"avg k after correct on harder-seeded item ({avg_hard:.3f}) must exceed "
        f"easier-seeded item ({avg_easy:.3f})"
    )

    vocab_state = UserDrillState(user_id="u1", skill_ids=[])
    vr = submit_vocab_answer(vocab_state, "word-1", correct=True, now=now)
    assert vr.correct
    assert "word-1" in vocab_state.vocab_cards

    streak = StreakState()
    streak = complete_drill_session(streak, now.date())
    assert streak.count == 1

    print("all smoke tests passed")
