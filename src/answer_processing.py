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

Item-level Urnings calibration (Novelty 1, PRD §3.1) is now wired in for
real (§9.12) -- this module stayed persistence-agnostic (per its own
original design) through the change: it ACCEPTS an item_urn and RETURNS
the updated one, same pattern as every other state object here (passed
in, mutated/returned, never stored by this module itself). Two cases:
  - Caller has a persisted item_urn (e.g. from a QuestionBank.BankEntry --
    items served from the bank genuinely ARE reused across many learners
    once approved, so their calibration can accumulate real evidence over
    time; the §9.2-era assumption that items are single-use predates the
    bank existing and no longer holds for entries that live there) --
    passed in, and the caller is responsible for writing the returned
    AnswerResult.updated_item_urn back to that entry.
  - Caller has no persistence target (an ephemeral, live-generated item
    with nowhere to store calibration across exposures) -- item_urn is
    omitted, and a one-shot prior is seeded fresh from
    difficulty_heuristic.estimate_difficulty() for just this event. Still
    real Novelty-1 behaviour (a harder-estimated item moves the learner's
    urn differently than an easier one via update()'s ball-exchange math),
    just not accumulated across exposures.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from daily_drill import UserDrillState
from difficulty_heuristic import estimate_difficulty
from exaone_provider import GeneratedItem
from qwen_feedback import QwenFeedbackProvider
from streak import StreakState
from urnings_tracker import UrnState, is_mastered, seed_item_from_encoder, should_escalate_full
from urnings_tracker import update as urnings_update
from vocab_scheduler import new_vocab_card
from vocab_scheduler import review as vocab_review


@dataclass
class AnswerResult:
    correct: bool
    feedback: str | None = None
    newly_mastered: bool = False
    updated_item_urn: UrnState | None = None


def submit_grammar_answer(state: UserDrillState, item: GeneratedItem, chosen_option: str,
                           feedback_provider: QwenFeedbackProvider,
                           rng: np.random.Generator, now: datetime | None = None,
                           item_urn: UrnState | None = None) -> AnswerResult:
    correct = chosen_option == item.answer
    learner_urn = state.skill_profile.get(item.skill_id)
    was_mastered = is_mastered(learner_urn)

    if item_urn is None:
        item_urn = seed_item_from_encoder(*estimate_difficulty(item.stem))

    new_learner_urn, new_item_urn = urnings_update(learner_urn, item_urn, correct, rng)
    state.skill_profile.set(item.skill_id, new_learner_urn)
    # Eq. 3's OTHER disjunct (PRD §9.38): previously the whole
    # consecutive(m_j)>=2 trigger had nothing to read from anywhere in
    # this codebase, so escalation silently only ever checked urn width.
    # Updated unconditionally (correct answers reset the streak).
    consecutive = state.skill_profile.record_outcome(item.skill_id, correct, item.kagas_code)

    feedback = None
    if not correct:
        correct_text = item.options[item.answer]
        wrong_text = item.options[chosen_option]
        if should_escalate_full(new_learner_urn, consecutive):
            feedback = feedback_provider.detailed_feedback(item.stem, correct_text, wrong_text)
        else:
            feedback = feedback_provider.fast_feedback(item.stem, correct_text, wrong_text)

    today = (now or datetime.now()).date()
    state.practice_log.record(item.skill_id, today)
    newly_mastered = (not was_mastered) and is_mastered(new_learner_urn)

    return AnswerResult(correct=correct, feedback=feedback, newly_mastered=newly_mastered,
                         updated_item_urn=new_item_urn)


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

    # Novelty 1 (§9.12): the ephemeral path (no item_urn given) must still
    # seed a real one, not silently skip it.
    assert result.updated_item_urn is not None, "even an ephemeral item must get a seeded item_urn"
    assert result2.updated_item_urn is not None

    # The persisted path: caller passes an existing item_urn (as if read
    # from a QuestionBank.BankEntry), gets back an updated one with the
    # SAME n (urn size doesn't change mid-exchange) and a valid k.
    from urnings_tracker import seed_item_from_encoder
    persisted_item_urn = seed_item_from_encoder(mu_hat=0.0, sigma_hat2=1.5)
    result3 = submit_grammar_answer(state, item, item.answer, feedback_provider, rng, now=now,
                                     item_urn=persisted_item_urn)
    assert result3.updated_item_urn.n == persisted_item_urn.n, "urn size must not change mid-exchange"
    assert 0 <= result3.updated_item_urn.k <= result3.updated_item_urn.n

    # The actual causal claim Novelty 1 makes (PRD §3.1's minimal required
    # evidence): the encoder-informed item prior must actually change
    # update()'s behaviour, not just exist unused. A SINGLE draw is too
    # noisy to assert on (both the ball-exchange's y and the MH acceptance
    # step are stochastic) -- averaged over many trials instead, the same
    # way urnings_tracker.py's own smoke test validates convergence over
    # 500 draws rather than one. Direction verified empirically against
    # simulate_trackers.run_urnings (the already-validated reference
    # implementation this module's update() mirrors exactly) before writing
    # this assertion, NOT assumed from intuition: a correct answer on an
    # item seeded as easier (higher item k/n) moves the learner's urn up
    # MORE on average than the same correct answer on an item seeded as
    # harder (lower item k/n) -- counter to a naive "beating a hard item
    # should earn more credit" intuition, but this is how the validated
    # ball-exchange algorithm actually behaves per-step; its correctness is
    # established by the RMSE/convergence results in docs/PRD.md, not by
    # single-step intuition.
    from urnings_tracker import UrnState as _UrnState
    from urnings_tracker import update as _urnings_update
    hard_item_urn = seed_item_from_encoder(mu_hat=2.5, sigma_hat2=1.5)   # low item k/n
    easy_item_urn = seed_item_from_encoder(mu_hat=-2.5, sigma_hat2=1.5)  # high item k/n
    start = _UrnState(k=16, n=32)
    trial_rng = np.random.default_rng(7)
    n_trials = 3000
    hard_ks = [_urnings_update(start, hard_item_urn, correct=True, rng=trial_rng)[0].k
               for _ in range(n_trials)]
    easy_ks = [_urnings_update(start, easy_item_urn, correct=True, rng=trial_rng)[0].k
               for _ in range(n_trials)]
    avg_hard, avg_easy = np.mean(hard_ks), np.mean(easy_ks)
    assert avg_easy > avg_hard, (
        "the encoder-seeded item prior must change update()'s average behaviour -- "
        f"avg k after correct on easier-seeded item ({avg_easy:.3f}) must exceed "
        f"harder-seeded item ({avg_hard:.3f}), confirming the prior isn't inert"
    )

    vocab_state = UserDrillState(user_id="u1", skill_ids=[])
    vr = submit_vocab_answer(vocab_state, "word-1", correct=True, now=now)
    assert vr.correct
    assert "word-1" in vocab_state.vocab_cards

    streak = StreakState()
    streak = complete_drill_session(streak, now.date())
    assert streak.count == 1

    print("all smoke tests passed")
