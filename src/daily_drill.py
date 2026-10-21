"""
Daily drill assembly (PRD feature: "merges new + due items").

Deliberately two separate "due" mechanisms merged at this layer, not one
unified concept -- see vocab_scheduler.py and grammar_scheduler.py for why:
  - Vocab: due = FSRS due-date, most overdue first.
  - Grammar: due = skill not yet mastered (Urnings), weakest/most uncertain first.

Content generation/question-bank lookup (EXAONE, KLUE-RoBERTa prior
injection, KAGAS distractors) doesn't exist in this repo yet -- that's
Novelty 1/2's job. This module depends on it only through the
`ContentProvider` protocol below, so the assembly logic is fully testable
with a mock today and pluggable to the real pipeline later.

How many vocab vs. grammar items, and which ones, is decided entirely by
due-logic above (vocab_ratio + FSRS/Urnings due state) -- never random,
never user-chosen, so the backlog/mastery-recheck guarantees above actually
hold. Only the *presentation order* of the already-chosen set is shuffled,
to interleave vocab and grammar rather than showing them in two blocks
(interleaved practice retains better than blocked practice -- Rohrer &
Taylor's interleaving-effect literature).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

import fsrs
import numpy as np

from grammar_scheduler import SkillPracticeLog, skills_needing_practice
from misconception_tracker import MisconceptionProfile
from urnings_tracker import LearnerSkillProfile, UrnState
from vocab_scheduler import due_vocab_cards


class ContentProvider(Protocol):
    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        """A vocab item_id the user hasn't seen yet, or None if none left."""

    def next_new_grammar_item(self, skill_id: str, learner_urn: UrnState | None = None,
                               exclude: set[str] = frozenset(),
                               target_misconception: str | None = None) -> str | None:
        """A grammar item_id targeting skill_id. Novelty 1's "Select" step
        (PRD §9.15): when learner_urn is given, a provider that can (e.g.
        BankContentProvider) should pick an item whose difficulty actually
        matches this learner's current ability for this skill, not just
        any valid item -- the piece of Generate-Calibrate-Select that
        §9.12's item_urn wiring made possible but never used until now.
        exclude avoids repeating an item already chosen earlier in the
        same drill assembly call. target_misconception, when given, is the
        misconception the learner currently holds for this skill
        (misconception_tracker): the item should offer a distractor
        realising it, so the drill keeps confronting it and keeps measuring
        whether it is still held (Novelty 1, PRD §9.69).
        """


@dataclass
class DailyDrillConfig:
    n_questions: int = 20
    vocab_ratio: float = 0.5  # remainder goes to grammar


@dataclass
class UserDrillState:
    user_id: str
    skill_ids: list[str]
    vocab_cards: dict[str, fsrs.Card] = field(default_factory=dict)
    skill_profile: LearnerSkillProfile = None
    practice_log: SkillPracticeLog = field(default_factory=SkillPracticeLog)
    # Which misconception the learner holds, per skill, from WHICH
    # distractor they pick (Novelty 1, PRD §9.69).
    misconceptions: MisconceptionProfile = field(default_factory=MisconceptionProfile)

    def __post_init__(self):
        if self.skill_profile is None:
            self.skill_profile = LearnerSkillProfile(self.user_id)


def assemble_daily_drill(state: UserDrillState, provider: ContentProvider,
                          config: DailyDrillConfig = DailyDrillConfig(),
                          now: datetime | None = None,
                          rng: np.random.Generator | None = None) -> list[str]:
    today = (now or datetime.now()).date()
    n_vocab = round(config.n_questions * config.vocab_ratio)
    n_grammar = config.n_questions - n_vocab

    vocab_items = due_vocab_cards(state.vocab_cards, now=now)[:n_vocab]
    seen = set(state.vocab_cards.keys())
    while len(vocab_items) < n_vocab:
        item = provider.next_new_vocab_item(state.user_id, seen)
        if item is None:
            break
        vocab_items.append(item)
        seen.add(item)

    due_skills = skills_needing_practice(state.skill_profile, state.skill_ids,
                                          state.practice_log, today)
    grammar_items: list[str] = []
    grammar_seen: set[str] = set()
    attempts = 0
    while len(grammar_items) < n_grammar and due_skills and attempts < n_grammar * 10:
        skill = due_skills[attempts % len(due_skills)]
        learner_urn = state.skill_profile.get(skill)
        item = provider.next_new_grammar_item(skill, learner_urn=learner_urn, exclude=grammar_seen,
                                              target_misconception=state.misconceptions.dominant(skill))
        if item is not None:
            grammar_items.append(item)
            grammar_seen.add(item)
            state.practice_log.record(skill, today)
        attempts += 1

    combined = vocab_items + grammar_items
    (rng or np.random.default_rng()).shuffle(combined)
    return combined


if __name__ == "__main__":
    from vocab_scheduler import new_vocab_card
    from urnings_tracker import UrnState

    class MockProvider:
        def __init__(self):
            self._vocab_counter = 0
            self._grammar_counter = 0
            self.received_learner_urns = []  # records what callers actually pass in

        def next_new_vocab_item(self, user_id, exclude):
            self._vocab_counter += 1
            return f"vocab-new-{self._vocab_counter}"

        def next_new_grammar_item(self, skill_id, learner_urn=None, exclude=frozenset(),
                                  target_misconception=None):
            self.received_learner_urns.append(learner_urn)
            self.received_targets = getattr(self, "received_targets", []) + [target_misconception]
            self._grammar_counter += 1
            return f"grammar-{skill_id}-{self._grammar_counter}"

    from datetime import timedelta, timezone
    created_at = datetime.now(timezone.utc)
    today = created_at + timedelta(minutes=1)  # strictly after the vocab card's own creation timestamp
    state = UserDrillState(
        user_id="u1",
        skill_ids=["particles", "honorifics"],
        vocab_cards={"vocab-old-1": new_vocab_card(1)},  # due immediately (never reviewed)
    )
    state.skill_profile.set("particles", UrnState(k=10, n=32))     # weak -> due
    state.skill_profile.set("honorifics", UrnState(k=28, n=32))    # just mastered -> not due yet
    state.practice_log.record("honorifics", today.date())          # mastered *today*, recheck not due

    mock = MockProvider()
    drill = assemble_daily_drill(state, mock, DailyDrillConfig(n_questions=10, vocab_ratio=0.4), now=today)
    assert len(drill) == 10
    assert "vocab-old-1" in drill, "the due vocab card must be included before any new vocab item"
    assert all("honorifics" not in item for item in drill), "freshly-mastered skill must not resurface before its recheck cooldown"

    # Novelty 1's "Select" step (§9.15): the caller must actually pass the
    # learner's real per-skill urn, not None/a placeholder -- this is what
    # lets an adaptive provider (BankContentProvider) match item difficulty
    # to this specific learner.
    assert mock.received_learner_urns, "next_new_grammar_item must have been called at least once"
    assert all(urn == UrnState(k=10, n=32) for urn in mock.received_learner_urns), (
        "every grammar call for skill 'particles' must receive THIS learner's actual urn for "
        f"that skill, got {mock.received_learner_urns}"
    )
    assert sum(1 for item in drill if item.startswith("grammar-")) == 6

    # 30+ days later, the mastered skill must resurface for a recheck even
    # though it's still "mastered" by k/n -- this is the bug fixed in
    # grammar_scheduler.py: mastery must not be a permanent, never-revisited label.
    later = today + timedelta(days=31)
    drill2 = assemble_daily_drill(state, MockProvider(), DailyDrillConfig(n_questions=10, vocab_ratio=0.4), now=later)
    assert any("honorifics" in item for item in drill2), "mastered skill overdue for recheck must resurface, not be dropped forever"

    print("all smoke tests passed")
