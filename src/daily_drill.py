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
from urnings_tracker import LearnerSkillProfile
from vocab_scheduler import due_vocab_cards


class ContentProvider(Protocol):
    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        """A vocab item_id the user hasn't seen yet, or None if none left."""

    def next_new_grammar_item(self, skill_id: str) -> str | None:
        """A freshly generated/selected grammar item_id targeting skill_id."""


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
    attempts = 0
    while len(grammar_items) < n_grammar and due_skills and attempts < n_grammar * 10:
        skill = due_skills[attempts % len(due_skills)]
        item = provider.next_new_grammar_item(skill)
        if item is not None:
            grammar_items.append(item)
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

        def next_new_vocab_item(self, user_id, exclude):
            self._vocab_counter += 1
            return f"vocab-new-{self._vocab_counter}"

        def next_new_grammar_item(self, skill_id):
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

    drill = assemble_daily_drill(state, MockProvider(), DailyDrillConfig(n_questions=10, vocab_ratio=0.4), now=today)
    assert len(drill) == 10
    assert "vocab-old-1" in drill, "the due vocab card must be included before any new vocab item"
    assert all("honorifics" not in item for item in drill), "freshly-mastered skill must not resurface before its recheck cooldown"
    assert sum(1 for item in drill if item.startswith("grammar-")) == 6

    # 30+ days later, the mastered skill must resurface for a recheck even
    # though it's still "mastered" by k/n -- this is the bug fixed in
    # grammar_scheduler.py: mastery must not be a permanent, never-revisited label.
    later = today + timedelta(days=31)
    drill2 = assemble_daily_drill(state, MockProvider(), DailyDrillConfig(n_questions=10, vocab_ratio=0.4), now=later)
    assert any("honorifics" in item for item in drill2), "mastered skill overdue for recheck must resurface, not be dropped forever"

    print("all smoke tests passed")
