from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

import fsrs
import numpy as np

from ..urnings.grammar_scheduler import SkillPracticeLog, skills_needing_practice
from ..urnings.urnings_tracker import LearnerSkillProfile, UrnState
from extras.vocab_scheduler import due_vocab_cards


class ContentProvider(Protocol):
    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        ...

    def next_new_grammar_item(self, skill_id: str, learner_urn: UrnState | None = None,
                               exclude: set[str] = frozenset()) -> str | None:
        ...


@dataclass
class DailyDrillConfig:
    n_questions: int = 20
    vocab_ratio: float = 0.5


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
    grammar_seen: set[str] = set()
    attempts = 0
    while len(grammar_items) < n_grammar and due_skills and attempts < n_grammar * 10:
        skill = due_skills[attempts % len(due_skills)]
        learner_urn = state.skill_profile.get(skill)
        item = provider.next_new_grammar_item(skill, learner_urn=learner_urn, exclude=grammar_seen)
        if item is not None:
            grammar_items.append(item)
            grammar_seen.add(item)
            state.practice_log.record(skill, today)
        attempts += 1

    combined = vocab_items + grammar_items
    (rng or np.random.default_rng()).shuffle(combined)
    return combined


if __name__ == "__main__":
    from extras.vocab_scheduler import new_vocab_card
    from ..urnings.urnings_tracker import UrnState

    class MockProvider:
        def __init__(self):
            self._vocab_counter = 0
            self._grammar_counter = 0
            self.received_learner_urns = []

        def next_new_vocab_item(self, user_id, exclude):
            self._vocab_counter += 1
            return f"vocab-new-{self._vocab_counter}"

        def next_new_grammar_item(self, skill_id, learner_urn=None, exclude=frozenset()):
            self.received_learner_urns.append(learner_urn)
            self._grammar_counter += 1
            return f"grammar-{skill_id}-{self._grammar_counter}"

    from datetime import timedelta, timezone
    created_at = datetime.now(timezone.utc)
    today = created_at + timedelta(minutes=1)
    state = UserDrillState(
        user_id="u1",
        skill_ids=["particles", "honorifics"],
        vocab_cards={"vocab-old-1": new_vocab_card(1)},
    )
    state.skill_profile.set("particles", UrnState(k=10, n=32))
    state.skill_profile.set("honorifics", UrnState(k=28, n=32))
    state.practice_log.record("honorifics", today.date())

    mock = MockProvider()
    drill = assemble_daily_drill(state, mock, DailyDrillConfig(n_questions=10, vocab_ratio=0.4), now=today)
    assert len(drill) == 10
    assert "vocab-old-1" in drill
    assert all("honorifics" not in item for item in drill)

    assert mock.received_learner_urns
    assert all(urn == UrnState(k=10, n=32) for urn in mock.received_learner_urns)
    assert sum(1 for item in drill if item.startswith("grammar-")) == 6

    later = today + timedelta(days=31)
    drill2 = assemble_daily_drill(state, MockProvider(), DailyDrillConfig(n_questions=10, vocab_ratio=0.4), now=later)
    assert any("honorifics" in item for item in drill2)

    print("all smoke tests passed")
