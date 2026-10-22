from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .urnings_tracker import LearnerSkillProfile, is_mastered

MASTERED_RECHECK_DAYS = 30


@dataclass
class RecheckOutcome:
    skill_id: str
    days_since_last_practice: int
    still_mastered: bool


@dataclass
class RecheckOutcomeLog:
    outcomes: list[RecheckOutcome] = field(default_factory=list)

    def record(self, skill_id: str, days_since_last_practice: int, still_mastered: bool) -> None:
        self.outcomes.append(RecheckOutcome(skill_id, days_since_last_practice, still_mastered))

    def regression_rate(self) -> float | None:
        if not self.outcomes:
            return None
        return sum(1 for o in self.outcomes if not o.still_mastered) / len(self.outcomes)


@dataclass
class SkillPracticeLog:
    last_practiced: dict[str, date] = field(default_factory=dict)

    def record(self, skill_id: str, today: date) -> None:
        self.last_practiced[skill_id] = today

    def days_since(self, skill_id: str, today: date) -> int | None:
        last = self.last_practiced.get(skill_id)
        return None if last is None else (today - last).days


def practice_priority(profile: LearnerSkillProfile, skill_id: str) -> float:
    state = profile.get(skill_id)
    return (1.0 - state.proportion()) + state.width()


def skills_needing_practice(profile: LearnerSkillProfile, skill_ids: list[str],
                             log: SkillPracticeLog, today: date,
                             recheck_days: int = MASTERED_RECHECK_DAYS) -> list[str]:
    weak = [s for s in skill_ids if not is_mastered(profile.get(s))]
    weak.sort(key=lambda s: practice_priority(profile, s), reverse=True)

    recheck = [s for s in skill_ids if s not in weak
               and (log.days_since(s, today) is None or log.days_since(s, today) >= recheck_days)]
    recheck.sort(key=lambda s: log.days_since(s, today) if log.days_since(s, today) is not None
                 else float("inf"), reverse=True)

    return weak + recheck


if __name__ == "__main__":
    from datetime import timedelta
    from .urnings_tracker import UrnState

    today = date(2026, 10, 19)
    profile = LearnerSkillProfile("u1")
    profile.set("honorifics", UrnState(k=28, n=32))
    profile.set("particles", UrnState(k=10, n=32))
    profile.set("conjugation", UrnState(k=29, n=32))

    log = SkillPracticeLog()
    log.record("honorifics", today - timedelta(days=2))
    log.record("particles", today - timedelta(days=2))
    log.record("conjugation", today - timedelta(days=30))

    due = skills_needing_practice(profile, ["honorifics", "particles", "conjugation"], log, today)
    assert due[0] == "particles", f"weakest skill should be first, got {due}"
    assert "honorifics" not in due
    assert due[-1] == "conjugation"

    outcome_log = RecheckOutcomeLog()
    assert outcome_log.regression_rate() is None
    outcome_log.record("conjugation", days_since_last_practice=30, still_mastered=True)
    outcome_log.record("honorifics", days_since_last_practice=45, still_mastered=False)
    assert outcome_log.regression_rate() == 0.5

    print("all smoke tests passed")
