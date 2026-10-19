"""
Grammar/skill scheduling (PRD §2.2 vs §2.1): "due" for a skill is NOT a
calendar date like FSRS -- grammar items are LLM-generated and mostly
non-repeating, so there is no fixed card to re-review. Instead, a skill is
"due" for practice when it is not yet mastered: weak (low k/n) or still
uncertain (wide urn), using the same primitives as the mastery criterion
and the uncertainty-gated escalation (urnings_tracker.py).

A skill that reaches mastery is NOT dropped from rotation forever: Urnings
has no time-based decay, so a mastered urn would otherwise stay "mastered"
indefinitely even if the learner forgets the skill from disuse. Mastered
skills get a periodic low-priority recheck instead (MASTERED_RECHECK_DAYS)
so mastery stays revocable in practice, not just in theory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from urnings_tracker import LearnerSkillProfile, is_mastered

MASTERED_RECHECK_DAYS = 30


@dataclass
class SkillPracticeLog:
    last_practiced: dict[str, date] = field(default_factory=dict)

    def record(self, skill_id: str, today: date) -> None:
        self.last_practiced[skill_id] = today

    def days_since(self, skill_id: str, today: date) -> int | None:
        last = self.last_practiced.get(skill_id)
        return None if last is None else (today - last).days


def practice_priority(profile: LearnerSkillProfile, skill_id: str) -> float:
    """Higher = more in need of practice. Weak and/or uncertain skills first."""
    state = profile.get(skill_id)
    return (1.0 - state.proportion()) + state.width()


def skills_needing_practice(profile: LearnerSkillProfile, skill_ids: list[str],
                             log: SkillPracticeLog, today: date,
                             recheck_days: int = MASTERED_RECHECK_DAYS) -> list[str]:
    """Weak/uncertain (non-mastered) skills first, most in-need first; then
    mastered skills whose recheck cooldown has elapsed, most-overdue first.
    """
    weak = [s for s in skill_ids if not is_mastered(profile.get(s))]
    weak.sort(key=lambda s: practice_priority(profile, s), reverse=True)

    recheck = [s for s in skill_ids if s not in weak
               and (log.days_since(s, today) is None or log.days_since(s, today) >= recheck_days)]
    recheck.sort(key=lambda s: log.days_since(s, today) if log.days_since(s, today) is not None
                 else float("inf"), reverse=True)

    return weak + recheck


if __name__ == "__main__":
    from datetime import timedelta
    from urnings_tracker import UrnState

    today = date(2026, 10, 19)
    profile = LearnerSkillProfile("u1")
    profile.set("honorifics", UrnState(k=28, n=32))   # mastered, recently practiced
    profile.set("particles", UrnState(k=10, n=32))     # weak
    profile.set("conjugation", UrnState(k=29, n=32))   # mastered, practiced long ago

    log = SkillPracticeLog()
    log.record("honorifics", today - timedelta(days=2))
    log.record("particles", today - timedelta(days=2))
    log.record("conjugation", today - timedelta(days=30))

    due = skills_needing_practice(profile, ["honorifics", "particles", "conjugation"], log, today)
    assert due[0] == "particles", f"weakest skill should be first, got {due}"
    assert "honorifics" not in due, "recently-mastered skill should not resurface yet"
    assert due[-1] == "conjugation", "mastered skill overdue for recheck must still resurface, not be dropped forever"

    print("all smoke tests passed")
