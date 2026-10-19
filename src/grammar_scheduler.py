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

MASTERED_RECHECK_DAYS is an explicit guess, not a validated constant, and
this is a genuinely different problem from the KLUE-RoBERTa/difficulty
gap: tested here (2026-10-19) whether FSRS's own, already-validated
forgetting-curve math could derive a principled number instead of a
guess -- it doesn't transfer. FSRS's intervals grow because each
simulated review is assumed SPACED (that's the whole mechanism); feeding
it many Good reviews in a row (an attempt to model "mastered after many
Urnings responses") makes the interval blow up to tens of thousands of
days, because that's modelling long-term fact retention, not "how much
practice evidence constitutes skill mastery" -- a different question
FSRS was never built to answer. No amount of code fixes a missing
Korean-skill-retention dataset, same root cause as the KLUE-RoBERTa gap.

What IS fixable without that data: not leaving this as a silent constant.
RecheckOutcomeLog below records what actually happens at each recheck, so
once real usage exists, this number can be tuned from evidence instead of
guessed again. Wire profile.record_outcome() in wherever an answer to a
recheck-sourced item gets processed (that handler doesn't exist yet in
this repo -- daily_drill.py only assembles items, it doesn't yet process
answers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from urnings_tracker import LearnerSkillProfile, is_mastered

MASTERED_RECHECK_DAYS = 30  # placeholder -- see module docstring; tune from RecheckOutcomeLog once real data exists


@dataclass
class RecheckOutcome:
    skill_id: str
    days_since_last_practice: int
    still_mastered: bool  # did the recheck response confirm mastery, or reveal regression?


@dataclass
class RecheckOutcomeLog:
    """Evidence for eventually tuning MASTERED_RECHECK_DAYS from real data,
    instead of guessing again. Record one outcome every time a recheck-
    sourced item (see skills_needing_practice's "recheck" list) gets
    answered -- if regressions are consistently found far later than the
    configured interval, the interval is too short (wasting practice
    slots); if never found, it's too long (mastery can silently lapse for
    longer than intended).
    """
    outcomes: list[RecheckOutcome] = field(default_factory=list)

    def record(self, skill_id: str, days_since_last_practice: int, still_mastered: bool) -> None:
        self.outcomes.append(RecheckOutcome(skill_id, days_since_last_practice, still_mastered))

    def regression_rate(self) -> float | None:
        """Fraction of rechecks that found a regression. None if no data yet.

        This single number is NOT a recommendation by itself -- it needs
        real volume (dozens+ of rechecks) before it means anything, and no
        such volume exists yet. It's here so that once it does, the
        question "is 30 days right?" has an actual number to look at
        instead of another guess.
        """
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

    outcome_log = RecheckOutcomeLog()
    assert outcome_log.regression_rate() is None, "no data yet -- must not fabricate a rate from nothing"
    outcome_log.record("conjugation", days_since_last_practice=30, still_mastered=True)
    outcome_log.record("honorifics", days_since_last_practice=45, still_mastered=False)
    assert outcome_log.regression_rate() == 0.5

    print("all smoke tests passed")
