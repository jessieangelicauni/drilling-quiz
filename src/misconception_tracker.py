"""
Misconception tracking: which specific KAGAS misconception a learner holds,
learned from WHICH wrong option they pick, not just whether they were
wrong (Novelty 1, PRD §9.69).

Why the chosen distractor carries information. Every distractor in an
item realises one KAGAS error code (GeneratedItem.option_codes). A learner
who holds misconception m is drawn to the option realising m: the nested
logit view of multiple-choice items (Suh & Bolt, 2010, Psychometrika).
First, whether the learner solves the item at all (the skill urn,
urnings_tracker). Then, given a wrong answer, which distractor attracts
them, a multinomial-logit (Luce) choice among the distractors actually on
offer:

    P(pick d_j | wrong) = exp(eta_{p,m_j}) / sum_k exp(eta_{p,m_k})

How it is tracked. One propensity eta per (learner, skill, misconception),
updated online after every wrong answer by one stochastic-gradient step on
that choice's log-likelihood -- the same relationship Elo has to the
Bradley-Terry model, generalised from pairs to a choice among the offered
set. Two properties fall out of the likelihood, and both are measured in
simulate_misconceptions.py:
  - availability cancels: a misconception is only ever compared with the
    others that were on offer in the same item, so being offered more
    often does not by itself make it look more attractive (a raw count of
    picks does not have this property);
  - change is followed: a constant step size weights recent choices more,
    so when a misconception is remediated and another takes over, the
    estimate moves to it (an all-history estimate, like a pick rate or a
    batch maximum-likelihood fit, keeps the old one on top for longer).

Tried first and dropped (PRD §9.69): one Urnings urn per misconception,
updated by pairwise ball exchanges. It followed change well but was
clearly less accurate than this when misconceptions were stable, because
the bounded urns add their own sampling noise.

Correct answers carry no misconception evidence here by design (nested
logit: the distractor choice is only observed given a wrong answer).
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

# Step size of the online update. 0.4 is the middle of the range that
# simulate_misconceptions.py found best: smaller is slightly more accurate
# when misconceptions never change, larger follows a change faster.
LEARNING_RATE = 0.4


class MisconceptionProfile:
    """Per-learner misconception propensities, keyed by (skill_id, KAGAS code)."""

    def __init__(self, learning_rate: float = LEARNING_RATE):
        self.learning_rate = learning_rate
        self._eta: Dict[Tuple[str, str], float] = {}
        self._wrong: Dict[str, int] = {}

    def record_choice(self, skill_id: str, chosen_code: str | None,
                      offered_codes: Iterable[str | None]) -> None:
        """A wrong answer: the learner picked the distractor realising
        `chosen_code` from distractors realising `offered_codes` (the chosen
        one included). Code-less options carry no comparison and are skipped.
        """
        codes = list(dict.fromkeys(c for c in offered_codes if c is not None))
        if chosen_code is None or chosen_code not in codes or len(codes) < 2:
            return
        eta = np.array([self._eta.get((skill_id, c), 0.0) for c in codes])
        p = np.exp(eta - eta.max())
        p /= p.sum()
        grad = -p
        grad[codes.index(chosen_code)] += 1.0  # d log P(chosen) / d eta
        for code, g in zip(codes, grad):
            self._eta[(skill_id, code)] = self._eta.get((skill_id, code), 0.0) + self.learning_rate * g
        self._wrong[skill_id] = self._wrong.get(skill_id, 0) + 1

    def propensities(self, skill_id: str) -> Dict[str, float]:
        """Logit-scale propensity per misconception seen so far for this skill.
        Only differences are meaningful, as in any Luce / Bradley-Terry model."""
        return {code: v for (s, code), v in self._eta.items() if s == skill_id}

    def dominant(self, skill_id: str) -> str | None:
        """The misconception this learner is currently most drawn to, or None
        before any informative wrong answer for the skill."""
        props = self.propensities(skill_id)
        return max(props, key=props.get) if props else None

    def evidence(self, skill_id: str) -> int:
        """Number of informative wrong answers recorded for the skill."""
        return self._wrong.get(skill_id, 0)

    def to_dict(self) -> dict:
        return {"learning_rate": self.learning_rate,
                "eta": [[s, c, v] for (s, c), v in self._eta.items()],
                "wrong": dict(self._wrong)}

    @staticmethod
    def from_dict(d: dict) -> "MisconceptionProfile":
        p = MisconceptionProfile(d["learning_rate"])
        p._eta = {(s, c): v for s, c, v in d["eta"]}
        p._wrong = dict(d["wrong"])
        return p


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    others_pool = ["PART-sub-obj", "PART-sub-dir", "PART-omit"]

    profile = MisconceptionProfile()
    assert profile.dominant("particle_loc") is None

    # A learner who always falls for the same misconception, whatever else is on offer.
    for _ in range(20):
        others = list(rng.choice(others_pool, size=2, replace=False))
        profile.record_choice("particle_loc", "PART-sub-loc", ["PART-sub-loc", *others, None])
    assert profile.dominant("particle_loc") == "PART-sub-loc", profile.propensities("particle_loc")
    assert profile.evidence("particle_loc") == 20

    # Then remediated: from now on they fall for a different one, and the estimate follows.
    for _ in range(20):
        profile.record_choice("particle_loc", "PART-sub-dir", ["PART-sub-dir", "PART-sub-loc", "PART-omit"])
    assert profile.dominant("particle_loc") == "PART-sub-dir", profile.propensities("particle_loc")

    # Propensities only move relative to each other: the update's gradient sums to zero.
    assert abs(sum(profile.propensities("particle_loc").values())) < 1e-9

    # Skills are independent; code-less or single-code choices are no-ops.
    profile.record_choice("honorific", None, ["END-hon"])
    profile.record_choice("honorific", "END-hon", ["END-hon", None])
    assert profile.propensities("honorific") == {} and profile.evidence("honorific") == 0

    restored = MisconceptionProfile.from_dict(profile.to_dict())
    assert restored.propensities("particle_loc") == profile.propensities("particle_loc")
    assert restored.evidence("particle_loc") == profile.evidence("particle_loc")

    print("all smoke tests passed")
