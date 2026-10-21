"""
Production Urnings tracker: per-(user, skill) and per-item urn state, as a
reusable module rather than a simulation loop.

Implements the same ball-exchange + Metropolis-Hastings update as
`simulate_trackers.run_urnings`, but operating on named, persistable state
objects for one event at a time -- what a real request handler calls.

Backs three PRD decisions (docs/PRD.md §2.1, §3.3):
  - Novelty 1: seed_item_from_encoder() converts an encoder's (mu_hat,
    sigma_hat2) difficulty prediction into initial urn composition/size.
  - Novelty 3: should_escalate() gates expensive feedback on urn width.
  - Mastery criterion: is_mastered() is k/n alone (PRD §9.28) -- it used
    to also gate on width, presented as reusing the escalation primitive
    for a second independent signal; proven structurally impossible for
    k >= n/2 at any n, not just mistuned, and dropped rather than
    re-tuned. See is_mastered()'s own docstring for the proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

SIG = lambda z: 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class UrnState:
    k: int
    n: int

    def __post_init__(self):
        if not (0 <= self.k <= self.n):
            raise ValueError(f"k={self.k} must be in [0, n={self.n}]")

    def proportion(self) -> float:
        """Raw k/n -- the number to show as a user-facing competency %."""
        return self.k / self.n

    def logit_estimate(self) -> float:
        """Continuity-corrected logit; used internally, never for display."""
        pi = (self.k + 0.5) / (self.n + 1.0)
        return float(np.log(pi / (1.0 - pi)))

    def width(self) -> float:
        """Beta(k+.5, n-k+.5) posterior SD on the proportion scale.

        Bounded and well-behaved at k=0 or k=n (unlike a logit-scale SE,
        which blows up at the boundary) -- required for is_mastered() to
        behave sensibly for a learner who is answering almost everything
        right.
        """
        a, b = self.k + 0.5, self.n - self.k + 0.5
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return float(2.0 * np.sqrt(var))

    def to_dict(self) -> dict:
        return {"k": self.k, "n": self.n}

    @staticmethod
    def from_dict(d: dict) -> "UrnState":
        return UrnState(k=d["k"], n=d["n"])


def seed_item_from_encoder(mu_hat: float, sigma_hat2: float,
                            n_min: int = 8, n_max: int = 64,
                            ref_var: float = 1.0) -> UrnState:
    """Novelty 1: encoder (mu_hat, sigma_hat2) -> initial item urn.

    mu_hat is the predicted item difficulty on the logit scale (same scale
    as learner theta). sigma_hat2 controls prior *strength*: a confident
    (low-variance) prediction gets a larger urn (slower to be overwritten
    by real responses); an uncertain one gets a smaller urn.

    The n_min/n_max/ref_var mapping is a documented placeholder, not a
    validated result -- PRD §5 still requires the encoder-prior-vs-uniform
    convergence experiment before this mapping is trusted in production.
    """
    if sigma_hat2 < 0:
        raise ValueError("sigma_hat2 must be >= 0")
    n = n_max / (1.0 + sigma_hat2 / ref_var)
    n = int(round(min(max(n, n_min), n_max)))
    # The item urn's proportion is a DIFFICULTY in update()'s kernel (a high
    # pi_i lowers P(correct)), so a harder item gets more balls. This was
    # SIG(-mu_hat) before the 2026-10-21 review, seeding every item with the
    # mirror image of its predicted difficulty (PRD §9.68).
    pi = float(SIG(mu_hat))
    k = int(round(pi * (n + 1) - 0.5))
    k = min(max(k, 0), n)
    return UrnState(k=k, n=n)


SELECT_SIGMA = 0.5


def selection_probs(theta_sel, beta_hat, target: float = 0.5, sigma: float = SELECT_SIGMA) -> np.ndarray:
    """Probability of serving each candidate item: a Gaussian kernel on how
    far the item's predicted success probability (on the logit scale) is
    from `target`, normalised over the candidates (Gergely et al., 2025,
    Eq. 17). Probabilistic rather than "closest few", because update()'s
    adaptive-selection correction needs the actual probability that the
    served item was selected, before and after the proposed exchange.
    """
    beta_hat = np.asarray(beta_hat, dtype=float)
    d = (theta_sel - beta_hat - np.log(target / (1 - target))) / sigma
    d2 = d * d
    w = np.exp(-0.5 * (d2 - d2.min()))  # shifted for numerical stability; cancels in the ratio
    return w / w.sum()


def update(learner: UrnState, item: UrnState, correct: bool,
           rng: np.random.Generator,
           selection_prob: Callable[[UrnState, UrnState], float] | None = None) -> tuple[UrnState, UrnState]:
    """One response event: ball-exchange proposal + MH acceptance.

    Mirrors simulate_trackers.urnings_step, generalised to arbitrary (not
    just n=32) urn sizes since item urns can vary in size.

    selection_prob(learner, item): the probability that this item would
    have been served given those urn states. Pass it whenever the item was
    chosen adaptively from the urns (BankContentProvider): the acceptance
    ratio is then multiplied by p(item | proposed) / p(item | current),
    without which adaptive selection inflates the rating scale exactly as it
    does for Elo (Bolsinova et al., 2022; Gergely et al., 2025, Eq. 4;
    simulate_trackers.py: rating SD 2.63 without, 1.00 with, true 1.04).
    Leave it None for an item that was not selected from the urn states (a
    freshly generated one).
    """
    n_p, n_i = learner.n, item.n
    eps = 1e-9
    pi_p = float(np.clip(learner.k / n_p, eps, 1 - eps))
    pi_i = float(np.clip(item.k / n_i, eps, 1 - eps))

    x = 1 if correct else 0
    # y: a response simulated from the CURRENT urn state, needed by the
    # ball-exchange proposal (see Bolsinova et al. 2022).
    num = pi_p * (1 - pi_i)
    q = num / (num + (1 - pi_p) * pi_i)
    y = int(rng.random() < q)

    kp_s, ki_s = learner.k + x - y, item.k - x + y
    if not (0 <= kp_s <= n_p and 0 <= ki_s <= n_i):
        return learner, item

    if kp_s == learner.k:
        return learner, item  # x == y: nothing to exchange

    new_learner, new_item = UrnState(k=kp_s, n=n_p), UrnState(k=ki_s, n=n_i)
    cur = learner.k * (n_i - item.k) + (n_p - learner.k) * item.k
    pro = kp_s * (n_i - ki_s) + (n_p - kp_s) * ki_s
    ratio = np.inf if cur <= 0 else pro / cur
    if selection_prob is not None:
        ratio *= selection_prob(new_learner, new_item) / selection_prob(learner, item)
    if rng.random() >= min(1.0, ratio):
        return learner, item
    return new_learner, new_item


def should_escalate(learner: UrnState, tau: float = 0.15) -> bool:
    """Novelty 3: uncertainty-gated escalation, w_k > tau.

    tau=0.15, not the previous 0.2: width() is bounded by the urn size.
    For the n=32 urns used everywhere in this codebase (LearnerSkillProfile's
    n_default, every example/smoke-test urn), the maximum possible width is
    ~0.1715 (at k=16, the most uncertain state) -- found by computing it
    directly, not assumed. tau=0.2 was therefore UNREACHABLE: this branch
    could never return True for any learner, at any k, ever -- the whole
    width-based escalation trigger was dead code in production
    (answer_processing.py calls should_escalate() with no tau override, so
    it always used this default). 0.15 fires for k in [12, 20] out of
    [0, 32] -- a genuinely-uncertain band around 50/50, not the extremes --
    while staying under the n=32 ceiling. If n_default ever changes, recompute
    the ceiling (2*sqrt(Beta(k+.5, n-k+.5) variance) at k=n/2) before trusting
    this default again.
    """
    return learner.width() > tau


def should_escalate_full(learner: UrnState, consecutive_misconceptions: int, tau: float = 0.15) -> bool:
    """Eq. 5 in full: w_k > tau  OR  consecutive(m_j) >= 2. should_escalate()
    above only ever implemented the first disjunct -- the second had
    nothing to read from anywhere in this codebase until
    LearnerSkillProfile.record_outcome() (PRD §9.38) started tracking it.
    Use this, not should_escalate() alone, wherever Eq. 5's actual trigger
    is meant to fire -- answer_processing.py was silently only checking
    half of it before this was found and fixed.
    """
    return should_escalate(learner, tau) or consecutive_misconceptions >= 2


def is_mastered(learner: UrnState, k_threshold: float = 0.8) -> bool:
    """Mastery criterion (PRD §2.1, corrected PRD §9.28): k/n >= k_threshold.

    Used to carry a second "AND width() < width_threshold" condition,
    presented as an independent "high estimate AND low uncertainty" check
    reusing the escalation primitive. Proven structurally impossible, not
    just mistuned: width(k, n) = 2*sqrt(Beta(k+.5, n-k+.5) variance) is
    monotonically DECREASING in k for every k >= n/2, for ANY n (the Beta
    variance is maximal at the midpoint by construction) -- so a width<tau
    condition on the SAME (k, n) pair the k/n condition reads is always
    reducible to some other k-threshold, never an independent second
    signal. Checked directly for n=32/64/128 at k_threshold=0.8,
    width_threshold=0.15: width is already below 0.15 the moment k/n first
    reaches 0.8, at every n tested -- the clause never excluded anything
    the k/n clause didn't already exclude. Dropped rather than re-tuned,
    since no width_threshold value could fix this for k >= n/2 -- if a
    genuinely independent "how settled is this" signal is wanted later, it
    needs a different source of evidence (e.g. recent k-trend volatility,
    not a snapshot of current (k, n)), not a second snapshot-based
    threshold on the same urn.

    Revocable by construction: if new wrong answers push k/n back down,
    this simply stops returning True on the next call -- there is no
    separate "mastered" flag to unset.
    """
    return learner.proportion() >= k_threshold


class LearnerSkillProfile:
    """Per-user vector of per-skill UrnStates (PRD §2.1 "Multi-skill")."""

    def __init__(self, user_id: str, n_default: int = 32):
        self.user_id = user_id
        self.n_default = n_default
        self._skills: Dict[str, UrnState] = {}
        # Eq. 5's consecutive(m_j) >= 2 trigger (PRD §9.38): per-skill
        # (last misconception KAGAS code, consecutive count). Found
        # entirely unimplemented -- answer_processing.py only ever
        # checked should_escalate()'s width disjunct; this half of the
        # OR had nothing to read from anywhere in the codebase.
        self._consecutive: Dict[str, tuple] = {}

    def get(self, skill_id: str) -> UrnState:
        if skill_id not in self._skills:
            self._skills[skill_id] = UrnState(k=self.n_default // 2, n=self.n_default)
        return self._skills[skill_id]

    def set(self, skill_id: str, state: UrnState) -> None:
        self._skills[skill_id] = state

    def record_outcome(self, skill_id: str, correct: bool, misconception_code: str | None = None) -> int:
        """Update and return the consecutive-same-misconception count for
        this skill: resets to 0 on a correct answer or when no
        misconception code is available (e.g. subj_topic, which carries
        kagas_code=None by design); increments only when the SAME code
        repeats back-to-back, otherwise restarts at 1.
        """
        if correct or misconception_code is None:
            self._consecutive[skill_id] = (None, 0)
            return 0
        last_code, count = self._consecutive.get(skill_id, (None, 0))
        count = count + 1 if misconception_code == last_code else 1
        self._consecutive[skill_id] = (misconception_code, count)
        return count

    def consecutive_misconceptions(self, skill_id: str) -> int:
        return self._consecutive.get(skill_id, (None, 0))[1]

    def competency_percent(self, skill_id: str) -> float:
        return 100.0 * self.get(skill_id).proportion()

    def mastered_skills(self, **kwargs) -> list[str]:
        return [s for s, state in self._skills.items() if is_mastered(state, **kwargs)]

    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "n_default": self.n_default,
                "skills": {s: st.to_dict() for s, st in self._skills.items()},
                "consecutive": {s: list(v) for s, v in self._consecutive.items()}}

    @staticmethod
    def from_dict(d: dict) -> "LearnerSkillProfile":
        p = LearnerSkillProfile(d["user_id"], d["n_default"])
        p._skills = {s: UrnState.from_dict(st) for s, st in d["skills"].items()}
        p._consecutive = {s: tuple(v) for s, v in d.get("consecutive", {}).items()}
        return p


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    learner = UrnState(k=16, n=32)
    item = seed_item_from_encoder(mu_hat=0.5, sigma_hat2=0.1)
    assert 8 <= item.n <= 64

    weak_prior_item = seed_item_from_encoder(mu_hat=0.5, sigma_hat2=10.0)
    assert weak_prior_item.n < item.n, "higher encoder variance must mean a smaller (weaker) prior urn"

    l, i = learner, item
    for _ in range(500):
        l, i = update(l, i, correct=True, rng=rng)
    assert l.proportion() > 0.8, f"learner who always answers correctly should trend high, got {l.proportion()}"
    assert is_mastered(l), "500 correct answers should be enough to be declared mastered"

    # Real assertions, not "x or not x" (that vacuous form is exactly how
    # the tau=0.2 bug above went undetected -- it can never fail).
    assert should_escalate(UrnState(k=16, n=32)), (
        "maximally uncertain urn (k=n/2) must escalate -- if this fails, "
        "tau is above the achievable width ceiling for this urn size again")
    assert not should_escalate(UrnState(k=0, n=32))
    assert not should_escalate(UrnState(k=32, n=32))

    mastered = UrnState(k=28, n=32)
    assert is_mastered(mastered)
    regressed = mastered
    for _ in range(30):
        regressed, _ = update(regressed, UrnState(k=16, n=32), correct=False, rng=rng)
    assert not is_mastered(regressed), "mastery must be revocable after repeated wrong answers"

    profile = LearnerSkillProfile("u1")
    profile.set("honorifics", mastered)
    assert profile.mastered_skills() == ["honorifics"]
    assert profile.competency_percent("honorifics") == 87.5

    d = profile.to_dict()
    assert LearnerSkillProfile.from_dict(d).competency_percent("honorifics") == 87.5

    # Eq. 5's consecutive(m_j) >= 2 trigger (PRD §9.38) -- real assertions,
    # not just "must not raise": a narrow (low-width) urn so should_escalate()
    # alone would say no, isolating this mechanism from the width disjunct.
    narrow = UrnState(k=29, n=32)  # width well under tau=0.15
    assert not should_escalate(narrow), "test setup must start width-safe to isolate this check"
    cprofile = LearnerSkillProfile("u2")
    assert cprofile.consecutive_misconceptions("particle_loc") == 0
    c1 = cprofile.record_outcome("particle_loc", correct=False, misconception_code="PART")
    assert c1 == 1 and not should_escalate_full(narrow, c1), "one wrong answer must not escalate"
    c2 = cprofile.record_outcome("particle_loc", correct=False, misconception_code="PART")
    assert c2 == 2 and should_escalate_full(narrow, c2), (
        "two CONSECUTIVE wrong answers with the SAME misconception must escalate "
        "even when width alone would not")
    c3 = cprofile.record_outcome("particle_loc", correct=False, misconception_code="END")
    assert c3 == 1, "a DIFFERENT misconception must restart the streak at 1, not keep counting"
    c4 = cprofile.record_outcome("particle_loc", correct=True)
    assert c4 == 0, "a correct answer must reset the streak to 0"
    assert cprofile.consecutive_misconceptions("particle_loc") == 0
    # subj_topic carries kagas_code=None by design (§9.11) -- must never
    # accumulate a streak, not silently count None==None as a match.
    s1 = cprofile.record_outcome("subj_topic", correct=False, misconception_code=None)
    s2 = cprofile.record_outcome("subj_topic", correct=False, misconception_code=None)
    assert s1 == 0 and s2 == 0, "a skill with no misconception code must never trigger this path"

    print("all smoke tests passed")
