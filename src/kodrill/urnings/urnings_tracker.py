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
        return self.k / self.n

    def logit_estimate(self) -> float:
        pi = (self.k + 0.5) / (self.n + 1.0)
        return float(np.log(pi / (1.0 - pi)))

    def width(self) -> float:
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
    if sigma_hat2 < 0:
        raise ValueError("sigma_hat2 must be >= 0")
    n = n_max / (1.0 + sigma_hat2 / ref_var)
    n = int(round(min(max(n, n_min), n_max)))
    pi = float(SIG(mu_hat))
    k = int(round(pi * (n + 1) - 0.5))
    k = min(max(k, 0), n)
    return UrnState(k=k, n=n)


SELECT_SIGMA = 0.5


def selection_probs(theta_sel, beta_hat, target: float = 0.5, sigma: float = SELECT_SIGMA) -> np.ndarray:
    beta_hat = np.asarray(beta_hat, dtype=float)
    d = (theta_sel - beta_hat - np.log(target / (1 - target))) / sigma
    d2 = d * d
    w = np.exp(-0.5 * (d2 - d2.min()))
    return w / w.sum()


def update(learner: UrnState, item: UrnState, correct: bool,
           rng: np.random.Generator,
           selection_prob: Callable[[UrnState, UrnState], float] | None = None) -> tuple[UrnState, UrnState]:
    n_p, n_i = learner.n, item.n
    eps = 1e-9
    pi_p = float(np.clip(learner.k / n_p, eps, 1 - eps))
    pi_i = float(np.clip(item.k / n_i, eps, 1 - eps))

    x = 1 if correct else 0
    num = pi_p * (1 - pi_i)
    q = num / (num + (1 - pi_p) * pi_i)
    y = int(rng.random() < q)

    kp_s, ki_s = learner.k + x - y, item.k - x + y
    if not (0 <= kp_s <= n_p and 0 <= ki_s <= n_i):
        return learner, item
    if kp_s == learner.k:
        return learner, item

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
    return learner.width() > tau


def should_escalate_full(learner: UrnState, consecutive_wrong: int, tau: float = 0.15) -> bool:
    return should_escalate(learner, tau) or consecutive_wrong >= 2


def is_mastered(learner: UrnState, k_threshold: float = 0.8) -> bool:
    return learner.proportion() >= k_threshold


class LearnerSkillProfile:
    def __init__(self, user_id: str, n_default: int = 32):
        self.user_id = user_id
        self.n_default = n_default
        self._skills: Dict[str, UrnState] = {}
        self._consecutive: Dict[str, int] = {}

    def get(self, skill_id: str) -> UrnState:
        if skill_id not in self._skills:
            self._skills[skill_id] = UrnState(k=self.n_default // 2, n=self.n_default)
        return self._skills[skill_id]

    def set(self, skill_id: str, state: UrnState) -> None:
        self._skills[skill_id] = state

    def record_outcome(self, skill_id: str, correct: bool) -> int:
        count = 0 if correct else self._consecutive.get(skill_id, 0) + 1
        self._consecutive[skill_id] = count
        return count

    def consecutive_wrong(self, skill_id: str) -> int:
        return self._consecutive.get(skill_id, 0)

    def competency_percent(self, skill_id: str) -> float:
        return 100.0 * self.get(skill_id).proportion()

    def mastered_skills(self, **kwargs) -> list[str]:
        return [s for s, state in self._skills.items() if is_mastered(state, **kwargs)]

    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "n_default": self.n_default,
                "skills": {s: st.to_dict() for s, st in self._skills.items()},
                "consecutive": dict(self._consecutive)}

    @staticmethod
    def from_dict(d: dict) -> "LearnerSkillProfile":
        p = LearnerSkillProfile(d["user_id"], d["n_default"])
        p._skills = {s: UrnState.from_dict(st) for s, st in d["skills"].items()}
        p._consecutive = dict(d.get("consecutive", {}))
        return p


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    learner = UrnState(k=16, n=32)
    item = seed_item_from_encoder(mu_hat=0.5, sigma_hat2=0.1)
    assert 8 <= item.n <= 64

    weak_prior_item = seed_item_from_encoder(mu_hat=0.5, sigma_hat2=10.0)
    assert weak_prior_item.n < item.n

    l, i = learner, item
    for _ in range(500):
        l, i = update(l, i, correct=True, rng=rng)
    assert l.proportion() > 0.8, f"learner who always answers correctly should trend high, got {l.proportion()}"
    assert is_mastered(l)

    assert should_escalate(UrnState(k=16, n=32))
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

    narrow = UrnState(k=29, n=32)
    assert not should_escalate(narrow)
    cprofile = LearnerSkillProfile("u2")
    assert cprofile.consecutive_wrong("particle_loc") == 0
    c1 = cprofile.record_outcome("particle_loc", correct=False)
    assert c1 == 1 and not should_escalate_full(narrow, c1)
    c2 = cprofile.record_outcome("particle_loc", correct=False)
    assert c2 == 2 and should_escalate_full(narrow, c2)
    c3 = cprofile.record_outcome("particle_loc", correct=True)
    assert c3 == 0
    assert cprofile.consecutive_wrong("particle_loc") == 0

    print("all smoke tests passed")
