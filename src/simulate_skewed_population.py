"""
What happens to Urnings when the learners are not N(0,1) but, as in a real
class, 500 beginners and 5 advanced learners (PRD §9.83)?

Same harness as simulate_item_prior.py (Rasch responses, adaptive
selection with the Metropolis-Hastings correction, items calibrated while
they are served), with the production target of 85% success
(question_bank._TARGET_SUCCESS). Two item pools:
  - wide:    beta ~ U(-3, 3), the same pool whatever the class
  - matched: the levels the class needs, beta = theta of a random learner
             - logit(0.85) + N(0, 0.5), as a generator writing items for
             its learners' current ability gives
Two populations of 505 learners, each receiving the same number of
responses:
  - balanced: theta ~ N(0, 1)
  - skewed:   500 beginners theta ~ N(-1, 0.5), 5 advanced theta ~ N(2.5, 0.3)
and two item priors:
  - flat:  every item starts at the middle of its urn (n=32)
  - model: an LLM-respondent-style prediction mu_hat = beta + N(0, 0.8)
           (correlation ~0.9 with beta here), its urn sized from the
           reported variance as urnings_tracker.seed_item_from_encoder does

Reported per group (advanced = the 5 advanced learners, or theta>1.5 in
the balanced class; others = everyone else) and per item range (easy
beta<-1; hard beta>0.25, the level the 85% target serves to theta~2.5):
ability RMSE, item difficulty RMSE, and how many responses an item in that
range received.

Usage: python3 src/simulate_skewed_population.py [--reps 4]
"""
import argparse

import numpy as np

from simulate_item_prior import seed_items_from_encoder
from simulate_trackers import RNG, _logit_urn, urnings_step

TARGET = 0.85
OFFSET = np.log(TARGET / (1 - TARGET))
N_P, N_FLAT = 32, 32


def population(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Abilities and a mask of the advanced learners."""
    if kind == "balanced":
        theta = RNG.normal(0, 1, 505)
        return theta, theta > 1.5
    theta = np.concatenate([RNG.normal(-1, 0.5, 500), RNG.normal(2.5, 0.3, 5)])
    return theta, np.arange(505) >= 500


def run(theta: np.ndarray, beta: np.ndarray, k0: np.ndarray, n0: np.ndarray, n_resp: int):
    kp = np.full(len(theta), N_P // 2)
    ki = k0.copy()
    served = np.zeros(len(beta), dtype=int)
    for _ in range(n_resp):
        p = RNG.integers(0, len(theta))
        served[urnings_step(kp, ki, p, theta[p], beta, N_P, n0, TARGET)] += 1
    return _logit_urn(kp, N_P), _logit_urn(ki, n0), served


def rmse(est: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((est[mask] - true[mask]) ** 2))) if mask.any() else float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--items", type=int, default=300)
    parser.add_argument("--responses", type=int, default=60_000, help="about 120 per learner")
    args = parser.parse_args()

    rows: dict[tuple[str, str], list[list[float]]] = {}
    for rep in range(args.reps):
        for pop in ("balanced", "skewed"):
            theta, adv = population(pop)
            pools = {"wide": RNG.uniform(-3, 3, args.items),
                     "matched": RNG.choice(theta, args.items) - OFFSET + RNG.normal(0, 0.5, args.items)}
            for pool, beta in pools.items():
                easy, hard = beta < -1, beta > 0.25
                k_m, n_m, _ = seed_items_from_encoder(beta, 0.8, sigma2=0.8 ** 2)
                priors = {"flat": (np.full(args.items, N_FLAT // 2), np.full(args.items, N_FLAT)),
                          "model": (k_m, n_m)}
                for prior, (k0, n0) in priors.items():
                    th, be, served = run(theta, beta, k0, n0, args.responses)
                    rows.setdefault((pop, pool, prior), []).append([
                        rmse(th, theta, ~adv), rmse(th, theta, adv),
                        rmse(be, beta, easy), rmse(be, beta, hard),
                        served[easy].mean(), served[hard].mean(), hard.sum()])
        print(f"replication {rep + 1}/{args.reps} done")

    print(f"\n{'population':10s} {'pool':8s} {'prior':6s} | {'ability RMSE':^19s} | {'item RMSE':^15s} | "
          f"{'responses/item':^15s} | {'hard':>5s}")
    print(f"{'':10s} {'':8s} {'':6s} | {'others':>8s} {'advanced':>9s} | {'easy':>6s} {'hard':>7s} | "
          f"{'easy':>6s} {'hard':>7s} | {'items':>5s}")
    for (pop, pool, prior), vals in rows.items():
        m = np.nanmean(vals, axis=0)
        print(f"{pop:10s} {pool:8s} {prior:6s} | {m[0]:8.2f} {m[1]:9.2f} | {m[2]:6.2f} {m[3]:7.2f} | "
              f"{m[4]:6.0f} {m[5]:7.1f} | {m[6]:5.0f}")
