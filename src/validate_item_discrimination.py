"""
Item discrimination index (point-biserial correlation between a binary
item score and examinee ability) -- the classical test theory metric for
"does getting this item right actually distinguish strong from weak
learners" (Crocker & Algina; conventional rule of thumb: <0.10 poor/should
be revised, 0.10-0.19 marginal, 0.20-0.29 reasonably good, >=0.30 good).
Missing from every validation script in this repo until now: generation
pass rate and solver key-acceptance (now validate_item_foundry.py) check whether
a distractor is morphologically valid and whether a blind model can find
the key, but neither checks the classical property that a well-constructed
item is SUPPOSED to have -- correlating with the thing the test is for.

For a true Rasch generating model (used throughout this repo's
simulate_*.py scripts), point-biserial correlation with ability reduces
exactly to Pearson correlation between the 0/1 response and theta, so no
new formula is needed beyond what simulate_trackers.py already computes
response probabilities with.

KO-DRILL-SPECIFIC QUESTION this answers, not just a generic sanity check:
Section IV-D (paper) states, but does not measure, that targeting 85%
success (Eq. 4, for pedagogical reasons) "deliberately sacrifices
measurement efficiency for learning efficiency" and calls a direct
comparison "a natural comparison for future work rather than one of the
two claims this paper sets out to validate." This script IS that
comparison: discrimination as a function of item difficulty, evaluated
at the 50%-target difficulty (classical maximum-information CAT) versus
the 85%-target difficulty (this system's actual policy, for an
average-ability learner) against the same population.

Important limitation, stated plainly: this is a FIXED-item field-test
simulation (every learner takes the same item at a given difficulty),
the standard way discrimination indices are computed and the standard
precondition for validating an item before adaptive use. It is NOT a
simulation of the adaptive deployment itself, where each learner is
served a DIFFERENT item matched to their own estimated ability -- that
changes what "discrimination" would even mean (every learner would be
at their own personal ~85% point by design), a different and harder
question this script does not attempt.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

RNG = np.random.default_rng(20261020)
SIG = lambda z: 1.0 / (1.0 + np.exp(-z))
LOGIT = lambda p: np.log(p / (1.0 - p))


def discrimination_at(beta: float, theta: np.ndarray) -> float:
    """Point-biserial correlation between response to a single item of
    difficulty `beta` and the population's true ability `theta`."""
    p_correct = SIG(theta - beta)
    responses = (RNG.random(len(theta)) < p_correct).astype(float)
    if responses.std() == 0:
        return float("nan")  # everyone got it right/wrong -- undefined, not zero
    return float(np.corrcoef(responses, theta)[0, 1])


def discrimination_curve(n_learners: int, beta_grid: np.ndarray, theta_std: float = 1.0,
                          n_replications: int = 40) -> dict:
    theta = RNG.normal(0, theta_std, n_learners)
    rows = []
    for beta in beta_grid:
        reps = [discrimination_at(beta, theta) for _ in range(n_replications)]
        reps = [r for r in reps if not np.isnan(r)]
        avg_success = float(SIG(theta - beta).mean())
        rows.append({
            "beta": float(beta),
            "avg_success_rate": avg_success,
            "mean_discrimination": float(np.mean(reps)) if reps else float("nan"),
            "sd_discrimination": float(np.std(reps)) if reps else float("nan"),
        })
    return {"n_learners": n_learners, "theta_std": theta_std, "rows": rows}


def quality_label(r: float) -> str:
    if r < 0.10:
        return "poor (should be revised)"
    if r < 0.20:
        return "marginal"
    if r < 0.30:
        return "reasonably good"
    return "good"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-learners", type=int, default=2000)
    parser.add_argument("--target-policy", type=float, default=0.85,
                         help="this system's actual Eq. 4 success target")
    parser.add_argument("--out", default="item_discrimination_results.json")
    args = parser.parse_args()

    beta_grid = np.linspace(-3.0, 3.0, 25)
    result = discrimination_curve(args.n_learners, beta_grid)

    print(f"n_learners={result['n_learners']}  theta ~ N(0, {result['theta_std']})\n")
    print(f"{'beta':>6s} {'avg success':>12s} {'discrimination':>15s} {'quality':>22s}")
    for r in result["rows"]:
        print(f"{r['beta']:6.2f} {r['avg_success_rate']:12.1%} "
              f"{r['mean_discrimination']:15.3f} {quality_label(r['mean_discrimination']):>22s}")

    beta_50 = 0.0  # difficulty = population mean ability -> ~50% success for an average learner
    # Eq. 4's implied beta for theta=0: sigma(0 - beta) = target, so
    # beta = -logit(target) (= -1.73 for 85%, an EASY item). This used to be
    # -LOGIT(1 - target) = +1.73, i.e. a 15%-success item; the reported
    # discrimination happened to be unchanged only because theta ~ N(0, 1)
    # is symmetric, while avg_success_rate came out as 19% (review 2026-10-21).
    beta_policy = -LOGIT(args.target_policy)

    d50 = discrimination_curve(args.n_learners, np.array([beta_50]), n_replications=200)["rows"][0]
    dpolicy = discrimination_curve(args.n_learners, np.array([beta_policy]), n_replications=200)["rows"][0]

    print(f"\n=== Section IV-D's unresolved trade-off, now measured ===")
    print(f"50%-target (max-information CAT):    beta={beta_50:+.2f}  "
          f"discrimination={d50['mean_discrimination']:.3f} ({quality_label(d50['mean_discrimination'])})")
    print(f"{args.target_policy:.0%}-target (this system's Eq. 4 policy): beta={beta_policy:+.2f}  "
          f"discrimination={dpolicy['mean_discrimination']:.3f} "
          f"({quality_label(dpolicy['mean_discrimination'])})")
    assert abs(dpolicy["avg_success_rate"] - args.target_policy) < 0.1, (
        f"the policy item must give an average learner roughly the target success rate, "
        f"got {dpolicy['avg_success_rate']:.1%} for target {args.target_policy:.0%}")
    rel_drop = 1 - dpolicy["mean_discrimination"] / d50["mean_discrimination"]
    print(f"relative drop in discrimination from choosing learning over measurement: {rel_drop:.1%}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"curve": result, "fifty_percent_target": d50,
                    "policy_target": dpolicy, "relative_discrimination_drop": rel_drop}, f, indent=2)
    print(f"\nFull results written to {args.out}")
