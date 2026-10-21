"""
Novelty 1 evidence (PRD §9.69): does tracking WHICH distractor a learner
picks (misconception_tracker.py) identify the misconception they actually
hold, and does it beat the obvious alternatives?

Generating model: nested logit for multiple-choice items (Suh & Bolt,
2010). Learner p answers an item correctly with sigmoid(theta_p - beta_i)
(Rasch, as in every other simulate_*.py). Given a wrong answer, they pick
among the item's offered distractors with probability proportional to
exp(eta_{p,m}) for the misconception m each distractor realises. Each
learner holds one dominant misconception (eta boosted by DOMINANCE),
the rest at N(0, 1).

Two things a real generator does that make this harder than it looks:
  - availability is uneven: some misconceptions get offered far more often
    than others (Zipf-like weights), because some errors are easier to
    realise as a clean distractor;
  - misconceptions change: halfway through, a REMEDIATED condition swaps
    each learner's dominant misconception for a different one.

Trackers compared, all fed the same responses:
  - skill only: the ordinary ability urn, no option information. It cannot
    name a misconception, so its accuracy is the best fixed guess (the
    misconception offered most often);
  - count: the misconception picked most often;
  - rate: picks divided by times offered (corrects for availability, not
    for which other distractors it was competing with);
  - batch choice model: the maximum-likelihood fit of the same Luce /
    Plackett-Luce choice model to the learner's WHOLE history (ridge 0.1,
    Newton iterations), refitted at every checkpoint -- the statistically
    efficient estimate when misconceptions never change;
  - online choice model (this system, misconception_tracker.py): one
    stochastic-gradient step on the same likelihood per wrong answer.

Metric: how often the tracker's top misconception is the learner's true
dominant one, after a given number of wrong answers (and, for the
remediated condition, wrong answers since the switch). Mean over
replications with a 95% interval over replications.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from misconception_tracker import LEARNING_RATE, MisconceptionProfile

SIG = lambda z: 1.0 / (1.0 + np.exp(-z))

N_MISCONCEPTIONS = 6       # misconception codes per skill
N_OFFERED = 3              # distractors per item (4-option items)
DOMINANCE = 1.5            # logit boost for the learner's dominant misconception
ZIPF = 1.0                 # availability skew: weight of the r-th code is 1 / r**ZIPF


def availability_weights(m: int, zipf: float) -> np.ndarray:
    w = 1.0 / np.arange(1, m + 1) ** zipf
    return w / w.sum()


def offered_codes(rng, avail) -> np.ndarray:
    return rng.choice(len(avail), size=N_OFFERED, replace=False, p=avail)


def make_eta(rng, n_learners, dominant):
    eta = rng.normal(0, 1, (n_learners, N_MISCONCEPTIONS))
    eta[np.arange(n_learners), dominant] += DOMINANCE
    return eta


def batch_choice_fit(history, iters: int = 60, ridge: float = 0.1) -> np.ndarray:
    """Penalised maximum-likelihood propensities of the Luce choice model
    from (offered codes, chosen code) pairs, by diagonal Newton steps."""
    eta = np.zeros(N_MISCONCEPTIONS)
    for _ in range(iters):
        grad = -ridge * eta
        hess = np.full(N_MISCONCEPTIONS, ridge)
        for offer, chosen in history:
            p = np.exp(eta[offer] - eta[offer].max()); p /= p.sum()
            grad[offer] -= p
            grad[chosen] += 1.0
            hess[offer] += p * (1 - p)
        eta += grad / hess
    return eta


def run(seed: int, n_learners: int, n_wrong: int, remediate: bool, zipf: float,
        learning_rate: float = LEARNING_RATE, checkpoints=(5, 10, 20, 40)) -> dict:
    """Simulates each learner until they have produced `n_wrong` wrong answers
    (in each phase, for the remediated condition) and records each tracker's
    top-1 accuracy after every wrong answer."""
    rng = np.random.default_rng(seed)
    avail = availability_weights(N_MISCONCEPTIONS, zipf)
    # The dominant misconception is drawn independently of availability --
    # an often-offered code is not more likely to be the one a learner holds.
    dominant = rng.integers(0, N_MISCONCEPTIONS, n_learners)
    eta = make_eta(rng, n_learners, dominant)
    theta = rng.normal(0, 1, n_learners)
    fixed_guess = int(np.argmax(avail))

    # Scored against the misconception the learner is ACTUALLY most drawn to
    # (argmax of eta), not the boosted one: with N(0, 1) on the others, the
    # boosted code is not always the strongest, and no tracker could find a
    # "dominant" misconception the learner doesn't behave as if they hold.
    phases = [np.argmax(eta, axis=1)]
    if remediate:
        shift = rng.integers(1, N_MISCONCEPTIONS, n_learners)
        new_dom = (dominant + shift) % N_MISCONCEPTIONS
        eta2 = eta.copy()
        eta2[np.arange(n_learners), dominant] -= DOMINANCE
        eta2[np.arange(n_learners), new_dom] += DOMINANCE
        phases.append(np.argmax(eta2, axis=1))

    names = ("skill only", "count", "rate", "batch choice model", "online choice model")
    scored = {c - 1 for c in checkpoints}
    hits = {name: np.zeros(n_wrong) for name in names}  # accuracy in the LAST phase

    for p in range(n_learners):
        profile = MisconceptionProfile(learning_rate)
        history = []
        picks = np.zeros(N_MISCONCEPTIONS)
        offers = np.zeros(N_MISCONCEPTIONS)
        for phase, truth in enumerate(phases):
            eta_p = eta[p] if phase == 0 else eta2[p]
            wrong = 0
            while wrong < n_wrong:
                beta = rng.normal(0, 1)
                if rng.random() < SIG(theta[p] - beta):
                    continue  # correct: no misconception evidence (nested logit)
                offer = offered_codes(rng, avail)
                probs = np.exp(eta_p[offer]); probs /= probs.sum()
                chosen = int(offer[rng.choice(N_OFFERED, p=probs)])
                picks[chosen] += 1
                offers[offer] += 1
                history.append((offer, chosen))
                profile.record_choice("s", str(chosen), [str(c) for c in offer])

                if phase == len(phases) - 1 and wrong in scored:
                    t = truth[p]
                    hits["skill only"][wrong] += fixed_guess == t
                    hits["count"][wrong] += int(np.argmax(picks)) == t
                    rate = np.divide(picks, offers, out=np.zeros_like(picks), where=offers > 0)
                    hits["rate"][wrong] += int(np.argmax(rate)) == t
                    hits["batch choice model"][wrong] += int(np.argmax(batch_choice_fit(history))) == t
                    hits["online choice model"][wrong] += int(profile.dominant("s")) == t
                wrong += 1
    return {name: (v / n_learners).tolist() for name, v in hits.items()}


def summarize(runs: list[dict], checkpoints) -> dict:
    out = {}
    for name in runs[0]:
        arr = np.array([r[name] for r in runs])  # reps x n_wrong
        out[name] = {}
        for c in checkpoints:
            col = arr[:, c - 1]
            half = 1.96 * col.std(ddof=1) / np.sqrt(len(col)) if len(col) > 1 else 0.0
            out[name][c] = {"mean": float(col.mean()), "ci95": [float(col.mean() - half), float(col.mean() + half)]}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learners", type=int, default=300)
    parser.add_argument("--wrong", type=int, default=40)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--zipf", type=float, default=ZIPF)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--out", default="misconception_results.json")
    args = parser.parse_args()

    checkpoints = [c for c in (5, 10, 20, 40) if c <= args.wrong]
    chance = 1.0 / N_MISCONCEPTIONS
    results = {"settings": {"learners": args.learners, "wrong_answers_per_phase": args.wrong,
                            "replications": args.reps, "misconceptions": N_MISCONCEPTIONS,
                            "offered_per_item": N_OFFERED, "dominance": DOMINANCE, "zipf": args.zipf,
                            "learning_rate": args.learning_rate,
                            "chance": chance}}
    for cond, remediate in (("stable", False), ("remediated", True)):
        runs = [run(1000 + r, args.learners, args.wrong, remediate, args.zipf, args.learning_rate, checkpoints) for r in range(args.reps)]
        results[cond] = summarize(runs, checkpoints)
        label = "after the switch" if remediate else ""
        print(f"\n=== {cond}: top-1 accuracy (true dominant misconception), chance = {chance:.0%} ===")
        print(f"{'tracker':22s}" + "".join(f"{f'{c} wrong {label}'.strip():>22s}" for c in checkpoints))
        for name, by_c in results[cond].items():
            print(f"{name:22s}" + "".join(
                f"{by_c[c]['mean']:>10.1%} [{by_c[c]['ci95'][0]:.0%},{by_c[c]['ci95'][1]:.0%}]" for c in checkpoints))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten to {args.out}")
