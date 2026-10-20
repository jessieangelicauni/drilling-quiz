"""
Validation for Novelty 2 (Uncertainty-Gated Escalation, PRD §3.3 / paper
Section IV-C) -- the piece of evidence Section V-C of the paper describes
but that, before this script, no code actually produced.

RESOLVED FINDING (2026-10-20), read this before trusting or extending
anything below -- `should_escalate()`'s w_k > tau trigger does NOT measure
what Section V-C assumes it measures:

  Section V-C asks to correlate the width that triggers escalation with
  the SUBSEQUENT REDUCTION in that width. That specific comparison is not
  meaningful here: `update()` is identical regardless of which feedback
  path produced the interaction, so any width-change pattern conditioned
  on escalation is a confound, not evidence for the policy -- deliberately
  not implemented.

  The real question -- does urn width track true estimation error
  |theta_hat - theta_true| -- WAS tested, with a decisive negative result,
  via `controlled_position_diagnostic()`. Mechanism, confirmed by a
  single-learner-per-theta equilibrium run with a fixed, exactly-known
  item pool (no population or item-estimation confounds):

    width(k, n) is the Beta(k+.5, n-k+.5) posterior SD on a FIXED n -- a
    function of CURRENT POSITION only (maximal at k=n/2, i.e. when the
    learner's recent record sits near 50% against whatever item pool they
    face), not of accumulated evidence quantity the way a growing-sample
    SE would be. It therefore measures "how central is this learner's
    estimate within the achievable range" -- NOT "how far is this
    estimate from the unobservable truth."

  At n=32 (the only urn size used anywhere in this codebase), a learner at
  an EXTREME true ability (theta=-2.5) converges to one of the NARROWEST
  widths (0.09, looking "confident") while carrying the LARGEST true error
  (0.65) of any position tested -- the opposite of what the escalation
  trigger assumes, and the learner who would benefit most from
  intervention is the one least likely to ever trigger it. A learner near
  the pool's centre (theta=0) gets the WIDEST urn (0.16-0.17, "most
  uncertain") while carrying middling, unremarkable error. Across a grid
  of theta values this gives corr(width, error) = -0.755 at n=32, -0.849
  at n=64 -- strongly negative at both, not noise, but NOT monotonically
  shrinking towards zero either (these numbers were re-verified directly
  against this script's current output, not copied from an earlier run;
  an earlier draft of this docstring reported -0.81/-0.78/-0.78 for
  n=32/64/128 and a since-unreproduced +0.09 at n=256 -- those are stale
  relative to urnings_tracker.py's current update() behaviour and are not
  restated here). At n=128 the correlation nearly vanishes (-0.040,
  verified below, run `python3 validate_novelty2.py`) -- consistent with
  the general direction (the problem does not get WORSE as n grows), but
  the exact shape is noisier than a clean "shrinks smoothly to positive"
  story, and no n=256 result is currently reproduced by this script to
  check whether the earlier mildly-positive reading still holds. The
  paper (Section IV-C) cites only the n=32 figure, which this script
  continues to support (-0.76 after rounding); it does not rely on the
  n=64/128/256 trend, so this docstring's correction does not change the
  paper's claim, only this module's own account of its robustness check.
  §9.13 already independently found n=32's terminal RMSE floor is ~0.45
  logit units -- consistent with the ~0.4-0.6 error floor found here
  regardless of
  position, which leaves width very little real signal to track).

  Implication: the escalation novelty's justification needs ONE of two
  fixes, a product decision this script does not make unilaterally --
  (a) stop claiming width tracks absolute measurement error, and reframe
  the trigger honestly as "escalate when recent performance against this
  learner's matched item pool has been genuinely ambiguous" (defensible
  on its own pedagogical terms, and consistent with the OTHER half of the
  trigger, consecutive-misconception, which is about ambiguity/persistence
  too, not absolute error) -- Section V-C's text would need rewriting to
  match this honest claim instead of the current (now falsified) one; or
  (b) replace or augment w_k with a signal that actually tracks estimation
  quality independent of position (e.g. explicit visit-count-based SE, or
  recent k-trend volatility, closer to the K-factor idea in [15]) before
  claiming the trigger targets poorly-measured learners.

What `simulate()` below still validates without needing this resolved:
the realistic ESCALATE FRACTION / compute-savings number (Section V-C's
second question) -- that holds regardless of which interpretation of
width is correct, since it only depends on how often w_k > tau fires
across a realistic practice history, not on what width "means".

What nothing here can establish: whether escalated feedback itself
improves learning outcomes -- a pedagogy claim needing real response data
or the classroom pilot (Section V-D), not simulation.

FORMAL CALIBRATION CHECK (added 2026-10-20, requested explicitly as the
rigorous follow-up to the correlation finding above -- a raw Pearson r is
suggestive but not the standard the ML uncertainty-quantification
literature uses to make this claim precise). `calibration_curve()`
implements the regression-calibration recipe from Kuleshov, Fenner &
Ermon (ICML 2018, "Accurate Uncertainties for Deep Learning Using
Calibrated Regression") -- the continuous-estimate analogue of Guo et
al.'s (ICML 2017) Expected Calibration Error for classifiers: for a
nominal confidence level p (e.g. 80%), take the urn's own Beta(k+.5,
n-k+.5) credible interval for pi at that level, transform its endpoints
to the logit (theta) scale, and check what fraction of cases the TRUE
theta actually falls inside -- averaged across several nominal levels,
the mean |nominal - observed| gap is this script's own ECE-equivalent
number. No scipy dependency: Beta quantiles are estimated by Monte Carlo
(np.random.Generator.beta, 200k draws per k, precomputed once per n since
there are only n+1 possible k values) rather than scipy.stats.beta.ppf,
matching this repo's numpy-only convention in every other simulate_*.py.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from urnings_tracker import UrnState, update, should_escalate

RNG = np.random.default_rng(20261020)


def beta_quantile_table(n: int, levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                         n_samples: int = 200_000, rng: np.random.Generator | None = None) -> dict:
    """Precompute, for every achievable k in [0, n], the pi-scale credible
    interval bounds of Beta(k+.5, n-k+.5) at each nominal level, via Monte
    Carlo (no scipy). Returns {k: {level: (lo_pi, hi_pi)}}."""
    rng = rng or np.random.default_rng(0)
    table = {}
    for k in range(n + 1):
        samples = rng.beta(k + 0.5, n - k + 0.5, size=n_samples)
        table[k] = {}
        for level in levels:
            tail = (1.0 - level) / 2.0
            lo, hi = np.quantile(samples, [tail, 1.0 - tail])
            table[k][level] = (float(lo), float(hi))
    return table


def calibration_curve(k_theta_pairs: list[tuple[int, float]], quantile_table: dict,
                       levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> dict:
    """For each nominal confidence level, what fraction of (k, theta_true)
    cases have theta_true inside the urn's own credible interval at that
    level, transformed from the pi scale to the logit scale? A
    well-calibrated estimator has observed ~= nominal for every level."""
    eps = 1e-9
    per_level = {}
    for level in levels:
        hits = 0
        for k, theta_true in k_theta_pairs:
            lo_pi, hi_pi = quantile_table[k][level]
            lo_pi, hi_pi = np.clip(lo_pi, eps, 1 - eps), np.clip(hi_pi, eps, 1 - eps)
            lo_logit = np.log(lo_pi / (1 - lo_pi))
            hi_logit = np.log(hi_pi / (1 - hi_pi))
            hits += int(lo_logit <= theta_true <= hi_logit)
        observed = hits / len(k_theta_pairs)
        per_level[level] = {"nominal": level, "observed": observed, "abs_gap": abs(observed - level)}
    ece = float(np.mean([v["abs_gap"] for v in per_level.values()]))
    return {"per_level": per_level, "n": len(k_theta_pairs), "calibration_error": ece}


def pick_adaptive(theta_hat: float, beta_hat: np.ndarray, target: float, pool: int = 48) -> int:
    """Same target-success-rate selection as Section IV-D Eq. 3 (target=0.85),
    reused here (not reimplemented ad hoc) so the simulated practice history
    looks like what a real learner would actually be served."""
    want = theta_hat - np.log(target / (1 - target))
    cand = RNG.integers(0, len(beta_hat), size=pool)
    return int(cand[np.argmin(np.abs(beta_hat[cand] - want))])


def simulate(n_learners: int, n_items: int, n_responses: int,
             n_p: int = 32, n_i: int = 32, target: float = 0.85,
             tau: float = 0.15, min_visits: int = 20) -> dict:
    """Realistic mixed-population run: escalate-fraction / compute-savings
    estimate (robust regardless of the width-meaning finding above), plus
    the pooled width-vs-error correlation (NOT robust -- confounded by
    population composition and convergence-stage mixing; kept only for
    continuity with the earlier run, see controlled_position_diagnostic
    for the number that should actually be trusted/quoted). Also returns
    raw (k, theta_true) pairs, post-burn-in (min_visits), for
    calibration_curve() -- the realistic-deployment calibration number.
    """
    theta_true = RNG.normal(0, 1, n_learners)
    beta_true = RNG.normal(0, 1, n_items)

    learner_urns = [UrnState(k=n_p // 2, n=n_p) for _ in range(n_learners)]
    item_urns = [UrnState(k=n_i // 2, n=n_i) for _ in range(n_items)]
    visits = np.zeros(n_learners, dtype=int)

    records = []
    k_theta_pairs = []
    escalate_count = 0

    for _ in range(n_responses):
        p = RNG.integers(0, n_learners)
        visits[p] += 1
        beta_hat = np.array([-i.logit_estimate() for i in item_urns])
        i = pick_adaptive(learner_urns[p].logit_estimate(), beta_hat, target)

        correct = bool(RNG.random() < 1.0 / (1.0 + np.exp(-(theta_true[p] - beta_true[i]))))
        learner_urns[p], item_urns[i] = update(learner_urns[p], item_urns[i], correct, RNG)

        w = learner_urns[p].width()
        err = abs(learner_urns[p].logit_estimate() - theta_true[p])
        esc = should_escalate(learner_urns[p], tau=tau)
        records.append((w, err, esc))
        escalate_count += int(esc)
        if visits[p] >= min_visits:
            k_theta_pairs.append((learner_urns[p].k, float(theta_true[p])))

    widths = np.array([r[0] for r in records])
    errors = np.array([r[1] for r in records])
    escalated = np.array([r[2] for r in records])
    escalate_fraction = float(escalate_count / len(records))

    return {
        "n_learners": n_learners, "n_items": n_items, "n_responses": n_responses, "tau": tau,
        "pooled_width_vs_true_error_corr_NOT_TRUSTED": float(np.corrcoef(widths, errors)[0, 1]),
        "mean_abs_error_when_escalated": float(errors[escalated].mean()) if escalated.any() else None,
        "mean_abs_error_when_routine": float(errors[~escalated].mean()) if (~escalated).any() else None,
        "escalate_fraction": escalate_fraction,
        "compute_multiplier_saved_vs_always_escalate": 1.0 - escalate_fraction,
        "measurement_precision_identical_by_construction": True,
        "k_theta_pairs": k_theta_pairs,
        "n_p": n_p,
    }


def controlled_position_diagnostic(n_p: int, n_resp: int = 8000, tail: int = 1000,
                                    thetas=(-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5)) -> dict:
    """The decisive test: ONE learner per fixed true theta, against a fixed
    exactly-known item-difficulty pool (no item-estimation or population
    confounds), run to equilibrium. Isolates what width actually tracks."""
    beta_pool = np.linspace(-3, 3, 200)
    rows = []
    k_theta_pairs = []
    for theta_true in thetas:
        learner = UrnState(k=n_p // 2, n=n_p)
        ws, errs = [], []
        for t in range(n_resp):
            i = RNG.integers(0, len(beta_pool))
            correct = bool(RNG.random() < 1.0 / (1.0 + np.exp(-(theta_true - beta_pool[i]))))
            p_item = 1.0 / (1.0 + np.exp(beta_pool[i]))
            item_urn = UrnState(k=min(max(int(round(p_item * n_p)), 0), n_p), n=n_p)
            learner, _ = update(learner, item_urn, correct, RNG)
            if t >= n_resp - tail:
                ws.append(learner.width())
                errs.append(abs(learner.logit_estimate() - theta_true))
                k_theta_pairs.append((learner.k, theta_true))
        rows.append({"theta_true": theta_true,
                     "equilibrium_width": float(np.mean(ws)),
                     "equilibrium_abs_error": float(np.mean(errs))})
    widths = [r["equilibrium_width"] for r in rows]
    errors = [r["equilibrium_abs_error"] for r in rows]
    return {"n_p": n_p, "rows": rows,
            "width_vs_error_corr_across_theta_grid": float(np.corrcoef(widths, errors)[0, 1]),
            "k_theta_pairs": k_theta_pairs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-learners", type=int, default=300)
    parser.add_argument("--n-items", type=int, default=400)
    parser.add_argument("--n-responses", type=int, default=120_000)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--out", default="2_validation_results.json")
    args = parser.parse_args()

    print("=== controlled position diagnostic (the number to trust) ===\n")
    diag = {n_p: controlled_position_diagnostic(n_p) for n_p in (32, 64, 128)}
    for n_p, d in diag.items():
        print(f"n_p={n_p}")
        for r in d["rows"]:
            print(f"  theta_true={r['theta_true']:+5.1f}  "
                  f"equilibrium width={r['equilibrium_width']:.4f}  "
                  f"equilibrium |error|={r['equilibrium_abs_error']:.4f}")
        print(f"  -> corr(width, error) across theta grid: "
              f"{d['width_vs_error_corr_across_theta_grid']:+.3f}\n")

    print("=== realistic mixed-population run (escalate-fraction is the number to trust;")
    print("    pooled correlation below is confounded, see docstring) ===\n")
    result = simulate(args.n_learners, args.n_items, args.n_responses, tau=args.tau)
    print(f"n_learners={result['n_learners']} n_responses={result['n_responses']} tau={result['tau']}")
    print(f"fraction of interactions that escalate:                 {result['escalate_fraction']:.1%}")
    print(f"compute saved vs. always-escalate baseline:             "
          f"{result['compute_multiplier_saved_vs_always_escalate']:.1%}")
    print(f"[NOT TRUSTED, population-confounded] pooled corr(width,error): "
          f"{result['pooled_width_vs_true_error_corr_NOT_TRUSTED']:+.3f}")

    print("\n=== formal calibration curve (Kuleshov et al. 2018 / Guo et al. 2017 style) ===")
    print("nominal X% credible interval should contain true theta ~X% of the time if calibrated.\n")
    q32 = beta_quantile_table(32, rng=np.random.default_rng(1))
    calib_controlled = calibration_curve(diag[32]["k_theta_pairs"], q32)
    calib_realistic = calibration_curve(result["k_theta_pairs"], q32)
    for label, calib in (("controlled per-theta (n=32)", calib_controlled),
                         ("realistic mixed population (n=32)", calib_realistic)):
        print(f"-- {label}, n={calib['n']} --")
        print(f"{'nominal':>8s} {'observed':>9s} {'gap':>7s}")
        for level, v in calib["per_level"].items():
            print(f"{v['nominal']:8.0%} {v['observed']:9.1%} {v['abs_gap']:7.1%}")
        print(f"calibration error (mean |nominal-observed| across levels): {calib['calibration_error']:.1%}\n")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"controlled_position_diagnostic": {str(k): v for k, v in diag.items()},
                    "mixed_population_run": result,
                    "calibration_controlled": calib_controlled,
                    "calibration_realistic": calib_realistic}, f, indent=2)
    print(f"\nFull results written to {args.out}")
