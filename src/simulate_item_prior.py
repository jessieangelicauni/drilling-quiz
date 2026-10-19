"""
Novelty 1 evidence (PRD §3.1, §9.12/§9.13): does seeding a new item's urn
from an encoder-style difficulty prediction make learner ability estimates
converge to the TRUE ability faster than starting every new item from a
flat, uninformative (neutral) prior -- the regime answer_processing.py was
actually in before §9.12 wired seed_item_from_encoder() in?

Same harness and ground-truth generating model (Rasch, theta/beta ~ N(0,1),
adaptive selection + simultaneous item calibration) as simulate_trackers.py
-- this is a follow-up to it, same pattern as simulate_drift.py, not a
rewrite: reuses RNG/SIG/pick_adaptive/_logit_urn directly.

difficulty_heuristic.py's own docstring is explicit that it is "not
calibrated to any real difficulty scale" -- so rather than claim one
specific noise level for it, this tests a RANGE of encoder quality
(correlation between the predicted mu_hat and true beta), reported
honestly. The question this answers is "does the MECHANISM help, and how
much prediction quality does it need before it's worth it" -- not "is our
current heuristic specifically good". PRD §3.1's own minimal evidence bar:
"encoder prior konvergen lebih cepat ke rating sebenarnya dibanding prior
seragam -- satu perbandingan, bukan grid ablasi besar." This runs that one
comparison at a few quality levels so the result is honest about
sensitivity, without turning into a big ablation.
"""
import numpy as np

from simulate_trackers import RNG, SIG, _logit_urn, pick_adaptive

# Matches urnings_tracker.seed_item_from_encoder()'s defaults exactly.
_N_MIN, _N_MAX, _REF_VAR = 8, 64, 1.0
# Matches difficulty_heuristic.py's ACTUAL behaviour: sigma_hat2 is a fixed
# constant (1.5), never inferred per-item -- so every encoder-seeded item
# below gets the SAME urn size; only k0 (predicted position) varies. An
# idealized varying-confidence encoder isn't what's shipped, so it isn't
# what's simulated here.
_FIXED_SIGMA2 = 1.5


def seed_items_from_encoder(beta_true: np.ndarray, noise_std: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulates an encoder whose prediction mu_hat is the item's true beta
    plus Gaussian error -- noise_std=0 is a perfect predictor, large
    noise_std makes mu_hat uninformative. Returns (k0, n0, mu_hat) so the
    caller can report the actual realised correlation, not an assumed one.
    """
    mu_hat = beta_true + RNG.normal(0, noise_std, size=len(beta_true))
    n = int(round(min(max(_N_MAX / (1.0 + _FIXED_SIGMA2 / _REF_VAR), _N_MIN), _N_MAX)))
    pi = SIG(-mu_hat)
    k = np.round(pi * (n + 1) - 0.5).astype(int)
    k = np.clip(k, 0, n)
    return k, np.full(len(beta_true), n), mu_hat


def run_urnings_item_prior(theta: np.ndarray, beta: np.ndarray, n_resp: int,
                            item_k0: np.ndarray, item_n0: np.ndarray,
                            n_p: int = 32, target: float = 0.5) -> np.ndarray:
    """Identical ball-exchange math to simulate_trackers.run_urnings, the
    ONLY difference being that items start from caller-supplied PER-ITEM
    (k0, n0) instead of always starting at a flat n_i//2 -- written as a
    separate function rather than modifying run_urnings, to leave that
    already-validated reference implementation untouched.
    """
    P, I = len(theta), len(beta)
    kp = np.full(P, n_p // 2)
    ki = item_k0.copy()
    ni = item_n0
    track = []
    eps = 1e-9
    for t in range(n_resp):
        p = RNG.integers(0, P)
        pi_p = np.clip(kp[p] / n_p, eps, 1 - eps)
        theta_sel = _logit_urn(kp[p], n_p)
        beta_hat = -_logit_urn(ki, ni)
        i = pick_adaptive(theta_sel, beta_hat, target)

        n_i = int(ni[i])
        x = int(RNG.random() < SIG(theta[p] - beta[i]))
        pi_i = np.clip(ki[i] / n_i, eps, 1 - eps)
        num = pi_p * (1 - pi_i)
        q = num / (num + (1 - pi_p) * pi_i)
        y = int(RNG.random() < q)

        kp_s, ki_s = kp[p] + x - y, ki[i] - x + y
        if 0 <= kp_s <= n_p and 0 <= ki_s <= n_i:
            cur = kp[p] * (n_i - ki[i]) + (n_p - kp[p]) * ki[i]
            pro = kp_s * (n_i - ki_s) + (n_p - kp_s) * ki_s
            if cur <= 0 or RNG.random() < min(1.0, pro / cur):
                kp[p], ki[i] = kp_s, ki_s

        if (t + 1) % 200 == 0:
            est = _logit_urn(kp, n_p)
            track.append((t + 1, est.std(), np.corrcoef(est, theta)[0, 1],
                          np.sqrt(np.mean((est - theta) ** 2))))
    return np.array(track)


def _rmse_at(track: np.ndarray, n: int) -> float:
    idx = np.searchsorted(track[:, 0], n)
    idx = min(idx, len(track) - 1)
    return float(track[idx, 3])


if __name__ == "__main__":
    P, I, N = 300, 400, 120_000
    N_REPS = 8  # independent replications (fresh theta/beta/response stream each time)

    def make_conditions(beta):
        # Labels are STABLE across replications (no realized value baked
        # in) so results actually aggregate into one group per condition --
        # the realized corr(mu_hat, beta) is tracked separately instead.
        I = len(beta)
        conds = {
            "neutral prior (pre-§9.12 baseline, n=32)": (np.full(I, 16), np.full(I, 32), None),
        }
        for label, noise_std in [("strong signal", 0.3), ("moderate signal", 0.8), ("weak signal", 2.0)]:
            k0, n0, mu_hat = seed_items_from_encoder(beta, noise_std)
            corr = np.corrcoef(mu_hat, beta)[0, 1]
            conds[f"encoder-informed, {label} (n=26)"] = (k0, n0, corr)
            # Isolate position (k0) from urn-size (n0): same informed k0, but
            # forced to n=32 like the baseline -- without this, "informed"
            # vs "neutral" also silently compares two different urn sizes
            # (n=26 from the fixed sigma_hat2=1.5 vs n=32), confounding
            # whether any effect is from the PRIOR POSITION or just a
            # smaller/less-sticky urn. Found empirically on the first run
            # of this script (RMSE got WORSE for every informed condition,
            # which made no sense until this confound was spotted).
            k0_n32 = np.round((k0 / n0) * (32 + 1) - 0.5).astype(int)
            k0_n32 = np.clip(k0_n32, 0, 32)
            conds[f"encoder-informed, {label}, n=32 (position only)"] = (k0_n32, np.full(I, 32), corr)
        return conds

    all_results: dict[str, list[tuple[float, float]]] = {}
    all_corrs: dict[str, list[float]] = {}
    for rep in range(N_REPS):
        theta = RNG.normal(0, 1, P)
        beta = RNG.normal(0, 1, I)
        for label, (k0, n0, corr) in make_conditions(beta).items():
            track = run_urnings_item_prior(theta, beta, N, k0, n0)
            all_results.setdefault(label, []).append((_rmse_at(track, 2000), track[-1, 3]))
            if corr is not None:
                all_corrs.setdefault(label, []).append(corr)
        print(f"replication {rep + 1}/{N_REPS} done")

    print(f"\n{'condition':52s} {'avg corr':>9s} {'RMSE@2000':>18s} {'RMSE@final':>18s}")
    for label, vals in all_results.items():
        early = np.array([v[0] for v in vals])
        final = np.array([v[1] for v in vals])
        corr_str = f"{np.mean(all_corrs[label]):.2f}" if label in all_corrs else "--"
        print(f"{label:52s} {corr_str:>9s} {early.mean():7.4f} +/- {early.std():5.4f}   "
              f"{final.mean():7.4f} +/- {final.std():5.4f}")
