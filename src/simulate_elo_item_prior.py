"""
Fixed-pool companion to simulate_item_prior.py, for Elo/Parallel Elo
instead of Urnings -- built to get an apples-to-apples comparison under
IDENTICAL conditions (same 400-item/120,000-response fixed pool as
simulate_trackers.py, same prior-quality grid as simulate_item_prior.py),
which did not exist before: simulate_item_coldstart_elo.py already
compares Elo/Parallel Elo under prior injection, but in the GROWING-pool
regime (PRD §9.13), a different condition with a different neutral
baseline -- mixing its numbers with simulate_item_prior.py's fixed-pool
Urnings numbers would compare two different experiments, not one.

This answers one question precisely: under the SAME fixed-pool regime,
does Urnings' robustness to prior quality (simulate_item_prior.py: RMSE@
final stays in a 0.42-0.44 band across correlation 0.47-0.96) hold for
Elo and Parallel Elo too, or is it Urnings-specific? Reuses
simulate_trackers.RNG/SIG/pick_adaptive and simulate_item_prior's prior
generator directly, not reimplemented.
"""
import numpy as np

from simulate_trackers import RNG, SIG, pick_adaptive
from simulate_item_prior import seed_items_from_encoder, _rmse_at

N_REPS = 8


def run_elo_item_prior(theta: np.ndarray, beta: np.ndarray, n_resp: int,
                        bh0: np.ndarray, K: float = 0.3, target: float = 0.5,
                        parallel: bool = False) -> np.ndarray:
    """Identical to simulate_trackers.run_elo, except bh starts from the
    caller-supplied prior bh0 instead of always starting at zero -- the
    ONLY change, so the comparison isolates the effect of prior quality."""
    P, I = len(theta), len(beta)
    th = np.zeros((2, P)) if parallel else np.zeros((1, P))
    bh = bh0.copy()
    track = []
    for t in range(n_resp):
        p = RNG.integers(0, P)
        if parallel:
            c = t % 2
            theta_sel = th[1 - c, p]
        else:
            c, theta_sel = 0, th[0, p]
        i = pick_adaptive(theta_sel, bh, target)
        x = RNG.random() < SIG(theta[p] - beta[i])
        exp = SIG(th[c, p] - bh[i])
        th[c, p] += K * (x - exp)
        bh[i] -= K * (x - exp)
        if (t + 1) % 200 == 0:
            est = th.mean(axis=0)
            track.append((t + 1, est.std(), np.corrcoef(est, theta)[0, 1],
                          np.sqrt(np.mean((est - theta) ** 2))))
    return np.array(track)


if __name__ == "__main__":
    P, I, N = 300, 400, 120_000
    conditions = [("neutral", None)] + [
        (f"noise_std={ns}", ns) for ns in (0.3, 0.8, 2.0)
    ]

    all_results: dict[str, dict[str, list]] = {"Elo": {}, "Parallel Elo": {}}
    for label, noise_std in conditions:
        for rep in range(N_REPS):
            theta = RNG.normal(0, 1, P)
            beta = RNG.normal(0, 1, I)
            if noise_std is None:
                bh0, corr = np.zeros(I), None
            else:
                mu_hat = beta + RNG.normal(0, noise_std, size=I)
                bh0 = mu_hat
                corr = float(np.corrcoef(mu_hat, beta)[0, 1])

            for name, parallel in (("Elo", False), ("Parallel Elo", True)):
                track = run_elo_item_prior(theta, beta, N, bh0, parallel=parallel)
                all_results[name].setdefault(label, {"corr": [], "rmse_final": []})
                all_results[name][label]["rmse_final"].append(track[-1, 3])
                if corr is not None:
                    all_results[name][label]["corr"].append(corr)
        print(f"condition {label} done")

    print(f"\n{'tracker':14s} {'condition':14s} {'avg corr':>9s} {'RMSE@final':>12s}")
    for name, by_cond in all_results.items():
        for label, vals in by_cond.items():
            corr = np.mean(vals["corr"]) if vals["corr"] else float("nan")
            rmse = np.mean(vals["rmse_final"])
            rmse_sd = np.std(vals["rmse_final"])
            print(f"{name:14s} {label:14s} {corr:9.2f} {rmse:8.4f}+/-{rmse_sd:.4f}")
