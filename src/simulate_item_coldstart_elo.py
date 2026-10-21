"""
Novelty 1 evidence, take 3 (PRD §3.1, §9.13) -- user's hypothesis: Urnings'
ball-exchange math showed a counter-intuitive per-step update direction
(simulate_item_prior.py's finding: a correct answer on an EASIER-seeded
item moves the learner's urn up MORE than on a harder one) -- maybe THAT
specific dynamic, not the idea of prior injection itself, is why
simulate_item_coldstart.py found no benefit from an encoder-informed item
prior under Urnings. Elo's update is much more directly tied to the
predicted probability (th += K*(x - sigmoid(theta-beta))), so if an
informed prior helps anywhere, it should show up more transparently here.

Tests this directly: same growing-pool regime as simulate_item_coldstart.py
(generative pool, most items new most of the time), same ground-truth
Rasch model, but tracked with plain Elo and Parallel Elo (both already
validated/compared in simulate_trackers.py) instead of Urnings. Item prior
injection is also simpler for Elo -- bh0 can be set directly to the
predicted beta, no urn/sigmoid/rounding transform needed, unlike
Urnings' seed_item_from_encoder().

This is a DIAGNOSTIC comparison to understand why Urnings behaved as it
did, not a proposal to replace the production tracker -- Urnings was kept
over Parallel Elo for reasons independent of this (PRD §2.1/§8: Novelty 1
AND Novelty 3 both built on the urn-width formalism specifically, and
Elo's own precision/drift problems under adaptive selection are already
documented in simulate_trackers.py/simulate_drift.py). Reconsidering the
tracker itself would need to re-litigate those results too, not just this
one experiment -- kept separate on purpose.
"""
import numpy as np

from simulate_item_prior import _rmse_at
from simulate_trackers import RNG, SIG, pick_adaptive


def neutral_bh0(beta_i: float) -> float:
    return 0.0


def encoder_bh0_fn(noise_std: float):
    def fn(beta_i: float) -> float:
        return beta_i + RNG.normal(0, noise_std)
    return fn


def run_elo_growing_pool(theta: np.ndarray, n_resp: int, item_bh0_fn,
                          I0: int = 20, intro_interval: int = 20,
                          K: float = 0.3, target: float = 0.5,
                          parallel: bool = False) -> np.ndarray:
    P = len(theta)
    I_max = I0 + n_resp // intro_interval + 1
    beta = np.empty(I_max)
    bh = np.empty(I_max)
    nchains = 2 if parallel else 1
    th = np.zeros((nchains, P))

    n_active = I0
    for i in range(I0):
        beta[i] = RNG.normal(0, 1)
        bh[i] = item_bh0_fn(beta[i])

    track = []
    next_intro = intro_interval
    for t in range(n_resp):
        if n_active < I_max and (t + 1) >= next_intro:
            beta[n_active] = RNG.normal(0, 1)
            bh[n_active] = item_bh0_fn(beta[n_active])
            n_active += 1
            next_intro += intro_interval

        p = RNG.integers(0, P)
        if parallel:
            c = t % 2
            theta_sel = th[1 - c, p]
        else:
            c, theta_sel = 0, th[0, p]
        i = pick_adaptive(theta_sel, bh[:n_active], target)

        x = RNG.random() < SIG(theta[p] - beta[i])
        exp = SIG(th[c, p] - bh[i])
        th[c, p] += K * (x - exp)
        bh[i] -= K * (x - exp)

        if (t + 1) % 200 == 0:
            est = th.mean(axis=0)
            track.append((t + 1, est.std(), np.corrcoef(est, theta)[0, 1],
                          np.sqrt(np.mean((est - theta) ** 2)), n_active))
    return np.array(track)


if __name__ == "__main__":
    P, N = 300, 120_000
    I0, INTRO_INTERVAL = 20, 20
    N_REPS = 8

    prior_conditions = {
        "neutral (bh0=0)": neutral_bh0,
        "encoder-informed, strong (noise_std=0.3)": encoder_bh0_fn(0.3),
        "encoder-informed, moderate (noise_std=0.8)": encoder_bh0_fn(0.8),
        "encoder-informed, weak (noise_std=2.0)": encoder_bh0_fn(2.0),
    }

    for tracker_label, parallel in [("Elo (single chain)", False), ("Parallel Elo", True)]:
        print(f"=== {tracker_label} ===")
        all_results: dict[str, list[tuple[float, float]]] = {}
        for rep in range(N_REPS):
            theta = RNG.normal(0, 1, P)
            for label, bh0_fn in prior_conditions.items():
                track = run_elo_growing_pool(theta, N, bh0_fn, I0=I0, intro_interval=INTRO_INTERVAL,
                                              parallel=parallel)
                all_results.setdefault(label, []).append((_rmse_at(track, 2000), track[-1, 3]))
            print(f"  replication {rep + 1}/{N_REPS} done")

        print(f"\n  {'condition':48s} {'RMSE@2000':>18s} {'RMSE@final':>18s}")
        for label, vals in all_results.items():
            early = np.array([v[0] for v in vals])
            final = np.array([v[1] for v in vals])
            print(f"  {label:48s} {early.mean():7.4f} +/- {early.std():5.4f}   "
                  f"{final.mean():7.4f} +/- {final.std():5.4f}")
        print()
