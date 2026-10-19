"""
Novelty 1 evidence, take 2 (PRD §3.1, §9.13): simulate_item_prior.py's
fixed-pool regime (400 items hammered by 120,000 responses, ~300
responses/item by the end) showed NO benefit from an encoder-informed item
prior -- and a plausible reason why: with that much exposure per item,
real response data calibrates every item quickly regardless of its
starting point, so the prior only matters briefly before being overwritten.

That isn't the regime Novelty 1 is actually meant to help with. This
app's real item pool is GENERATIVE and constantly growing (exaone_provider
+ question_bank.py) -- most items, most of the time, are relatively new
and have had few responses. This simulates THAT regime instead: a pool
that starts small and grows by one freshly-generated item every
`intro_interval` responses, so at any point a large fraction of available
items are still "new". If an informed prior matters anywhere, it should
show up here, not in simulate_item_prior.py's regime.

Same ground-truth model (Rasch) and ball-exchange math as
simulate_item_prior.py/simulate_trackers.py -- reuses their primitives
directly rather than redefining them.
"""
import numpy as np

from simulate_item_prior import _rmse_at
from simulate_trackers import RNG, SIG, _logit_urn, pick_adaptive


def neutral_seed(beta_i: float) -> tuple[int, int]:
    return 16, 32


def encoder_seed_fn(noise_std: float):
    """Position-only encoder prior (n fixed at 32, matching the baseline --
    see simulate_item_prior.py's §9.12 note on why mixing in a different n
    confounds the comparison)."""
    def fn(beta_i: float) -> tuple[int, int]:
        mu_hat = beta_i + RNG.normal(0, noise_std)
        pi = SIG(-mu_hat)
        k = int(np.clip(round(pi * 33 - 0.5), 0, 32))
        return k, 32
    return fn


def run_urnings_growing_pool(theta: np.ndarray, n_resp: int, item_k0_fn,
                              I0: int = 20, intro_interval: int = 20,
                              n_p: int = 32, target: float = 0.5) -> np.ndarray:
    """Items are introduced one at a time (fresh true beta ~ N(0,1), seeded
    via item_k0_fn) every `intro_interval` responses, instead of all
    existing from t=0 -- the actual generative-pool regime, not a fixed
    pool. pick_adaptive only ever sees already-introduced items.
    """
    P = len(theta)
    I_max = I0 + n_resp // intro_interval + 1
    beta = np.empty(I_max)
    ki = np.empty(I_max, dtype=int)
    ni = np.empty(I_max, dtype=int)
    kp = np.full(P, n_p // 2)

    n_active = I0
    for i in range(I0):
        beta[i] = RNG.normal(0, 1)
        ki[i], ni[i] = item_k0_fn(beta[i])

    track = []
    eps = 1e-9
    next_intro = intro_interval
    for t in range(n_resp):
        if n_active < I_max and (t + 1) >= next_intro:
            beta[n_active] = RNG.normal(0, 1)
            ki[n_active], ni[n_active] = item_k0_fn(beta[n_active])
            n_active += 1
            next_intro += intro_interval

        p = RNG.integers(0, P)
        pi_p = np.clip(kp[p] / n_p, eps, 1 - eps)
        theta_sel = _logit_urn(kp[p], n_p)
        beta_hat = -_logit_urn(ki[:n_active], ni[:n_active])
        i = pick_adaptive(theta_sel, beta_hat, target, pool=min(48, n_active))

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
                          np.sqrt(np.mean((est - theta) ** 2)), n_active))
    return np.array(track)


if __name__ == "__main__":
    P, N = 300, 120_000
    I0, INTRO_INTERVAL = 20, 20   # ~6000 items introduced over the run, ~20 responses/item on average
    N_REPS = 8

    conditions = {
        "neutral prior (every new item starts flat)": neutral_seed,
        "encoder-informed, strong signal (noise_std=0.3)": encoder_seed_fn(0.3),
        "encoder-informed, moderate signal (noise_std=0.8)": encoder_seed_fn(0.8),
        "encoder-informed, weak signal (noise_std=2.0)": encoder_seed_fn(2.0),
    }

    all_results: dict[str, list[tuple[float, float]]] = {}
    for rep in range(N_REPS):
        theta = RNG.normal(0, 1, P)
        for label, seed_fn in conditions.items():
            track = run_urnings_growing_pool(theta, N, seed_fn, I0=I0, intro_interval=INTRO_INTERVAL)
            all_results.setdefault(label, []).append((_rmse_at(track, 2000), track[-1, 3]))
        print(f"replication {rep + 1}/{N_REPS} done")

    print(f"\nFinal pool size: ~{I0 + N // INTRO_INTERVAL} items, "
          f"~{N / (I0 + N // INTRO_INTERVAL):.1f} responses/item on average\n")
    print(f"{'condition':52s} {'RMSE@2000':>18s} {'RMSE@final':>18s}")
    for label, vals in all_results.items():
        early = np.array([v[0] for v in vals])
        final = np.array([v[1] for v in vals])
        print(f"{label:52s} {early.mean():7.4f} +/- {early.std():5.4f}   "
              f"{final.mean():7.4f} +/- {final.std():5.4f}")
