"""
Investigation (review 2026-10-21, PRD §9.68): is the Urnings item urn a
DIFFICULTY or an EASINESS, and what happens to the Section III-B
simulation when adaptive selection uses the sign the update kernel
actually implies?

The kernel (urnings_tracker.update / simulate_trackers.run_urnings) uses
P(correct) = pi_p (1 - pi_i) / (pi_p (1 - pi_i) + (1 - pi_p) pi_i), so a
HIGH item proportion pi_i means a HARD item: logit(pi_i) estimates +beta.
The selection code everywhere uses beta_hat = -logit(k_i), i.e. the
opposite. Part 1 checks the kernel's convention empirically (random item
selection, so selection can't bias it); part 2 reruns run_urnings with
the old sign, the fixed sign alone, and the fixed sign with the
adaptive-selection Metropolis-Hastings correction now in
simulate_trackers.urnings_step, on the paper's setup (300 learners, 400
items, 120,000 responses).
"""
import numpy as np

import simulate_trackers as st
from urnings_tracker import UrnState, update


def run_old_sign(theta, beta, n_resp, n_p=32, n_i=32, target=0.5):
    """run_urnings as it was before PRD §9.68: beta_hat = -logit(item urn),
    no adaptive-selection correction. Uses today's selection kernel, so it
    differs from the original run only in the sign."""
    P, I = len(theta), len(beta)
    kp = np.full(P, n_p // 2); ki = np.full(I, n_i // 2); eps = 1e-9
    for _ in range(n_resp):
        p = st.RNG.integers(0, P)
        i = st.pick_adaptive(st._logit_urn(kp[p], n_p), -st._logit_urn(ki, n_i), target)
        x = int(st.RNG.random() < st.SIG(theta[p] - beta[i]))
        pi_p = np.clip(kp[p] / n_p, eps, 1 - eps); pi_i = np.clip(ki[i] / n_i, eps, 1 - eps)
        num = pi_p * (1 - pi_i); q = num / (num + (1 - pi_p) * pi_i)
        y = int(st.RNG.random() < q)
        kp_s, ki_s = kp[p] + x - y, ki[i] - x + y
        if 0 <= kp_s <= n_p and 0 <= ki_s <= n_i:
            cur = kp[p] * (n_i - ki[i]) + (n_p - kp[p]) * ki[i]
            pro = kp_s * (n_i - ki_s) + (n_p - kp_s) * ki_s
            if cur <= 0 or st.RNG.random() < min(1.0, pro / cur):
                kp[p], ki[i] = kp_s, ki_s
    est = st._logit_urn(kp, n_p)
    return est.std(), np.corrcoef(est, theta)[0, 1], np.sqrt(np.mean((est - theta) ** 2))


def kernel_convention(seed: int = 1, P: int = 200, I: int = 40, n_resp: int = 60_000) -> float:
    """corr(logit(item k), true beta) under RANDOM selection."""
    rng = np.random.default_rng(seed)
    theta, beta = rng.normal(0, 1, P), rng.normal(0, 1, I)
    learners = [UrnState(16, 32) for _ in range(P)]
    items = [UrnState(16, 32) for _ in range(I)]
    for _ in range(n_resp):
        p, i = rng.integers(P), rng.integers(I)
        x = rng.random() < st.SIG(theta[p] - beta[i])
        learners[p], items[i] = update(learners[p], items[i], bool(x), rng)
    return float(np.corrcoef([u.logit_estimate() for u in items], beta)[0, 1])


if __name__ == "__main__":
    corr = kernel_convention()
    print(f"part 1: corr(logit item k, true beta) = {corr:+.2f} under random selection "
          f"({'item urn = DIFFICULTY' if corr > 0 else 'item urn = EASINESS'})")
    print("part 2: Section III-B setup, Urnings with each selection sign")
    rng = np.random.default_rng(20261018)
    theta = rng.normal(0, 1, 300); beta = rng.normal(0, 1, 400)
    print(f"true SD(theta) = {theta.std():.2f}")
    sd, r, rmse = run_old_sign(theta, beta, 120_000)
    print(f"{'old sign, no correction':34s} SD={sd:.2f} r={r:.3f} RMSE={rmse:.3f}")
    for name, corr in (("fixed sign, no correction", False), ("fixed sign + MH correction", True)):
        v = st.run_urnings(theta, beta, 120_000, correct_selection=corr)
        print(f"{name:34s} SD={v[-1, 1]:.2f} r={v[-1, 2]:.3f} RMSE={v[-1, 3]:.3f}")
