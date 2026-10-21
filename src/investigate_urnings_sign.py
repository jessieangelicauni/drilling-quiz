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
both signs on the paper's setup (300 learners, 400 items, 120,000
responses). Nothing here changes the repo's simulations -- it measures.
"""
import numpy as np

import simulate_trackers as st
from urnings_tracker import UrnState, update


def run(theta, beta, n_resp, sign, seed, n_p=32, n_i=32, target=0.5):
    rng = np.random.default_rng(seed); st.RNG = rng
    P, I = len(theta), len(beta)
    kp = np.full(P, n_p // 2); ki = np.full(I, n_i // 2); eps = 1e-9
    served_p = []
    for t in range(n_resp):
        p = rng.integers(0, P)
        pi_p = np.clip(kp[p] / n_p, eps, 1 - eps)
        beta_hat = sign * st._logit_urn(ki, n_i)
        i = st.pick_adaptive(st._logit_urn(kp[p], n_p), beta_hat, target)
        pc = st.SIG(theta[p] - beta[i]); served_p.append(pc)
        x = int(rng.random() < pc)
        pi_i = np.clip(ki[i] / n_i, eps, 1 - eps)
        num = pi_p * (1 - pi_i); q = num / (num + (1 - pi_p) * pi_i); y = int(rng.random() < q)
        kp_s, ki_s = kp[p] + x - y, ki[i] - x + y
        if 0 <= kp_s <= n_p and 0 <= ki_s <= n_i:
            cur = kp[p] * (n_i - ki[i]) + (n_p - kp[p]) * ki[i]
            pro = kp_s * (n_i - ki_s) + (n_p - kp_s) * ki_s
            if cur <= 0 or rng.random() < min(1.0, pro / cur):
                kp[p], ki[i] = kp_s, ki_s
    est = st._logit_urn(kp, n_p); sp = np.array(served_p[-20000:])
    return est.std(), np.corrcoef(est, theta)[0, 1], np.sqrt(np.mean((est - theta) ** 2)), \
        np.corrcoef(st._logit_urn(ki, n_i), beta)[0, 1], sp.mean(), np.mean(np.abs(sp - 0.5))

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
    for name, sign in (("repo (beta_hat=-logit ki)", -1), ("corrected (+logit ki)", +1)):
        sd, r, rmse, rb, ps, dev = run(theta, beta, 120_000, sign, 7)
        print(f"{name:28s} SD={sd:.2f} r={r:.3f} RMSE={rmse:.3f} corr(item,beta)={rb:+.2f} "
              f"mean P(correct) served={ps:.2f} mean|P-0.5|={dev:.2f}")
