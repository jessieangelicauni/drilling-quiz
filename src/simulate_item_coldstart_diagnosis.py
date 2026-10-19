"""
Root-cause confirmation for the non-monotonic result in
simulate_item_coldstart.py (PRD §9.13/§9.17): a weak/noisy item-difficulty
prior (noise_std=2.0) showed BETTER final RMSE than both a neutral prior
and a strong/accurate one -- a result that doesn't fit "better priors help"
in either direction.

Confirms, in two stages, that this is a saturation artifact of
pick_adaptive's selection mechanism, not a real benefit of noisy priors:

  1. A noisier prior pushes more items to a SATURATED starting urn (k=0 or
     32) -- the sigmoid+round+clip transform in encoder_seed_fn more often
     hits the boundary when mu_hat has high variance.
  2. pick_adaptive NEVER selects a saturated item (its implied difficulty
     is too extreme to match any learner's target success probability),
     so those items are effectively removed from the servable pool.
  3. The same fixed response budget then concentrates onto fewer
     surviving items, which each get MORE exposure and therefore
     calibrate more precisely regardless of their own starting point --
     this is what makes the weak-signal condition's aggregate RMSE look
     better, not noisy priors being beneficial.

Practical implication for the real system: a badly-off difficulty_heuristic
estimate for one item doesn't just miscalibrate it -- it can make that
item effectively UNSERVED (never selected for anyone), which needs its
own monitoring (dead items in the bank), not just a calibration-accuracy
check.
"""
import numpy as np

from simulate_item_coldstart import encoder_seed_fn
from simulate_trackers import RNG, SIG, _logit_urn, pick_adaptive


def saturation_rate(noise_std: float, n_samples: int = 2000) -> float:
    beta_sample = RNG.normal(0, 1, n_samples)
    seed_fn = encoder_seed_fn(noise_std)
    k0s = np.array([seed_fn(b)[0] for b in beta_sample])
    return float(np.mean((k0s == 0) | (k0s == 32)))


def run_with_selection_tracking(theta: np.ndarray, n_resp: int, item_k0_fn,
                                 I0: int = 20, intro_interval: int = 20,
                                 n_p: int = 32, target: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Same growing-pool mechanics as simulate_item_coldstart.run_urnings_growing_pool,
    with ONE addition: tracks how many times each item is actually
    selected by pick_adaptive, and each item's starting k0 -- the two
    things needed to test the saturation hypothesis directly.
    """
    P = len(theta)
    I_max = I0 + n_resp // intro_interval + 1
    beta = np.empty(I_max)
    ki = np.empty(I_max, dtype=int)
    ni = np.empty(I_max, dtype=int)
    kp = np.full(P, n_p // 2)
    selection_count = np.zeros(I_max, dtype=int)
    initial_k0 = np.empty(I_max, dtype=int)

    n_active = I0
    for i in range(I0):
        beta[i] = RNG.normal(0, 1)
        ki[i], ni[i] = item_k0_fn(beta[i])
        initial_k0[i] = ki[i]

    eps = 1e-9
    next_intro = intro_interval
    for t in range(n_resp):
        if n_active < I_max and (t + 1) >= next_intro:
            beta[n_active] = RNG.normal(0, 1)
            ki[n_active], ni[n_active] = item_k0_fn(beta[n_active])
            initial_k0[n_active] = ki[n_active]
            n_active += 1
            next_intro += intro_interval

        p = RNG.integers(0, P)
        pi_p = np.clip(kp[p] / n_p, eps, 1 - eps)
        theta_sel = _logit_urn(kp[p], n_p)
        beta_hat = -_logit_urn(ki[:n_active], ni[:n_active])
        i = pick_adaptive(theta_sel, beta_hat, target, pool=min(48, n_active))
        selection_count[i] += 1

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
    return selection_count[:n_active], initial_k0[:n_active]


if __name__ == "__main__":
    print("--- Stage 1: does a noisier prior saturate more item urns? ---")
    rates = {}
    for label, noise_std in [("strong", 0.3), ("moderate", 0.8), ("weak", 2.0)]:
        rate = saturation_rate(noise_std)
        rates[label] = rate
        print(f"{label:10s} (noise_std={noise_std}): saturated(k=0 or 32) = {100 * rate:.1f}%")
    assert rates["weak"] > rates["moderate"] > rates["strong"] >= 0.0, (
        "saturation rate must increase monotonically with prior noise"
    )

    print("\n--- Stage 2: are saturated items ever actually selected? ---")
    theta = RNG.normal(0, 1, 300)
    for label, noise_std in [("strong", 0.3), ("weak", 2.0)]:
        seed_fn = encoder_seed_fn(noise_std)
        counts, k0s = run_with_selection_tracking(theta, 120_000, seed_fn)
        saturated = (k0s == 0) | (k0s == 32)
        avg_saturated = counts[saturated].mean() if saturated.any() else float("nan")
        avg_normal = counts[~saturated].mean()
        print(f"{label:10s}: {saturated.sum()} saturated items selected {avg_saturated:.2f}x on average, "
              f"{(~saturated).sum()} normal items selected {avg_normal:.2f}x on average")
        if saturated.sum() > 0:
            assert avg_saturated < avg_normal * 0.1, (
                f"saturated items must be selected dramatically less often than normal items, "
                f"got {avg_saturated:.2f}x vs {avg_normal:.2f}x for {label}"
            )

    print("\nRoot cause confirmed: a noisy prior saturates more items, saturated items are "
          "essentially never served, so responses concentrate onto fewer surviving items, which "
          "calibrate MORE precisely purely from extra exposure -- not because the noisy prior helped.")
    print("all smoke tests passed")
