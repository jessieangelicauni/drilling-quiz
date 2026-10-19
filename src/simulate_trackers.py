"""
Motivating simulation: rating-tracker behaviour under *simultaneous*
adaptive item selection and dynamic item calibration -- the exact regime a
generative (infinite) item pool forces on a drill system.

Reproduces the variance-inflation phenomenon reported by
Bolsinova, Gergely & Brinkhuis (2026), Br. J. Math. Stat. Psychol. 79(1):95-110.

Compares: (a) standard Elo, (b) Parallel Elo, (c) Urnings.
Ground-truth generating model is Rasch: P(correct) = sigmoid(theta - beta).
"""
import numpy as np

RNG = np.random.default_rng(20261018)
SIG = lambda z: 1.0 / (1.0 + np.exp(-z))


def pick_adaptive(theta_sel, beta_hat, target=0.5, pool=48):
    """Select item whose estimated difficulty puts success prob nearest target."""
    want = theta_sel - np.log(target / (1 - target))
    cand = RNG.integers(0, len(beta_hat), size=pool)
    return cand[np.argmin(np.abs(beta_hat[cand] - want))]


def run_elo(theta, beta, n_resp, K=0.3, target=0.5, parallel=False, update_items=True):
    P, I = len(theta), len(beta)
    if parallel:
        th = np.zeros((2, P))      # two independent chains
    else:
        th = np.zeros((1, P))
    bh = np.zeros(I)
    track = []
    for t in range(n_resp):
        p = RNG.integers(0, P)
        if parallel:
            c = t % 2                       # chain being updated
            theta_sel = th[1 - c, p]        # select using the OTHER chain
        else:
            c, theta_sel = 0, th[0, p]
        i = pick_adaptive(theta_sel, bh, target)
        x = RNG.random() < SIG(theta[p] - beta[i])
        exp = SIG(th[c, p] - bh[i])
        th[c, p] += K * (x - exp)
        if update_items:
            bh[i] -= K * (x - exp)
        if (t + 1) % 200 == 0:
            est = th.mean(axis=0)
            track.append((t + 1, est.std(), np.corrcoef(est, theta)[0, 1],
                          np.sqrt(np.mean((est - theta) ** 2))))
    return np.array(track)


def _logit_urn(k, n):
    """Urn proportion -> logit with a (k+.5)/(n+1) continuity correction.

    A raw k/n hits 0 or 1 whenever an urn saturates, which would send the
    logit to +/-inf; the correction keeps boundary urns finite and bounded.
    """
    pi = (k + 0.5) / (n + 1.0)
    return np.log(pi / (1.0 - pi))


def run_urnings(theta, beta, n_resp, n_p=32, n_i=32, target=0.5):
    """Urnings-1 (Bolsinova et al., 2022) with adaptive selection."""
    P, I = len(theta), len(beta)
    kp = np.full(P, n_p // 2)
    ki = np.full(I, n_i // 2)
    track = []
    eps = 1e-9
    for t in range(n_resp):
        p = RNG.integers(0, P)
        pi_p = np.clip(kp[p] / n_p, eps, 1 - eps)
        theta_sel = _logit_urn(kp[p], n_p)       # used only for item selection
        beta_hat = -_logit_urn(ki, n_i)
        i = pick_adaptive(theta_sel, beta_hat, target)

        x = int(RNG.random() < SIG(theta[p] - beta[i]))
        pi_i = np.clip(ki[i] / n_i, eps, 1 - eps)
        # simulate a response from the CURRENT urn state
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


def run_glicko(theta, beta, n_resp, target=0.5, parallel=False, update_items=True,
               rd0=1.0, rd_min=0.05):
    """Glicko-1-style tracker: continuous rating + native rating deviation (RD).

    RD plays the role Urnings' urn width plays -- a calibrated, per-entity
    uncertainty that shrinks with informative data -- but stays in continuous
    rating space, so an encoder's (mu_hat, sigma_hat^2) prior maps directly to
    (initial rating, initial RD) with no urn-size discretisation step.
    """
    P, I = len(theta), len(beta)
    nchains = 2 if parallel else 1
    th = np.zeros((nchains, P))
    rd_p = np.full((nchains, P), rd0)
    bh = np.zeros(I)
    rd_i = np.full(I, rd0)
    track = []
    for t in range(n_resp):
        p = RNG.integers(0, P)
        if parallel:
            c = t % 2
            theta_sel = th[1 - c, p]
        else:
            c, theta_sel = 0, th[0, p]
        i = pick_adaptive(theta_sel, bh, target)
        x = int(RNG.random() < SIG(theta[p] - beta[i]))

        theta_c, rd_p_c = th[c, p], rd_p[c, p]
        beta_i, rd_ii = bh[i], rd_i[i]

        g_i = 1.0 / np.sqrt(1.0 + 3.0 * rd_ii ** 2 / np.pi ** 2)
        E_p = 1.0 / (1.0 + np.exp(-g_i * (theta_c - beta_i)))
        d2_p = 1.0 / (g_i ** 2 * E_p * (1 - E_p) + 1e-9)
        prec_p = 1.0 / rd_p_c ** 2 + 1.0 / d2_p
        th[c, p] = theta_c + g_i * (x - E_p) / prec_p
        rd_p[c, p] = max(rd_min, 1.0 / np.sqrt(prec_p))

        if update_items:
            g_p = 1.0 / np.sqrt(1.0 + 3.0 * rd_p_c ** 2 / np.pi ** 2)
            E_i = 1.0 / (1.0 + np.exp(-g_p * (beta_i - theta_c)))
            d2_i = 1.0 / (g_p ** 2 * E_i * (1 - E_i) + 1e-9)
            prec_i = 1.0 / rd_ii ** 2 + 1.0 / d2_i
            bh[i] = beta_i + g_p * ((1 - x) - E_i) / prec_i
            rd_i[i] = max(rd_min, 1.0 / np.sqrt(prec_i))

        if (t + 1) % 200 == 0:
            est = th.mean(axis=0)
            track.append((t + 1, est.std(), np.corrcoef(est, theta)[0, 1],
                          np.sqrt(np.mean((est - theta) ** 2)), rd_p.mean()))
    return np.array(track)


if __name__ == "__main__":
    P, I, N = 300, 400, 120_000
    theta = RNG.normal(0, 1, P)
    beta = RNG.normal(0, 1, I)
    print(f"Ground truth: SD(theta) = {theta.std():.3f}\n")
    out = {
        "Elo (adaptive + item updating)": run_elo(theta, beta, N, parallel=False),
        "Parallel Elo":                   run_elo(theta, beta, N, parallel=True),
        "Urnings":                        run_urnings(theta, beta, N),
    }
    for k, v in out.items():
        print(f"{k:34s} final SD={v[-1,1]:6.3f}  r={v[-1,2]:.3f}  RMSE={v[-1,3]:.3f}")
    np.savez("figures/tracker_sim.npz", truth_sd=theta.std(),
             **{k: v for k, v in out.items()})

    # Exploratory: Glicko-1-style tracker (native RD, continuous prior
    # injection) -- not part of the paper figure pipeline, kept out of the
    # npz so make_figures.py's fixed 3-key layout is untouched.
    print()
    glicko_out = {
        "Glicko (single chain)": run_glicko(theta, beta, N, parallel=False),
        "Glicko (parallel chains)": run_glicko(theta, beta, N, parallel=True),
    }
    for k, v in glicko_out.items():
        print(f"{k:34s} final SD={v[-1,1]:6.3f}  r={v[-1,2]:.3f}  RMSE={v[-1,3]:.3f}  mean RD={v[-1,4]:.3f}")
