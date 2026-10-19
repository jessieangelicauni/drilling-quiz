"""
Follow-up to simulate_trackers.py: does each tracker keep up when true
ability is *not* static but drifts upward over time (a learner who is
actually improving), instead of just converging precisely to a fixed value?

Same regime as simulate_trackers.py (adaptive selection + simultaneous item
calibration), but each learner's true theta now increases steadily with
their own response count. Error is measured against the CURRENT true value
at each checkpoint, not a single fixed ground truth.

Reports both RMSE (tracking accuracy) and mean lag = mean(estimate - true)
-- a persistently negative, growing lag means the tracker is too sticky to
follow real improvement.
"""
import numpy as np
from simulate_trackers import RNG, SIG, pick_adaptive, _logit_urn


def run_elo_drift(theta0, beta, n_resp, drift_rate, K=0.3, target=0.5, parallel=False):
    P, I = len(theta0), len(beta)
    nchains = 2 if parallel else 1
    th = np.zeros((nchains, P))
    bh = np.zeros(I)
    count = np.zeros(P, dtype=int)
    track = []
    for t in range(n_resp):
        p = RNG.integers(0, P)
        true_now = theta0[p] + drift_rate * count[p]
        if parallel:
            c = t % 2
            theta_sel = th[1 - c, p]
        else:
            c, theta_sel = 0, th[0, p]
        i = pick_adaptive(theta_sel, bh, target)
        x = int(RNG.random() < SIG(true_now - beta[i]))
        exp = SIG(th[c, p] - bh[i])
        th[c, p] += K * (x - exp)
        bh[i] -= K * (x - exp)
        count[p] += 1
        if (t + 1) % 200 == 0:
            est = th.mean(axis=0)
            err = est - (theta0 + drift_rate * count)
            track.append((t + 1, np.sqrt(np.mean(err ** 2)), err.mean()))
    return np.array(track)


def run_urnings_drift(theta0, beta, n_resp, drift_rate, n_p=32, n_i=32, target=0.5):
    P, I = len(theta0), len(beta)
    kp = np.full(P, n_p // 2)
    ki = np.full(I, n_i // 2)
    count = np.zeros(P, dtype=int)
    track = []
    eps = 1e-9
    for t in range(n_resp):
        p = RNG.integers(0, P)
        true_now = theta0[p] + drift_rate * count[p]
        pi_p = np.clip(kp[p] / n_p, eps, 1 - eps)
        theta_sel = _logit_urn(kp[p], n_p)
        beta_hat = -_logit_urn(ki, n_i)
        i = pick_adaptive(theta_sel, beta_hat, target)
        x = int(RNG.random() < SIG(true_now - beta[i]))
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
        count[p] += 1
        if (t + 1) % 200 == 0:
            est = _logit_urn(kp, n_p)
            err = est - (theta0 + drift_rate * count)
            track.append((t + 1, np.sqrt(np.mean(err ** 2)), err.mean()))
    return np.array(track)


def run_glicko_drift(theta0, beta, n_resp, drift_rate, target=0.5, parallel=False,
                      rd0=1.0, rd_min=0.05):
    P, I = len(theta0), len(beta)
    nchains = 2 if parallel else 1
    th = np.zeros((nchains, P))
    rd_p = np.full((nchains, P), rd0)
    bh = np.zeros(I)
    rd_i = np.full(I, rd0)
    count = np.zeros(P, dtype=int)
    track = []
    for t in range(n_resp):
        p = RNG.integers(0, P)
        true_now = theta0[p] + drift_rate * count[p]
        if parallel:
            c = t % 2
            theta_sel = th[1 - c, p]
        else:
            c, theta_sel = 0, th[0, p]
        i = pick_adaptive(theta_sel, bh, target)
        x = int(RNG.random() < SIG(true_now - beta[i]))

        theta_c, rd_p_c = th[c, p], rd_p[c, p]
        beta_i, rd_ii = bh[i], rd_i[i]

        g_i = 1.0 / np.sqrt(1.0 + 3.0 * rd_ii ** 2 / np.pi ** 2)
        E_p = 1.0 / (1.0 + np.exp(-g_i * (theta_c - beta_i)))
        d2_p = 1.0 / (g_i ** 2 * E_p * (1 - E_p) + 1e-9)
        prec_p = 1.0 / rd_p_c ** 2 + 1.0 / d2_p
        th[c, p] = theta_c + g_i * (x - E_p) / prec_p
        rd_p[c, p] = max(rd_min, 1.0 / np.sqrt(prec_p))

        g_p = 1.0 / np.sqrt(1.0 + 3.0 * rd_p_c ** 2 / np.pi ** 2)
        E_i = 1.0 / (1.0 + np.exp(-g_p * (beta_i - theta_c)))
        d2_i = 1.0 / (g_p ** 2 * E_i * (1 - E_i) + 1e-9)
        prec_i = 1.0 / rd_ii ** 2 + 1.0 / d2_i
        bh[i] = beta_i + g_p * ((1 - x) - E_i) / prec_i
        rd_i[i] = max(rd_min, 1.0 / np.sqrt(prec_i))

        count[p] += 1
        if (t + 1) % 200 == 0:
            est = th.mean(axis=0)
            err = est - (theta0 + drift_rate * count)
            track.append((t + 1, np.sqrt(np.mean(err ** 2)), err.mean(), rd_p.mean()))
    return np.array(track)


if __name__ == "__main__":
    P, I, N = 300, 400, 120_000
    theta0 = RNG.normal(0, 1, P)
    beta = RNG.normal(0, 1, I)
    avg_resp_per_learner = N / P
    drift_total = 1.5   # logit-units of real ability gain over the course of participation
    drift_rate = drift_total / avg_resp_per_learner

    print(f"Ground truth: SD(theta0) = {theta0.std():.3f}, "
          f"simulated learning gain = +{drift_total} logits over ~{avg_resp_per_learner:.0f} "
          f"responses/learner (drift_rate={drift_rate:.5f}/response)\n")

    runs = {
        "Elo (single chain)":       run_elo_drift(theta0, beta, N, drift_rate, parallel=False),
        "Parallel Elo":             run_elo_drift(theta0, beta, N, drift_rate, parallel=True),
        "Urnings":                  run_urnings_drift(theta0, beta, N, drift_rate),
        "Glicko (single chain)":    run_glicko_drift(theta0, beta, N, drift_rate, parallel=False),
        "Glicko (parallel chains)": run_glicko_drift(theta0, beta, N, drift_rate, parallel=True),
    }

    print(f"{'Tracker':28s} {'RMSE@40%':>9s} {'RMSE@100%':>10s} {'lag@40%':>9s} {'lag@100%':>9s}")
    for k, v in runs.items():
        mid = v[len(v) // 2 - 1]
        end = v[-1]
        print(f"{k:28s} {mid[1]:9.3f} {end[1]:10.3f} {mid[2]:9.3f} {end[2]:9.3f}")
