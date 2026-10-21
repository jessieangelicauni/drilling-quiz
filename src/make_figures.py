"""Publication figures for the KO-DRILL paper (IEEE, grayscale-safe)."""
import numpy as np
from figlib import (canvas, box, free, arrow, tag, finish, report, ortho,
                    KOF, INK, MUTE, F_GEN, F_EST, F_VER, F_PSY, F_POL, F_EXT)
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ Fig 1
def fig1():
    fig, ax = canvas(7.15, 4.35)
    B = lambda *a, **k: box(fig, ax, *a, **k)

    # -- Row 1: item foundry (left) + difficulty prior (right)
    ax.add_patch(plt.Rectangle((.025, .695), .685, .295, fill=False,
                 ec=MUTE, lw=0.85, ls=(0, (4, 3)), zorder=1))
    free(fig, ax, .040, .957, "ITEM FOUNDRY", 6.9, bold=True, color=MUTE)
    B(.045, .725, .195, .185, "Role 1\nGENERATOR\nEXAONE\nstem + key", F_GEN, 6.9)
    B(.265, .725, .195, .185, "Role 2\nDISTRACTORS\nEXAONE\none per $m_j$", F_GEN, 6.9)
    B(.485, .725, .205, .185, "VERIFIER GATE\nKAGAS annotator\n+ log-prob\n+ critic", F_VER, 6.9)
    B(.755, .725, .220, .185, "Role 3\nDIFFICULTY PRIOR\nKLUE-RoBERTa\n$(\\hat\\mu_i,\\hat\\sigma_i^2)$", F_EST, 6.9)
    arrow(ax, (.240, .8175), (.265, .8175))
    arrow(ax, (.460, .8175), (.485, .8175))
    arrow(ax, (.690, .8175), (.755, .8175))

    # -- Row 2: item pool (left) + Urnings tracker (right)
    B(.045, .495, .375, .155, "CALIBRATED ITEM POOL", F_EXT, 7.0, True)
    B(.520, .495, .455, .155,
      "MULTIDIMENSIONAL URNINGS TRACKER\nper-skill ability + uncertainty", F_PSY, 6.8, True)
    ortho(ax, [(.5875, .725), (.5875, .678), (.2325, .678), (.2325, .650)])
    arrow(ax, (.8650, .725), (.8650, .650))
    tag(fig, ax, .905, .688, "prior", 6.2)

    # -- Row 3: selection policy (left) + uncertainty gate (right)
    B(.045, .265, .375, .155, "SELECTION POLICY\ntarget success $\\approx 0.85$", F_POL, 7.0)
    B(.520, .265, .285, .155, "UNCERTAINTY GATE\nescalate if urn is wide", F_POL, 7.0)
    arrow(ax, (.2325, .495), (.2325, .420))
    arrow(ax, (.5700, .495), (.5700, .420))
    ortho(ax, [(.5350, .495), (.5350, .455), (.4700, .455),
               (.4700, .3425), (.4250, .3425)])
    tag(fig, ax, .470, .392, "ability", 6.2)

    # -- Row 4: learner, explainer, response log
    B(.045, .060, .255, .155, "LEARNER\n(KFL student)", F_EXT, 7.0, True)
    B(.390, .060, .290, .155, "Role 4 — EXPLAINER\nQwen3 · English feedback", F_GEN, 7.0)
    B(.790, .060, .185, .155, "RESPONSE\nLOG", F_EXT, 7.0, True)
    arrow(ax, (.2325, .265), (.2325, .215))
    tag(fig, ax, .300, .240, "item", 6.2)
    arrow(ax, (.6625, .265), (.6625, .215))
    arrow(ax, (.390, .1375), (.300, .1375))

    # -- bottom return lane (no crossings)
    ax.plot([.1725, .1725, .8825, .8825], [.060, .020, .020, .060],
            color=INK, lw=0.9, zorder=4)
    arrow(ax, (.8820, .030), (.8825, .060))
    tag(fig, ax, .527, .020, "responses", 6.2)
    # -- right channel back up to the tracker
    ax.plot([.9300, .9300], [.215, .495], color=INK, lw=0.9, zorder=4)
    arrow(ax, (.9300, .470), (.9300, .495))
    tag(fig, ax, .930, .355, "posterior\nupdate", 6.2)

    finish(fig, "figures/fig1_architecture.png")


# ------------------------------------------------------------------ Fig 2
def fig2():
    fig, ax = canvas(7.15, 3.5)
    B = lambda *a, **k: box(fig, ax, *a, **k)
    B(.015, .700, .315, .230,
      "SKILL CODEBOOK\nEXAONE names misconceptions;\nkept if one edit in the\nskill's KAGAS category", F_EXT, 6.2, True)
    B(.015, .345, .315, .290,
      "LEARNER MODEL (Eqs. 2-3)\npropensity $\\eta_{p,m}$ per misconception,\nmultinomial choice among the\noffered distractors, online update\ndominant $m^*$ = argmax $\\eta$", F_PSY, 6.0)

    B(.365, .680, .295, .250,
      "GENERATION (EXAONE)\nsentence, key span, and\n3 distractors, one per $m_j$\n(one realises $m^*$ when known)", F_GEN, 6.3)
    arrow(ax, (.330, .815), (.365, .815))
    arrow(ax, (.330, .490), (.370, .730), rad=-0.15)

    B(.370, .330, .285, .300,
      "VERIFIER (no model judges)\n(a) one edit, tested category\n(b) code = edit, or codebook\n(c) the three codes differ\n(d) key outscores each $d_j$\n     (Qwen3 log-probability)", F_VER, 6.0)
    arrow(ax, (.5125, .680), (.5125, .630))

    B(.370, .110, .285, .170,
      "CRITIC\nfailure reasons back to\nEXAONE, up to 2 revisions", F_POL, 6.2)
    arrow(ax, (.5125, .330), (.5125, .280))
    ortho(ax, [(.370, .195), (.348, .195), (.348, .760), (.365, .760)])

    B(.700, .680, .29, .190,
      "DIAGNOSTIC ITEM\nstem · key · $\\{(d_j,m_j)\\}$\nevery distractor coded", F_GEN, 6.8, True)
    arrow(ax, (.655, .480), (.700, .740), rad=-0.22)
    B(.700, .400, .29, .215,
      "WRONG ANSWER\nchosen $d_j$ against the offered\ncodes updates the learner\nmodel: one step on $\\eta$ (Eq. 3)", F_PSY, 6.5)
    B(.700, .140, .29, .190,
      "NEXT ITEM\ntargets $m^*$; same $m_j$ twice\nescalates feedback (Eq. 5)", F_POL, 6.5)
    arrow(ax, (.845, .680), (.845, .615))
    arrow(ax, (.845, .400), (.845, .330))

    ax.text(.015, .060,
            "동생이 공원___ 축구를 해요.    key 에서    d: 에 → PART:에서>에 · 을 → PART:에서>을 · 이 → PART:에서>이",
            fontsize=6.6, fontname=KOF, va="center", ha="left",
            bbox=dict(fc="#f7f7f7", ec=MUTE, lw=0.7, pad=4.0))
    finish(fig, "figures/fig2_distractors.png")


# ------------------------------------------------------------------ Fig 3
def fig3():
    fig, ax = canvas(7.15, 1.85)
    B = lambda *a, **k: box(fig, ax, *a, **k)
    B(.015, .30, .205, .40, "LEARNER STATE\nurn width $w_k$\nfor skill $k$", F_PSY, 7.0, True)
    B(.270, .30, .175, .40, "GATE\n$w_k>\\tau$  or\npersistent $m_j$ ?", F_POL, 7.0, True)
    arrow(ax, (.220, .50), (.270, .50))

    B(.510, .525, .475, .455,
      "ESCALATED PATH — high diagnostic value\n"
      "Qwen3 reasoning mode: contrastive feedback\n"
      "naming the specific misconception $m_j$", F_GEN, 7.0)
    B(.510, .030, .475, .405,
      "ROUTINE PATH — low diagnostic value\n"
      "Qwen3 non-reasoning;\nshort cached feedback", F_VER, 7.0)
    ortho(ax, [(.4450, .620), (.4775, .620), (.4775, .7525), (.5100, .7525)])
    free(fig, ax, .4775, .880, "yes", 6.5, "center")
    ortho(ax, [(.4450, .380), (.4775, .380), (.4775, .2325), (.5100, .2325)])
    free(fig, ax, .4775, .110, "no", 6.5, "center")
    finish(fig, "figures/fig3_escalation.png")


# ------------------------------------------------------------------ Fig 4
def fig4():
    d = np.load("figures/tracker_sim.npz")
    truth = float(d["truth_sd"])
    keys = [k for k in d.files if k != "truth_sd"]
    st = {keys[0]: ("-", "#111111", 1.5, "Elo (adaptive + item updating)"),
          keys[1]: ("--", "#555555", 1.3, "Parallel Elo"),
          keys[2]: ("-.", "#8d8d8d", 1.5, "Urnings")}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.5))
    for k in keys:
        ls, c, lw, lab = st[k]
        axes[0].plot(d[k][:, 0] / 1000, d[k][:, 1], ls, color=c, lw=lw, label=lab)
        axes[1].plot(d[k][:, 0] / 1000, d[k][:, 3], ls, color=c, lw=lw)
    axes[0].axhline(truth, color="#000", lw=0.9, ls=(0, (1, 2)))
    axes[0].text(119, truth - .20, "true SD", fontsize=6.5, ha="right", style="italic")
    axes[0].set_ylabel("SD of learner ratings (logit)", fontsize=7.2)
    axes[0].set_title("(a) Rating-scale stability", fontsize=7.6)
    axes[0].legend(fontsize=6.2, frameon=False, loc="upper left")
    axes[1].set_ylabel("RMSE vs. true ability", fontsize=7.2)
    axes[1].set_title("(b) Recovery of true ability", fontsize=7.6)
    for a in axes:
        a.set_xlabel("responses observed (thousands)", fontsize=7.2)
        a.tick_params(labelsize=6.6)
        a.grid(alpha=.22, lw=.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig("figures/fig4_simulation.png", dpi=400)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 5
def fig5():
    fig, ax = canvas(7.15, 1.75)
    B = lambda *a, **k: box(fig, ax, *a, **k)
    B(.015, .20, .205, .60, "n = 180\nKFL learners\nblock-randomised\nby TOPIK level", F_EXT, 7.0, True)
    arms = [("A — static bank, random drill", F_EXT),
            ("B — static bank, Elo adaptive", F_VER),
            ("C — KO-DRILL (full system)", F_PSY)]
    for n, (t, f) in enumerate(arms):
        B(.280, .675 - n * .245, .375, .165, t, f, 7.0, n == 2)
    for yy in (.7575, .5125, .2675):
        ortho(ax, [(.2200, .50), (.2500, .50), (.2500, yy), (.2800, yy)])

    for x, t in [(.700, "Pre-test\nwk 0"), (.798, "Intervention\nwk 1–6"),
                 (.892, "Post-test\nwk 7"), (.965, "Retention\nwk 11")]:
        ax.plot([x, x], [.22, .80], color=MUTE, lw=0.7, ls=(0, (2, 2)))
        ax.text(x, .905, t, ha="center", fontsize=6.0, style="italic")
    ax.annotate("", xy=(.995, .14), xytext=(.690, .14),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK))
    finish(fig, "figures/fig5_experiment.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(); print("rendered:", f.__name__)
    report()
