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
    B(.485, .725, .205, .185, "VERIFIER GATE\nmorph. analyser\n+ grammar\n+ round-trip", F_VER, 6.9)
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
    fig, ax = canvas(7.15, 2.85)
    B = lambda *a, **k: box(fig, ax, *a, **k)
    ys, h, w = .49, .275, .205
    B(.015, ys, w, h, "(1) GENERATE\nEXAONE emits\nitem for skill $k$", F_GEN, 7.0)
    B(.265, ys, w, h, "(2) PRIOR\nKLUE-RoBERTa\n$(\\hat\\mu_i,\\hat\\sigma_i^2)$", F_EST, 7.0)
    B(.515, ys, w, h, "(3) SEED URN\ngreen share from $\\hat\\mu_i$\nsize from $\\hat\\sigma_i^{-2}$", F_PSY, 7.0)
    B(.762, ys, .228, h, "(4) SERVE\nadaptive selection;\nurn absorbs data", F_POL, 7.0)
    for a, b in [(.220, .265), (.470, .515), (.720, .762)]:
        arrow(ax, (a, ys + h / 2), (b, ys + h / 2))

    B(.280, .055, .44, .225,
      "(5) RE-DISTIL\nre-train the encoder on Urnings posteriors\n"
      "→ sharper priors for unseen items", F_VER, 7.0)
    ortho(ax, [(.876, ys), (.876, .1675), (.720, .1675)])
    ortho(ax, [(.280, .1675), (.118, .1675), (.118, ys)])
    tag(fig, ax, .800, .345, "responses", 6.2)
    tag(fig, ax, .190, .345, "new prior", 6.2)
    free(fig, ax, .015, .935,
         "Cold start is eliminated: an item is adaptively servable the moment it is generated.",
         7.0, italic=True, maxw=.97, name="fig2-caption")
    finish(fig, "figures/fig2_gcs_loop.png")


# ------------------------------------------------------------------ Fig 3
def fig3():
    fig, ax = canvas(7.15, 3.5)
    B = lambda *a, **k: box(fig, ax, *a, **k)
    B(.015, .775, .315, .155, "KOREAN LEARNER\nCORPUS (annotated)", F_EXT, 7.0, True)
    B(.015, .345, .315, .365,
      "MISCONCEPTION TAXONOMY $\\mathcal{M}$\n(KAGAS, 4 of 14 shown)\n"
      "$m_1$ topic/nominative swap\n$m_2$ object-particle omission\n"
      "$m_3$ honorific agreement\n$m_4$ irregular-stem over-reg.", F_VER, 6.0)
    arrow(ax, (.1725, .775), (.1725, .710))

    B(.365, .630, .295, .215,
      "CONDITIONED GENERATION\nEXAONE: one distractor per\nsampled $m_j$, tagged", F_GEN, 6.6)
    arrow(ax, (.330, .575), (.370, .675), rad=-0.15)

    B(.370, .345, .285, .215,
      "VERIFIER GATE\n(a) $d_j$ realises $m_j$ only\n"
      "(b) $d_j$ is not also correct\n(c) solver still picks key", F_VER, 6.4)
    arrow(ax, (.5125, .630), (.5125, .560))
    arrow(ax, (.655, .435), (.700, .52), rad=-0.22)

    B(.370, .135, .285, .175,
      "CRITIC (LLM-as-judge)\nrejected item: diagnose\nfailure, draft a revision", F_PSY, 6.2)
    arrow(ax, (.5125, .345), (.5125, .305))
    ortho(ax, [(.370, .222), (.348, .222), (.348, .780), (.365, .780)])

    B(.700, .650, .29, .19,
      "DIAGNOSTIC ITEM\nstem · key · $\\{(d_j,m_j)\\}$\nevery option labelled", F_GEN, 7.0, True)
    B(.700, .370, .29, .215,
      "RESPONSE → DIAGNOSIS\nchoosing $d_j$ is evidence\nagainst skill $m_j$ only", F_PSY, 7.0)
    B(.700, .140, .29, .165,
      "PER-SKILL UPDATE\nonly the $m_j$ dimension", F_POL, 7.0)
    arrow(ax, (.845, .650), (.845, .585))
    arrow(ax, (.845, .370), (.845, .305))

    ax.text(.015, .070,
            "오늘 날씨가 ___ .      key 추워요      d: 춥어요  →  m₄\n"
            "할아버지께서 ___ .    key 오십니다    d: 옵니다  →  m₃",
            fontsize=7.0, fontname=KOF, va="center", ha="left", linespacing=1.75,
            bbox=dict(fc="#f7f7f7", ec=MUTE, lw=0.7, pad=4.0))
    finish(fig, "figures/fig3_distractors.png")


# ------------------------------------------------------------------ Fig 4
def fig4():
    fig, ax = canvas(7.15, 1.85)
    B = lambda *a, **k: box(fig, ax, *a, **k)
    B(.015, .30, .205, .40, "LEARNER STATE\nurn width $w_k$\nfor skill $k$", F_PSY, 7.0, True)
    B(.270, .30, .175, .40, "GATE\n$w_k>\\tau$  or\npersistent $m_j$ ?", F_POL, 7.0, True)
    arrow(ax, (.220, .50), (.270, .50))

    B(.510, .525, .475, .455,
      "ESCALATED PATH — high diagnostic value\n"
      "EXAONE reasoning mode: minimal-pair probe;\n"
      "Qwen3: contrastive feedback naming $m_j$", F_GEN, 7.0)
    B(.510, .030, .475, .405,
      "ROUTINE PATH — low diagnostic value\n"
      "EXAONE non-reasoning;\ncached short feedback", F_VER, 7.0)
    ortho(ax, [(.4450, .620), (.4775, .620), (.4775, .7525), (.5100, .7525)])
    free(fig, ax, .4775, .880, "yes", 6.5, "center")
    ortho(ax, [(.4450, .380), (.4775, .380), (.4775, .2325), (.5100, .2325)])
    free(fig, ax, .4775, .110, "no", 6.5, "center")
    finish(fig, "figures/fig4_escalation.png")


# ------------------------------------------------------------------ Fig 5
def fig5():
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
    fig.savefig("figures/fig5_simulation.png", dpi=400)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 6
def fig6():
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
    finish(fig, "figures/fig6_experiment.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f(); print("rendered:", f.__name__)
    report()
