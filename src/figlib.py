"""Shared drawing helpers with automatic text-fit verification.

Every box measures its own rendered text against its own geometry and raises
if the text would overflow, so layout bugs fail loudly instead of silently.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

KO_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
fm.fontManager.addfont(KO_PATH)
KOF = fm.FontProperties(fname=KO_PATH).get_name()

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 7.0, "axes.linewidth": 0.7,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "mathtext.fontset": "dejavuserif",
})

INK, MUTE = "#111111", "#5a5a5a"
F_GEN, F_EST, F_VER = "#e8e8e8", "#d0d0d0", "#f5f5f5"
F_PSY, F_POL, F_EXT = "#bcbcbc", "#ebebeb", "#ffffff"

_WARN = []


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig._dims = (w, h)
    return fig, ax


def _extent(fig, artist):
    """Rendered bbox of an artist in axis-fraction units."""
    fig.canvas.draw()
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    inv = fig.axes[0].transAxes.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x1 - x0, y1 - y0


def box(fig, ax, x, y, w, h, text, fill=F_GEN, fs=7.0, bold=False,
        ls="-", lw=0.9, pad=0.90, name=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.010,rounding_size=0.020",
                 fc=fill, ec=INK, lw=lw, linestyle=ls, zorder=2))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.42,
                fontweight="bold" if bold else "normal")
    tw, th = _extent(fig, t)
    if tw > w * pad or th > h * pad:
        _WARN.append(
            f"  OVERFLOW {name or text.splitlines()[0][:34]!r}: "
            f"text {tw:.3f}x{th:.3f} vs box {w*pad:.3f}x{h*pad:.3f}")
    return t


def free(fig, ax, x, y, text, fs=7.0, ha="left", va="center", bold=False,
        italic=False, color=INK, maxw=None, name=""):
    t = ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
                linespacing=1.6, zorder=3,
                fontweight="bold" if bold else "normal",
                style="italic" if italic else "normal")
    if maxw:
        tw, _ = _extent(fig, t)
        if tw > maxw:
            _WARN.append(f"  OVERFLOW(free) {name or text[:30]!r}: {tw:.3f} > {maxw:.3f}")
    return t


def arrow(ax, p, q, ls="-", lw=0.9, rad=0.0, color=INK, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7.5,
                 lw=lw, ls=ls, color=color, zorder=4,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))


def tag(fig, ax, x, y, t, fs=6.2, color=MUTE):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, style="italic",
            color=color, zorder=6, linespacing=1.4,
            bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.95))


def finish(fig, path):
    fig.savefig(path, dpi=400)
    plt.close(fig)


def report():
    if _WARN:
        print("\n!! LAYOUT PROBLEMS:")
        for w in _WARN:
            print(w)
        raise SystemExit(1)
    print("All boxes fit.")


def ortho(ax, pts, lw=0.9, color=INK, head=True):
    """Orthogonal polyline route with an arrowhead on the final segment."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, zorder=4,
            solid_joinstyle="miter", solid_capstyle="butt")
    if head:
        arrow(ax, pts[-2], pts[-1], lw=lw, color=color)
