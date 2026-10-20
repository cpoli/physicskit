"""
Build the figures and tables of the manuscript.

- fig_<pair>: six panels each, two subpackages per figure, drawn by the
  panel functions of ``docs/make_readme_subpackage_figures.py`` (condensed
  gallery examples), relabelled (a)-(f) for print.
- fig_validation: the largest deviation of every cross-check.
- fig_benchmarks: time against problem size.
- tables/*.tex: the cross-checks, the analytic catalogue and the benchmarks.

Reads ``paper/results/*.json`` (written by validation/run_all.py and
benchmarks/run_benchmarks.py). Writes to ``paper/manuscript/figures`` and
``paper/manuscript/tables``.

Usage (from the repository root)::

    python paper/figures/build_all.py            # everything
    python paper/figures/build_all.py tables     # tables and data figures only
"""

import json
import string
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import Collection  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper" / "results"
FIGS = ROOT / "paper" / "manuscript" / "figures"
TABLES = ROOT / "paper" / "manuscript" / "tables"
sys.path.insert(0, str(ROOT / "docs"))

# Categorical slots of the reference palette, fixed per package across every
# figure; each package also has its own marker and line style, so color is
# never the only cue.
COLORS = {"physicskit": "#2a78d6", "scipy": "#eb6834", "qutip": "#1baf7a", "astropy": "#4a3aa7"}
MARKERS = {"physicskit": "o", "scipy": "s", "qutip": "^", "astropy": "D"}
STYLES = {"physicskit": "-", "scipy": "--", "qutip": ":", "astropy": "-."}
LABELS = {"physicskit": "physicskit", "scipy": "SciPy", "qutip": "QuTiP", "astropy": "Astropy"}
WIDTH = 7.0  # inches, full text width

# Two subpackages per figure, in the order of the paper's sections.
PAIRS = {
    "mechanics": ("classical", "chaos"),
    "quantum": ("quantum", "semiclassical"),
    "matter": ("condensed", "statphys"),
    "waves": ("fields", "optics"),
    "flows": ("fluids", "plasma"),
    "cosmos": ("relativity", "astro"),
    "particles": ("particle", "rmt"),
}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#555555",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)


def rasterize_dense(ax):
    """Rasterize images and many-point artists, so a panel of 10^5 dots stays a small PDF."""
    many_lines = len(ax.get_lines()) > 100
    for artist in ax.get_children():
        if isinstance(artist, AxesImage | Collection) or (isinstance(artist, Line2D) and (many_lines or len(artist.get_xdata()) > 2000)):
            artist.set_rasterized(True)


def reading_order(ax):
    """Row (top first), then left to right, from where the panel sits on the figure."""
    box = ax.get_position()
    return -round(2 * (box.y0 + box.y1) / 2), box.x0


def panels(name, subpackages):
    import make_readme_subpackage_figures as readme

    fig, grid = plt.subplots(2, 3, figsize=(WIDTH, 4.9))
    for row, sub in zip(grid, subpackages):
        readme.FIGURES[sub](row)
    for ax in fig.axes:
        ax.set_title("")
    fig.tight_layout()
    main = sorted((ax for ax in fig.axes if not hasattr(ax, "_colorbar")), key=reading_order)  # skip colorbars
    for letter, ax in zip(string.ascii_lowercase, main):
        if ax.get_legend() is not None:  # the README sizes legends for the web
            ax.legend(fontsize=6, frameon=False, loc="best")
        rasterize_dense(ax)
        if ax.name == "3d":
            ax.text2D(0.0, 1.02, f"({letter})", transform=ax.transAxes, fontweight="bold")
        else:
            ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", ha="right")
    fig.savefig(FIGS / f"fig_{name}.pdf")
    plt.close(fig)


def validation():
    data = json.loads((RESULTS / "validation.json").read_text())
    rows = [(ref, name, dev) for ref, checks in data["cross_checks"].items() for name, dev in checks.items()]
    floor = 1e-17  # exact agreement, drawn at the axis edge
    fig, ax = plt.subplots(figsize=(0.75 * WIDTH, 0.2 * len(rows) + 0.6))
    for y, (ref, _, dev) in enumerate(rows):
        ax.hlines(y, floor, max(dev, floor), color="#cccccc", lw=1)
        ax.plot(max(dev, floor), y, MARKERS[ref], color=COLORS[ref], ms=5)
    ax.set_yticks(range(len(rows)), [name for _, name, _ in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(floor, 1e-6)
    ax.set_xlabel("largest deviation from the reference package")
    for ref in data["cross_checks"]:
        ax.plot([], [], MARKERS[ref], color=COLORS[ref], label=f"vs {LABELS[ref]}")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x")
    fig.savefig(FIGS / "fig_validation.pdf")
    plt.close(fig)


BENCH_PANELS = [
    ("ode", "Kepler orbit, RK45", "orbital periods"),
    ("lindblad", "Lindblad evolution", "cavity dimension $d$"),
    ("spin_chain", "Ising chain, ground state", "spins $N$"),
    ("cosmology", "luminosity distances", "redshifts"),
]


def benchmarks():
    data = json.loads((RESULTS / "benchmarks.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.2))
    for letter, ax, (task, title, xlabel) in zip("abcd", axes, BENCH_PANELS):
        for package in dict.fromkeys(r["package"] for r in data[task]):
            rows = sorted((r for r in data[task] if r["package"] == package), key=lambda r: r["size"])
            ax.loglog(
                [r["size"] for r in rows],
                [r["seconds"] for r in rows],
                STYLES[package],
                marker=MARKERS[package],
                color=COLORS[package],
                ms=4,
                lw=1.5,
                label=LABELS[package],
            )
        sizes = sorted({r["size"] for r in data[task]})
        ax.set_xticks(sizes[:: 2 if len(sizes) > 4 else 1], minor=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(f"({letter}) {title}", loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(which="major")
        ax.legend(frameon=False)
    axes[0].set_ylabel("time (s)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_benchmarks.pdf")
    plt.close(fig)


def tex(s):
    """Escape text for LaTeX, leaving $...$ math untouched."""
    parts = s.split("$")
    for n in range(0, len(parts), 2):
        parts[n] = parts[n].replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    return "$".join(parts)


def num(x):
    return "0" if x == 0 else f"\\num{{{x:.1e}}}"


def tables():
    data = json.loads((RESULTS / "validation.json").read_text())
    lines = [f"{LABELS[ref]} & {tex(name)} & {num(dev)} \\\\" for ref, checks in data["cross_checks"].items() for name, dev in checks.items()]
    (TABLES / "crosschecks.tex").write_text("\n".join(lines) + "\n")
    lines = [f"{e['result']} & {e['reference']} & \\url{{{e['example']}}} & {tex(e['tolerance'])} \\\\" for e in data["analytic"]]
    (TABLES / "analytic.tex").write_text("\n".join(lines) + "\n")
    v = data["versions"]
    (TABLES / "versions.tex").write_text("".join(f"\\newcommand{{\\{p}version}}{{{v[p]}}}\n" for p in v))
    if (RESULTS / "benchmarks.json").exists():
        bench = json.loads((RESULTS / "benchmarks.json").read_text())
        lines = []
        for task, title, xlabel in BENCH_PANELS:
            sizes = sorted({r["size"] for r in bench[task]})
            first = True
            for size in (sizes[0], sizes[-1]):
                a, b = (r for r in bench[task] if r["size"] == size)
                label = title if first else ""
                first = False
                lines.append(
                    f"{label} & {xlabel} = {size:,} & {a['seconds']:.3g} & {LABELS[b['package']]} & {b['seconds']:.3g} & "
                    f"{b['seconds'] / a['seconds']:.2g} \\\\".replace(",", "{,}")
                )
        (TABLES / "bench.tex").write_text("\n".join(lines) + "\n")
        (TABLES / "bench_setup.tex").write_text(f"\\newcommand{{\\benchplatform}}{{{tex(bench['platform'])}}}\n")


def main(only_tables=False):
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    if not only_tables:
        for name, subpackages in PAIRS.items():
            panels(name, subpackages)
            print(f"fig_{name}.pdf")
    validation()
    print("fig_validation.pdf")
    if (RESULTS / "benchmarks.json").exists():
        benchmarks()
        print("fig_benchmarks.pdf")
    tables()
    print("tables/")


if __name__ == "__main__":
    main(only_tables="tables" in sys.argv[1:])
