"""Regenerate the README hero figure: python docs/make_readme_figure.py

Writes docs/source/_static/images/readme_hero.png, which README.md embeds by
its raw.githubusercontent.com URL so it also renders on PyPI. The three panels
are shared with the per-subpackage figures (make_readme_subpackage_figures.py).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_readme_subpackage_figures import black_hole_shadow, hofstadter_butterfly, kelvin_helmholtz

OUT = Path(__file__).parent / "source" / "_static" / "images" / "readme_hero.png"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
black_hole_shadow(ax1)
hofstadter_butterfly(ax2)
kelvin_helmholtz(ax3)
fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")
