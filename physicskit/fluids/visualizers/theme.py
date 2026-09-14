"""Shared color constants for physicskit.fluids's Matplotlib visualizers.

Re-exports the same palette :mod:`physicskit.chaos.visualizers.theme` defines,
so a plot moving between the two packages (e.g. a von Karman street built
from :mod:`physicskit.fluids.systems.vortex_dynamics` next to a chaotic
trajectory) reads as one consistent visual language rather than two
accidentally different ones. Every visualizer still accepts its own
color/colormap keyword arguments to override these, so nothing here is
load-bearing.
"""

from __future__ import annotations

from physicskit.chaos.visualizers.theme import (
    ACCENT,
    DIVERGING_CMAP,
    MUTED,
    PRIMARY,
    QUALITATIVE_CMAP,
    SEQUENTIAL_CMAP,
    STRUCTURE,
)

__all__ = [
    "PRIMARY",
    "ACCENT",
    "STRUCTURE",
    "MUTED",
    "QUALITATIVE_CMAP",
    "SEQUENTIAL_CMAP",
    "DIVERGING_CMAP",
    "VORTICITY_CMAP",
]

#: Diverging colormap for vorticity fields (signed, centered at zero): an
#: alias for :data:`DIVERGING_CMAP`, named for its most common use in this
#: package.
VORTICITY_CMAP: str = DIVERGING_CMAP
