"""Shared color constants for physicskit.chaos's Matplotlib and Plotly visualizers.

Every visualizer still accepts its own color/colormap keyword arguments to
override these, so nothing here is load-bearing -- it exists so that "the
trajectory color" or "the boundary color" is defined once, consistently,
rather than each visualizer module picking its own ad hoc literal string.
"""

from __future__ import annotations

#: Primary data color: trajectories, orbits, scatter points, histograms.
PRIMARY: str = "steelblue"

#: Accent/highlight color: a second trajectory, a moving point, a fit line, a mean marker.
ACCENT: str = "crimson"

#: Structural elements: billiard boundaries, walls, reference geometry.
STRUCTURE: str = "black"

#: Muted/reference color: guide lines, surrogate distributions, secondary annotations.
MUTED: str = "gray"

#: Qualitative colormap name for discretely-labeled data (e.g. basins of attraction).
QUALITATIVE_CMAP: str = "tab10"

#: Sequential colormap name for scalar-valued data (e.g. a trajectory colored by time or speed).
SEQUENTIAL_CMAP: str = "viridis"

#: Diverging colormap name for signed scalar data centered at zero (e.g. a wavefunction's
#: amplitude, where sign as well as magnitude matters).
DIVERGING_CMAP: str = "RdBu_r"
