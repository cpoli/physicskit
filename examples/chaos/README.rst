Examples
========

This gallery walks through every public feature of ``physicskit.chaos``: five
families of 2D billiards, discrete chaotic maps, continuous chaotic flows,
their quantum-mechanical analogues, the diagnostics used to quantify chaos
(Lyapunov exponents, fractal dimension, Poincare recurrence, phase-space
volume), and the interactive/animated visualizers built on top of them.

Each script in this gallery is self-contained and can be run directly with
``python examples/chaos/<section>/<script>.py``. Every script also carries
an RST module docstring as its title/description and uses ``# %%`` markers
to split narrative text from code, which is exactly what Sphinx-Gallery
renders into the pages below -- the script *is* the source of truth for what
you see, not a copy of it.

Sections
--------

- **billiards** -- trajectories and Poincare (boundary phase-space) sections
  for the Circle, Ellipse, Rectangle, Sinai, Bunimovich Stadium, and
  Truncated Circle billiards, a side-by-side comparison of all five
  canonical geometries, and a demonstration of extreme sensitivity to
  initial conditions in a barely-perturbed circle.
- **maps** -- discrete-time chaos: the period-doubling route to chaos in the
  Logistic Map, the area-preserving Chirikov-Taylor Standard Map, the
  stretch-cut-and-stack Baker's Map, and the Henon Map.
- **continuous_systems** -- classic flows: the Lorenz and Rossler attractors,
  Chua's Circuit (the double-scroll attractor), the driven Duffing
  oscillator, the double pendulum, the Magnetic Pendulum's fractal basins of
  attraction, and the restricted three-body problem that first led Poincare
  to chaos.
- **quantum_chaos** -- what becomes of these systems under quantization: the
  Quantum Baker's Map, eigenstates of a quantum billiard, and the Quantum
  Kicked Rotor.
- **chaos_metrics** -- quantifying chaos rather than just plotting it:
  Lyapunov divergence and the full Lyapunov spectrum (Benettin's QR method),
  fractal (box-counting and correlation) dimension, bifurcation diagrams,
  energy drift, phase-space volume contraction (Liouville's theorem),
  Poincare recurrence and Kac's lemma, and extracting chaos diagnostics from
  a raw time series with no equations at all.
- **interactive_and_animation** -- a live Matplotlib animation, trajectory
  color-coding, and interactive Plotly figures.
- **advanced** -- working below the ``System`` abstraction: building a custom
  dynamical system directly from the low-level Numba integrators, comparing
  symplectic against non-symplectic integrators over long time horizons,
  saving and loading results, and interactive 3D viewing with PyVista.
