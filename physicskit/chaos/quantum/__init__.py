"""Quantum chaos: quantized versions of physicskit.chaos's classical maps and billiards.

Two complementary playgrounds, both built directly on top of physicskit.chaos's
existing classical systems:

- :class:`~physicskit.chaos.quantum.maps.QuantumKickedRotor` and
  :class:`~physicskit.chaos.quantum.maps.QuantumBakersMap` quantize
  :class:`~physicskit.chaos.systems.maps.StandardMap` and
  :class:`~physicskit.chaos.systems.maps.BakersMap` respectively, as finite
  ``dim``-dimensional unitary Floquet operators built from the discrete
  Fourier transform.
- :class:`~physicskit.chaos.quantum.billiards.QuantumBilliard` solves the Dirichlet
  Helmholtz eigenproblem inside any
  :class:`~physicskit.chaos.core.base_system.BilliardSystem` shape physicskit.chaos ships.

:func:`~physicskit.chaos.quantum.husimi.husimi_function` builds the phase-space
(Husimi) representation shared by both quantum map classes.

Random-matrix-theory analyses of any of these objects' spectra --
nearest-neighbor spacing distributions, spectral rigidity, and the like --
are intentionally not provided here; that is the job of the separate
``physicskit.rmt`` package, which can consume the eigenphases/wavenumbers these
classes expose directly.
"""

from __future__ import annotations

from physicskit.chaos.quantum.billiards import QuantumBilliard, points_in_billiard
from physicskit.chaos.quantum.husimi import husimi_function
from physicskit.chaos.quantum.maps import QuantumBakersMap, QuantumKickedRotor

__all__ = [
    "QuantumBakersMap",
    "QuantumBilliard",
    "QuantumKickedRotor",
    "husimi_function",
    "points_in_billiard",
]
