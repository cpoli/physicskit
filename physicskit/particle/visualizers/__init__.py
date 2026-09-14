"""Plotting and animation helpers for physicskit.particle.

- :mod:`physicskit.particle.visualizers.static` -- decay-chain and
  differential-cross-section line plots.
- :mod:`physicskit.particle.visualizers.animations` -- animations of the
  toy shower/confinement/electroweak/neutrino/decay demonstrations
  built on :mod:`physicskit.particle`'s collider, confinement,
  electroweak, and neutrinos modules.
"""

from physicskit.particle.visualizers.animations import (
    animate_cp_asymmetry,
    animate_decay_chain_bars,
    animate_detector_event,
    animate_higgs_rollover,
    animate_michel_histogram,
    animate_neutrino_oscillation,
    animate_particle_cascade,
    animate_parton_shower,
    animate_qed_angular_distribution,
    animate_string_breaking,
)
from physicskit.particle.visualizers.static import plot_decay_chain, plot_differential_cross_section

__all__ = [
    "plot_decay_chain",
    "plot_differential_cross_section",
    "animate_particle_cascade",
    "animate_parton_shower",
    "animate_detector_event",
    "animate_string_breaking",
    "animate_higgs_rollover",
    "animate_decay_chain_bars",
    "animate_neutrino_oscillation",
    "animate_qed_angular_distribution",
    "animate_michel_histogram",
    "animate_cp_asymmetry",
]
