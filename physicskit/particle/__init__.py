"""physicskit.particle: relativistic kinematics, decays, scattering, and nuclear physics.

Works throughout in natural units with :math:`c=1` (:mod:`~physicskit.particle.nuclear`
specifically uses MeV, the standard convention for its formulas).

- :mod:`physicskit.particle.kinematics` -- four-vectors, Lorentz boosts,
  invariant mass, rapidity, and center-of-momentum frames.
- :mod:`physicskit.particle.decays` -- two-body decay kinematics,
  radioactive decay chains (the Bateman equations), and the muon's
  three-body Michel-spectrum decay.
- :mod:`physicskit.particle.scattering` -- Mandelstam kinematics and
  Rutherford scattering.
- :mod:`physicskit.particle.nuclear` -- the semi-empirical mass formula
  and nuclear reaction Q-values.
- :mod:`physicskit.particle.collider` -- toy branching cascades
  (particle showers, parton showers) and a schematic detector geometry.
- :mod:`physicskit.particle.confinement` -- toy quark confinement and
  QCD string breaking.
- :mod:`physicskit.particle.electroweak` -- toy classical symmetry
  breaking (Higgs mechanism), leading-order QED annihilation, and
  neutral-meson CP violation.
- :mod:`physicskit.particle.neutrinos` -- two-flavor vacuum neutrino oscillations.
- :mod:`physicskit.particle.visualizers` -- decay-chain and
  differential-cross-section plots, plus animations of all of the above.
"""

from physicskit.particle.collider import (
    ShowerParticle,
    charged_track_points,
    cluster_into_jets,
    flatten_shower,
    parton_shower,
    shower_leaves,
    simple_shower,
)
from physicskit.particle.confinement import string_break_chain, string_tension_energy
from physicskit.particle.decays import (
    activity,
    bateman_decay_chain,
    decay_constant,
    half_life,
    michel_spectrum,
    muon_decay_event,
    radioactive_decay_number,
    sample_michel_electron_energies,
    two_body_decay,
    two_body_decay_momentum,
)
from physicskit.particle.electroweak import (
    cp_asymmetry,
    higgs_field_rollover,
    higgs_potential,
    higgs_vev,
    meson_decay_rates_cp_eigenstate,
    qed_dsigma_domega_mumu,
    qed_total_cross_section_mumu,
)
from physicskit.particle.kinematics import FourVector, boost, boost_generic, boost_to_com, invariant_mass, rapidity
from physicskit.particle.neutrinos import oscillation_probability, survival_probability
from physicskit.particle.nuclear import binding_energy_per_nucleon, q_value, semf_binding_energy
from physicskit.particle.scattering import (
    ALPHA_FS,
    impact_parameter,
    mandelstam_s,
    mandelstam_t,
    mandelstam_u,
    rutherford_dsigma_domega,
)
from physicskit.particle.visualizers import (
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
    plot_decay_chain,
    plot_differential_cross_section,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # kinematics
    "FourVector",
    "boost",
    "boost_generic",
    "invariant_mass",
    "rapidity",
    "boost_to_com",
    # decays
    "two_body_decay_momentum",
    "two_body_decay",
    "decay_constant",
    "half_life",
    "radioactive_decay_number",
    "activity",
    "bateman_decay_chain",
    "michel_spectrum",
    "sample_michel_electron_energies",
    "muon_decay_event",
    # scattering
    "ALPHA_FS",
    "mandelstam_s",
    "mandelstam_t",
    "mandelstam_u",
    "rutherford_dsigma_domega",
    "impact_parameter",
    # nuclear
    "semf_binding_energy",
    "binding_energy_per_nucleon",
    "q_value",
    # collider
    "ShowerParticle",
    "simple_shower",
    "parton_shower",
    "flatten_shower",
    "shower_leaves",
    "cluster_into_jets",
    "charged_track_points",
    # confinement
    "string_tension_energy",
    "string_break_chain",
    # electroweak
    "higgs_potential",
    "higgs_vev",
    "higgs_field_rollover",
    "qed_dsigma_domega_mumu",
    "qed_total_cross_section_mumu",
    "meson_decay_rates_cp_eigenstate",
    "cp_asymmetry",
    # neutrinos
    "oscillation_probability",
    "survival_probability",
    # visualizers
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
