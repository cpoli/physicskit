"""Condensed matter physics: tight-binding models, topological band theory, correlated electrons, and quantum spin chains.

Typical usage::

    import physicskit as pk
    import numpy as np

    H = lambda k1, k2: pk.condensed.haldane_model(k1, k2, phi=np.pi / 2)
    chern_numbers = pk.condensed.compute_chern_number(H, grid_size=30)
"""

from .anderson_localization import (
    anderson_chain_hamiltonian,
    inverse_participation_ratio,
    localization_length,
)
from .correlated import (
    bdg_bcs_hamiltonian,
    bdg_spectrum,
    hubbard_1d_exact_diagonalization,
    hubbard_spin_correlations,
)
from .ginzburg_landau import (
    ginzburg_landau_parameter,
    gl_coherence_length,
    gl_equilibrium_order_parameter,
    gl_free_energy_density,
    gl_order_parameter_profile,
    gl_penetration_depth,
)
from .landau_levels import (
    cyclotron_frequency,
    filling_factor,
    landau_degeneracy,
    landau_density_of_states,
    landau_level_energies,
    magnetic_length,
)
from .laughlin import laughlin_metropolis_sweep, laughlin_pair_correlation, laughlin_radial_density
from .models import (
    bhz_hamiltonian,
    bhz_ribbon_hamiltonian,
    graphene_hamiltonian,
    graphene_lattice_hamiltonian,
    haldane_lattice_hamiltonian,
    haldane_model,
    harper_hofstadter_hamiltonian,
    kane_mele_hamiltonian,
    kitaev_chain_bdg_real_space,
    kitaev_chain_hamiltonian,
    ssh_hamiltonian,
    ssh_lattice_hamiltonian,
)
from .spin_chains import (
    bethe_ansatz_xxx_ground_energy,
    calabrese_cardy_entropy,
    embed_state,
    energy_gap,
    entanglement_entropy,
    entanglement_profile,
    fit_central_charge,
    lowest_eigenstates,
    spin_chain_basis,
    tfim_free_fermion_spectrum,
    tfim_ground_state_energy,
    tfim_hamiltonian,
    xxz_hamiltonian,
)
from .tight_binding import Hamiltonian, Lattice, apply_peierls_phase, build_ribbon
from .topological_insulator_3d import (
    surface_dirac_hamiltonian,
    topological_insulator_3d_hamiltonian,
    topological_insulator_3d_slab_hamiltonian,
)
from .topology import compute_berry_curvature, compute_chern_number, z2_invariant, zak_phase
from .visualizers import (
    plot_band_structure,
    plot_berry_curvature,
    plot_edge_state_density,
    plot_fermi_surface_3d,
)
from .weyl import weyl_node_locations, weyl_semimetal_hamiltonian, weyl_semimetal_slab_hamiltonian

__all__ = [
    "Lattice",
    "Hamiltonian",
    "build_ribbon",
    "apply_peierls_phase",
    "ssh_hamiltonian",
    "ssh_lattice_hamiltonian",
    "graphene_hamiltonian",
    "graphene_lattice_hamiltonian",
    "haldane_model",
    "haldane_lattice_hamiltonian",
    "kane_mele_hamiltonian",
    "bhz_hamiltonian",
    "bhz_ribbon_hamiltonian",
    "kitaev_chain_hamiltonian",
    "kitaev_chain_bdg_real_space",
    "harper_hofstadter_hamiltonian",
    "compute_berry_curvature",
    "compute_chern_number",
    "zak_phase",
    "z2_invariant",
    "cyclotron_frequency",
    "magnetic_length",
    "landau_level_energies",
    "landau_degeneracy",
    "landau_density_of_states",
    "filling_factor",
    "laughlin_metropolis_sweep",
    "laughlin_pair_correlation",
    "laughlin_radial_density",
    "gl_free_energy_density",
    "gl_equilibrium_order_parameter",
    "gl_coherence_length",
    "gl_penetration_depth",
    "ginzburg_landau_parameter",
    "gl_order_parameter_profile",
    "anderson_chain_hamiltonian",
    "inverse_participation_ratio",
    "localization_length",
    "topological_insulator_3d_hamiltonian",
    "topological_insulator_3d_slab_hamiltonian",
    "surface_dirac_hamiltonian",
    "bdg_bcs_hamiltonian",
    "bdg_spectrum",
    "hubbard_1d_exact_diagonalization",
    "hubbard_spin_correlations",
    "plot_band_structure",
    "plot_berry_curvature",
    "plot_edge_state_density",
    "plot_fermi_surface_3d",
    "weyl_semimetal_hamiltonian",
    "weyl_node_locations",
    "weyl_semimetal_slab_hamiltonian",
    "spin_chain_basis",
    "xxz_hamiltonian",
    "tfim_hamiltonian",
    "lowest_eigenstates",
    "energy_gap",
    "embed_state",
    "entanglement_entropy",
    "entanglement_profile",
    "calabrese_cardy_entropy",
    "fit_central_charge",
    "bethe_ansatz_xxx_ground_energy",
    "tfim_free_fermion_spectrum",
    "tfim_ground_state_energy",
]
