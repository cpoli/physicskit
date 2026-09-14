"""Classical and quantum field theory: electrodynamics, solitons, and BEC vortex lattices.

Typical usage::

    import physicskit as pk
    import numpy as np

    # A dark soliton on a defocusing NLS background:
    x = np.linspace(-40, 40, 1024)
    psi0 = pk.fields.nls_dark_soliton(x, t=0.0)
"""

from .electrodynamics import (
    C0,
    EPS0,
    MU0,
    courant_limit_1d,
    courant_limit_2d,
    dielectric_slab,
    fdtd_1d,
    fdtd_2d_tmz,
    fdtd_2d_tmz_evolve,
    flux_tube_energy_density_2d,
    flux_tube_field_1d,
    oscillating_dipole_source,
    pml_conductivity_profile,
    pml_conductivity_profile_2d,
    poynting_vector_tmz,
    tmz_cavity_mode,
)
from .quantum_fields import (
    casimir_energy_1d,
    casimir_mode_frequencies,
    count_vortices,
    gpe_energy,
    gpe_evolve,
    gpe_imprint_vortex,
    gpe_relax,
    harmonic_trap_grid,
)
from .solitons import (
    kdv_evolve,
    kdv_evolve_frames,
    kdv_soliton,
    kdv_step,
    nls_bright_soliton,
    nls_dark_soliton,
    nls_evolve,
    nls_evolve_frames,
    sine_gordon_evolve,
    sine_gordon_evolve_frames,
    sine_gordon_kink,
)
from .visualizers import (
    animate_casimir_modes,
    animate_density_2d,
    animate_field_1d,
    animate_field_2d,
    animate_flux_tube,
    plot_bec_density,
    plot_bec_phase,
    plot_field_1d,
    plot_poynting_field,
)

__all__ = [
    "EPS0",
    "MU0",
    "C0",
    "courant_limit_1d",
    "courant_limit_2d",
    "fdtd_1d",
    "fdtd_2d_tmz",
    "fdtd_2d_tmz_evolve",
    "pml_conductivity_profile",
    "pml_conductivity_profile_2d",
    "oscillating_dipole_source",
    "dielectric_slab",
    "tmz_cavity_mode",
    "poynting_vector_tmz",
    "flux_tube_field_1d",
    "flux_tube_energy_density_2d",
    "kdv_soliton",
    "kdv_step",
    "kdv_evolve",
    "kdv_evolve_frames",
    "nls_bright_soliton",
    "nls_dark_soliton",
    "nls_evolve",
    "nls_evolve_frames",
    "sine_gordon_kink",
    "sine_gordon_evolve",
    "sine_gordon_evolve_frames",
    "harmonic_trap_grid",
    "gpe_imprint_vortex",
    "gpe_relax",
    "gpe_evolve",
    "gpe_energy",
    "count_vortices",
    "casimir_mode_frequencies",
    "casimir_energy_1d",
    "plot_field_1d",
    "plot_poynting_field",
    "plot_bec_density",
    "plot_bec_phase",
    "animate_field_1d",
    "animate_field_2d",
    "animate_density_2d",
    "animate_flux_tube",
    "animate_casimir_modes",
]
