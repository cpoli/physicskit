"""physicskit.optics: classical and quantum optics -- rays, waves, Gaussian beams, and quantized light.

Typical usage::

    import physicskit as pk
    import numpy as np

    beam = pk.optics.GaussianBeam(wavelength=1.064e-3, w0=0.05)
    system = pk.optics.OpticalSystem([
        pk.optics.OpticalElement(pk.optics.free_space(1.0), length=1.0),
        pk.optics.OpticalElement(pk.optics.thin_lens(0.5)),
    ])
    q_out = pk.optics.propagate_q(beam.q_parameter(0.0), system.system_matrix())

- :mod:`physicskit.optics.ray` -- geometric ray optics and ABCD matrices:
  free space, thin/thick lenses, curved and flat interfaces, spherical
  mirrors, GRIN media, cascaded optical systems, and laser-cavity
  stability.
- :mod:`physicskit.optics.wave` -- scalar wave optics: circular/slit
  apertures and Fraunhofer, Fresnel, and angular-spectrum diffraction.
- :mod:`physicskit.optics.gaussian` -- Gaussian beam propagation via the
  complex beam parameter :math:`q`, Hermite-Gaussian and Laguerre-Gaussian
  higher-order modes, and :math:`M^2` beam-quality propagation.
- :mod:`physicskit.optics.quantum_optics` -- quantum optical states (Fock,
  coherent, squeezed) in a truncated photon-number basis, Wigner
  quasi-probability distributions, and the Jaynes-Cummings model of a
  two-level atom coupled to a quantized cavity mode.
- :mod:`physicskit.optics.visualizers` -- ray-trace, beam-envelope,
  diffraction-pattern, and interactive Wigner-surface plots.
"""

from physicskit.optics.gaussian import (
    GaussianBeam,
    hermite_gaussian_mode,
    laguerre_gaussian_mode,
    m2_beam_waist,
    propagate_q,
    q_to_beam_params,
)
from physicskit.optics.quantum_optics import (
    JaynesCummingsModel,
    coherent_state,
    compute_wigner_function,
    fock_state,
    squeezed_state,
    wigner_negativity,
)
from physicskit.optics.ray import (
    OpticalElement,
    OpticalSystem,
    cavity_round_trip_matrix,
    cavity_stability,
    curved_interface,
    flat_interface,
    free_space,
    grin_medium,
    spherical_mirror,
    thick_lens,
    thin_lens,
)
from physicskit.optics.visualizers import (
    animate_diffraction_propagation,
    interactive_wigner_surface,
    plot_beam_envelope,
    plot_diffraction_pattern,
    plot_ray_trace,
)
from physicskit.optics.wave import (
    angular_spectrum_propagate,
    circular_aperture,
    double_slit_aperture,
    field_grid,
    fraunhofer_diffraction,
    fresnel_diffraction,
    intensity,
    single_slit_aperture,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # ray
    "free_space",
    "thin_lens",
    "flat_interface",
    "curved_interface",
    "thick_lens",
    "spherical_mirror",
    "grin_medium",
    "OpticalElement",
    "OpticalSystem",
    "cavity_round_trip_matrix",
    "cavity_stability",
    # wave
    "circular_aperture",
    "single_slit_aperture",
    "double_slit_aperture",
    "fraunhofer_diffraction",
    "fresnel_diffraction",
    "angular_spectrum_propagate",
    "intensity",
    "field_grid",
    # gaussian
    "propagate_q",
    "q_to_beam_params",
    "GaussianBeam",
    "hermite_gaussian_mode",
    "laguerre_gaussian_mode",
    "m2_beam_waist",
    # quantum_optics
    "fock_state",
    "coherent_state",
    "squeezed_state",
    "compute_wigner_function",
    "wigner_negativity",
    "JaynesCummingsModel",
    # visualizers
    "plot_ray_trace",
    "plot_beam_envelope",
    "plot_diffraction_pattern",
    "interactive_wigner_surface",
    "animate_diffraction_propagation",
]
