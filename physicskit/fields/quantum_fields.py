"""Gross-Pitaevskii mean-field theory for trapped, rotating Bose-Einstein condensates.

Solves the time-dependent Gross-Pitaevskii equation (GPE)

.. math::

    i\\partial_t\\psi = \\Big[-\\tfrac{1}{2}\\nabla^2 + V(\\mathbf{r}) + g|\\psi|^2 - \\Omega L_z\\Big]\\psi

in the rotating frame (units :math:`\\hbar=m=1`) by imaginary-time
propagation (:math:`\\tau=it`), which turns the Schrodinger-like evolution
into a gradient flow that relaxes any initial state toward a stationary
point of the rotating-frame energy. The kinetic and potential/interaction
terms are each handled exactly in their natural representation (spectral
and real-space respectively, split-step); the angular-momentum term
:math:`\\Omega L_z` is advanced explicitly via centered finite differences,
which is non-stiff at the small imaginary-time steps used here.

A single quantized vortex, imprinted by multiplying the wavefunction by
:math:`(x-x_0)+i(y-y_0)`, is used throughout as the worked example of a
"vortex lattice" building block: :func:`gpe_relax` finds it as a genuine
stationary GPE solution, and comparing its rotating-frame energy against
the vortex-free ground state reproduces the standard textbook criterion
for the critical rotation frequency :math:`\\Omega_c` above which nucleating
a vortex lowers the energy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "harmonic_trap_grid",
    "gpe_imprint_vortex",
    "gpe_relax",
    "gpe_evolve",
    "gpe_energy",
    "count_vortices",
    "casimir_mode_frequencies",
    "casimir_energy_1d",
]


def harmonic_trap_grid(n: int, length: float) -> tuple:
    """Build a centered real-space and wavenumber grid for a 2D harmonic trap.

    Parameters
    ----------
    n : int
        Number of grid points along each axis.
    length : float
        Physical domain size (the domain is ``[-length/2, length/2)``).

    Returns
    -------
    X, Y : ndarray, shape (n, n)
        Real-space coordinate grids, centered at the trap origin.
    KX, KY : ndarray, shape (n, n)
        Wavenumber grids.
    K2 : ndarray, shape (n, n)
        :math:`K_X^2 + K_Y^2`.

    Examples
    --------
    >>> X, Y, KX, KY, K2 = harmonic_trap_grid(32, 12.0)
    >>> float(X[0, 0])
    -6.0
    """
    x = np.linspace(-length / 2, length / 2, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    return X, Y, KX, KY, K2


def gpe_imprint_vortex(psi: np.ndarray, X: np.ndarray, Y: np.ndarray, positions) -> np.ndarray:
    """Imprint one singly-quantized vortex per position by multiplying in a phase winding.

    Parameters
    ----------
    psi : ndarray of complex, shape (n, n)
        Wavefunction to imprint onto.
    X, Y : ndarray, shape (n, n)
        Real-space coordinate grids from :func:`harmonic_trap_grid`.
    positions : sequence of tuple(float, float)
        ``(x0, y0)`` center of each vortex to imprint.

    Returns
    -------
    ndarray of complex, shape (n, n)
        ``psi`` multiplied by :math:`\\prod_i [(x-x_{0,i}) + i(y-y_{0,i})]`,
        renormalized to preserve the input norm.

    Examples
    --------
    >>> import numpy as np
    >>> X, Y, KX, KY, K2 = harmonic_trap_grid(32, 12.0)
    >>> psi0 = np.exp(-0.5 * (X ** 2 + Y ** 2)).astype(complex)
    >>> psi = gpe_imprint_vortex(psi0, X, Y, [(0.0, 0.0)])
    >>> bool(abs(psi[16, 16]) < 1e-10)
    True
    """
    dx = X[1, 0] - X[0, 0]
    norm0 = np.sum(np.abs(psi) ** 2) * dx * dx
    out = psi.astype(complex).copy()
    for x0, y0 in positions:
        out = out * ((X - x0) + 1j * (Y - y0))
    norm1 = np.sum(np.abs(out) ** 2) * dx * dx
    return out * np.sqrt(norm0 / norm1)


def gpe_relax(
    psi0: np.ndarray,
    V: np.ndarray,
    g: float,
    dtau: float,
    steps: int,
    X: np.ndarray,
    Y: np.ndarray,
    K2: np.ndarray,
    Omega: float = 0.0,
    n_particles: float = 1.0,
) -> np.ndarray:
    """Relax a GPE initial condition toward a stationary state via imaginary-time propagation.

    Parameters
    ----------
    psi0 : ndarray of complex, shape (n, n)
        Initial wavefunction.
    V : ndarray, shape (n, n)
        External trapping potential.
    g : float
        Interaction (nonlinearity) strength.
    dtau : float
        Imaginary-time step.
    steps : int
        Number of steps to advance.
    X, Y, K2 : ndarray, shape (n, n)
        Grids from :func:`harmonic_trap_grid` (``KX``, ``KY`` are not needed here).
    Omega : float, default=0.0
        Rotation frequency of the trap.
    n_particles : float, default=1.0
        Total particle number; the wavefunction is renormalized to this
        value after every step (imaginary-time propagation does not
        conserve norm on its own).

    Returns
    -------
    ndarray of complex, shape (n, n)
        The relaxed (approximately stationary) wavefunction.

    See Also
    --------
    gpe_energy : Evaluate the energy of a relaxed state.
    gpe_imprint_vortex : Seed an initial condition with quantized vortices.

    Notes
    -----
    Energy decreases monotonically under this propagation whenever it is
    implemented correctly (a basic sanity check worth verifying on any new
    potential or parameter regime).

    Examples
    --------
    Comparing the rotating-frame energy of the vortex-free ground state
    against a state seeded with one centered vortex reproduces the
    standard vortex-nucleation criterion: :math:`\\Omega_c = \\Delta E/\\Delta L_z`,
    above which the vortex state has lower energy in the rotating frame:

    >>> import numpy as np
    >>> n, length, g = 64, 12.0, 4.0
    >>> X, Y, KX, KY, K2 = harmonic_trap_grid(n, length)
    >>> V = 0.5 * (X ** 2 + Y ** 2)
    >>> psi_vf = gpe_relax(np.exp(-0.5 * (X**2 + Y**2)).astype(complex), V, g, dtau=5e-4, steps=4000, X=X, Y=Y, K2=K2)
    >>> psi0_v = gpe_imprint_vortex(np.exp(-0.5 * (X**2 + Y**2)).astype(complex), X, Y, [(0.0, 0.0)])
    >>> psi_v = gpe_relax(psi0_v, V, g, dtau=5e-4, steps=4000, X=X, Y=Y, K2=K2)
    >>> E0, E1 = gpe_energy(psi_vf, V, g, X, Y, K2), gpe_energy(psi_v, V, g, X, Y, K2)
    >>> Omega_c = (E1["total"] - E0["total"]) / (E1["angular_momentum"] - E0["angular_momentum"])
    >>> bool(0.0 < Omega_c < 1.0)
    True
    """
    dx = X[1, 0] - X[0, 0]
    psi = np.array(psi0, dtype=complex, copy=True)
    norm = np.sum(np.abs(psi) ** 2) * dx * dx
    psi *= np.sqrt(n_particles / norm)
    for _ in range(steps):
        psi *= np.exp(-dtau * (V + g * np.abs(psi) ** 2))
        if Omega != 0.0:
            dpsidx = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dx)
            dpsidy = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dx)
            psi = psi - 1j * dtau * Omega * (X * dpsidy - Y * dpsidx)
        psi_hat = np.fft.fft2(psi)
        psi_hat *= np.exp(-dtau * 0.5 * K2)
        psi = np.fft.ifft2(psi_hat)
        norm = np.sum(np.abs(psi) ** 2) * dx * dx
        psi *= np.sqrt(n_particles / norm)
    return psi


def gpe_evolve(psi0: np.ndarray, V: np.ndarray, g: float, dt: float, steps: int, K2: np.ndarray, snapshot_stride: int = 1) -> tuple:
    """Real-time propagation of the 2D Gross-Pitaevskii / cubic NLS equation via split-step Fourier.

    Solves :math:`i\\partial_t\\psi = [-\\tfrac{1}{2}\\nabla^2 + V + g|\\psi|^2]\\psi`
    forward in *real* time (unlike :func:`gpe_relax`'s imaginary-time
    relaxation, which only finds stationary states): the kinetic term is
    exact in Fourier space, and the potential-plus-nonlinear term is exact
    as a pointwise phase rotation in real space, the same split-step
    structure as :func:`physicskit.fields.solitons.nls_evolve` generalized
    to 2D and to an external potential. Real-time evolution is unitary and
    conserves the norm on its own (no renormalization needed, unlike
    imaginary time).

    This one stepper serves two purposes, distinguished only by ``V`` and
    the sign of ``g``:

    - With a harmonic trap (``V = 0.5*(X**2+Y**2)``) and repulsive
      interactions (``g > 0``), an off-center vortex genuinely precesses
      around the trap under the density gradient -- real vortex dynamics,
      as opposed to the static relaxed state :func:`gpe_relax` finds.
    - With ``V = 0`` (or weak) and attractive interactions (``g < 0``
      here, opposite sign convention from ``nls_evolve``'s ``g`` since this
      module's equation carries a ``+g|psi|^2`` term), a sufficiently tall,
      narrow initial packet self-focuses, concentrating into a narrower,
      taller peak -- a numerical stand-in for wave collapse. True collapse
      is a singularity in finite time; this integrator (like any
      finite-grid scheme) cannot resolve it and the calculation should be
      stopped once the peak density is still visibly growing, not carried
      through the blow-up itself.

    Parameters
    ----------
    psi0 : ndarray of complex, shape (n, n)
        Initial wavefunction.
    V : ndarray, shape (n, n)
        External potential (use ``np.zeros_like`` for the free/collapse case).
    g : float
        Interaction strength and sign (see above).
    dt : float
        Time step.
    steps : int
        Number of steps to advance.
    K2 : ndarray, shape (n, n)
        Wavenumber-squared grid from :func:`harmonic_trap_grid`.
    snapshot_stride : int, default=1
        Record a snapshot every this many steps (plus the initial condition).

    Returns
    -------
    snapshots : ndarray of complex, shape (n_recorded, n, n)
        Wavefunction at ``t=0`` and after every recorded step.
    times : ndarray, shape (n_recorded,)
        Time of each recorded snapshot.

    See Also
    --------
    gpe_relax : Imaginary-time relaxation to a stationary state.
    physicskit.fields.solitons.nls_evolve : The 1D analog (opposite sign convention for ``g``).

    Examples
    --------
    Norm is conserved under real-time evolution, unlike the imaginary-time
    propagation in :func:`gpe_relax` (which requires explicit renormalization):

    >>> import numpy as np
    >>> n, length = 48, 12.0
    >>> X, Y, KX, KY, K2 = harmonic_trap_grid(n, length)
    >>> dxg = X[1, 0] - X[0, 0]
    >>> V = 0.5 * (X ** 2 + Y ** 2)
    >>> psi0 = np.exp(-0.5 * (X ** 2 + Y ** 2)).astype(complex)
    >>> psi0 *= 1.0 / np.sqrt(np.sum(np.abs(psi0) ** 2) * dxg * dxg)
    >>> snaps, times = gpe_evolve(psi0, V, g=2.0, dt=1e-3, steps=200, K2=K2, snapshot_stride=50)
    >>> norms = np.sum(np.abs(snaps) ** 2, axis=(1, 2)) * dxg * dxg
    >>> bool(np.max(np.abs(norms - norms[0])) < 1e-6)
    True
    """
    psi = np.array(psi0, dtype=complex, copy=True)
    lin_prop = np.exp(-1j * 0.5 * K2 * dt)
    n_snap = steps // snapshot_stride + 1
    snapshots = np.empty((n_snap, *psi.shape), dtype=complex)
    times = np.empty(n_snap)
    snapshots[0] = psi
    times[0] = 0.0
    idx = 1
    for step in range(1, steps + 1):
        psi_hat = np.fft.fft2(psi)
        psi_hat *= lin_prop
        psi = np.fft.ifft2(psi_hat)
        psi *= np.exp(-1j * (V + g * np.abs(psi) ** 2) * dt)
        if step % snapshot_stride == 0:
            snapshots[idx] = psi
            times[idx] = step * dt
            idx += 1
    return snapshots[:idx], times[:idx]


def gpe_energy(psi: np.ndarray, V: np.ndarray, g: float, X: np.ndarray, Y: np.ndarray, K2: np.ndarray) -> dict:
    """Evaluate the energy and angular momentum of a GPE wavefunction.

    Parameters
    ----------
    psi : ndarray of complex, shape (n, n)
        Wavefunction.
    V : ndarray, shape (n, n)
        External trapping potential.
    g : float
        Interaction strength.
    X, Y, K2 : ndarray, shape (n, n)
        Grids from :func:`harmonic_trap_grid` (``KX``, ``KY`` are not needed here).

    Returns
    -------
    dict
        ``{"kinetic", "potential", "interaction", "angular_momentum", "total"}``,
        where ``total`` is the lab-frame energy (kinetic + potential + interaction)
        and ``angular_momentum`` is :math:`\\langle L_z \\rangle`. The
        rotating-frame energy at rotation rate ``Omega`` is
        ``total - Omega * angular_momentum``.

    Examples
    --------
    >>> import numpy as np
    >>> X, Y, KX, KY, K2 = harmonic_trap_grid(48, 10.0)
    >>> V = 0.5 * (X ** 2 + Y ** 2)
    >>> psi = np.exp(-0.5 * (X ** 2 + Y ** 2)).astype(complex)
    >>> psi *= 1.0 / np.sqrt(np.sum(np.abs(psi) ** 2) * (X[1, 0] - X[0, 0]) ** 2)
    >>> E = gpe_energy(psi, V, g=0.0, X=X, Y=Y, K2=K2)
    >>> round(float(E["kinetic"] + E["potential"]), 4)
    1.0
    """
    dx = X[1, 0] - X[0, 0]
    n = X.shape[0]
    psi_hat = np.fft.fft2(psi)
    kinetic = np.sum(0.5 * K2 * np.abs(psi_hat) ** 2) / n**2 * dx * dx
    dens = np.abs(psi) ** 2
    potential = np.sum(V * dens) * dx * dx
    interaction = 0.5 * g * np.sum(dens**2) * dx * dx
    dpsidx = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dx)
    dpsidy = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dx)
    angular_momentum = float(np.real(np.sum(np.conj(psi) * (-1j) * (X * dpsidy - Y * dpsidx))) * dx * dx)
    total = kinetic + potential + interaction
    return {"kinetic": kinetic, "potential": potential, "interaction": interaction, "angular_momentum": angular_momentum, "total": total}


def count_vortices(psi: np.ndarray, density_threshold: float = 0.05) -> np.ndarray:
    """Locate quantized vortices by summing the phase winding around each grid plaquette.

    Plaquettes in very low density regions are excluded, since the phase
    of a near-zero-amplitude wavefunction is dominated by numerical noise
    and produces spurious windings there.

    A vortex core that sits exactly on a grid vertex (rather than strictly
    inside a plaquette) can go undetected, since its circulation is then
    split ambiguously between the four plaquettes touching that vertex; in
    practice this is avoided by seeding vortices at positions not exactly
    on the grid.

    Parameters
    ----------
    psi : ndarray of complex, shape (n, n)
        Wavefunction.
    density_threshold : float, default=0.05
        Plaquettes where :math:`|\\psi|^2` (relative to its maximum) falls
        below this fraction are excluded from the search.

    Returns
    -------
    ndarray of int, shape (n, n)
        Integer winding number around each plaquette's corner ``(i, j)``;
        nonzero entries mark a vortex core (``+1`` or ``-1`` for a
        singly-quantized vortex/antivortex) enclosed by that plaquette.

    Examples
    --------
    >>> import numpy as np
    >>> X, Y, KX, KY, K2 = harmonic_trap_grid(48, 10.0)
    >>> psi0 = np.exp(-0.5 * (X ** 2 + Y ** 2)).astype(complex)
    >>> psi_vortex = gpe_imprint_vortex(psi0, X, Y, [(0.0, 0.0)])
    >>> winding = count_vortices(psi_vortex)
    >>> int(np.sum(np.abs(winding)))
    1
    """
    phase = np.angle(psi)
    dphi_x = np.angle(np.exp(1j * (np.roll(phase, -1, axis=0) - phase)))
    dphi_y = np.angle(np.exp(1j * (np.roll(phase, -1, axis=1) - phase)))
    circ = dphi_x + np.roll(dphi_y, -1, axis=0) - np.roll(dphi_x, -1, axis=1) - dphi_y
    winding = np.round(circ / (2 * np.pi)).astype(int)
    dens = np.abs(psi) ** 2
    mask = dens < density_threshold * dens.max()
    winding[mask] = 0
    return winding


# -- QFT vacuum fluctuations / Casimir effect (1D toy cavity) ----------------
#
# A simplified, standard textbook treatment, not a 3D electromagnetic
# calculation: a scalar field confined between two parallel "plates" a
# distance d apart has only the discrete standing-wave modes
# omega_n = n*pi*c/d (n=1,2,...) rather than a continuum, and the
# difference between this discrete zero-point sum and the continuum it
# approaches at large d is the (finite, physical) Casimir energy. The bare
# sum (1/2) * sum_n omega_n diverges and is regularized here by an
# exponential (Abel-summation) cutoff, exp(-omega_n * cutoff), with
# cutoff -> 0 taken as a limit; the leading cutoff^-2 divergence is the same
# one a plate-free continuum calculation would produce with the same
# regulator, so it is subtracted exactly (via the known Laurent expansion of
# the regularized sum) rather than fit numerically, leaving the finite,
# cutoff-independent remainder that is the physical (measurable) energy.


def casimir_mode_frequencies(d: float, c: float = 1.0, n_max: int = 200) -> np.ndarray:
    """Discrete standing-wave mode frequencies of a 1D cavity of plate separation ``d``.

    Parameters
    ----------
    d : float
        Plate separation.
    c : float, default=1.0
        Wave speed (``=1`` in natural units; use the physical speed of
        light for SI-unit frequencies).
    n_max : int, default=200
        Number of modes to return.

    Returns
    -------
    ndarray, shape (n_max,)
        :math:`\\omega_n = n\\pi c/d` for :math:`n=1,\\dots,n_{max}`.

    See Also
    --------
    casimir_energy_1d : The regularized zero-point energy of this mode spectrum.

    Examples
    --------
    >>> omega = casimir_mode_frequencies(d=2.0, c=1.0, n_max=3)
    >>> [round(float(w), 4) for w in omega]
    [1.5708, 3.1416, 4.7124]
    """
    n = np.arange(1, n_max + 1)
    return n * np.pi * c / d


def casimir_energy_1d(d: float, c: float = 1.0, cutoff: float | None = None) -> float:
    """Regularized zero-point (Casimir) energy of the discrete 1D cavity spectrum.

    Uses the standard exponential-cutoff regularization: the sum
    :math:`\\sum_n n x^n = x/(1-x)^2` (with :math:`x=e^{-a}`,
    :math:`a=\\pi c\\,\\text{cutoff}/d`) has an exact closed form, so no
    series truncation is needed. Its small-``cutoff`` (small-``a``)
    expansion is :math:`\\tfrac{1}{4\\sinh^2(a/2)} = 1/a^2 - 1/12 + O(a^2)`;
    the :math:`1/a^2` piece is the (unphysical, cutoff-scheme-dependent)
    divergence that a continuum reference calculation regularized the same
    way would also produce, so it is subtracted exactly, leaving the finite
    remainder that survives as ``cutoff -> 0``:

    .. math::

        E(d) \\to -\\frac{\\pi c}{24 d}

    the standard 1D massless-field Casimir energy (equivalently, the
    Casimir energy of a CFT strip with central charge 1).

    Parameters
    ----------
    d : float
        Plate separation.
    c : float, default=1.0
        Wave speed.
    cutoff : float, optional
        Regulator scale. Defaults to ``1e-3 * d / c``. This subtraction is
        a difference of two large, nearly-equal floating-point terms
        (:math:`\\propto 1/\\text{cutoff}^2`), so making ``cutoff`` too
        small *loses* precision to cancellation rather than gaining it;
        ``1e-4*d/c`` to ``1e-2*d/c`` is the well-behaved range in double
        precision.

    Returns
    -------
    float
        The regularized (finite, cutoff-subtracted) zero-point energy.

    See Also
    --------
    casimir_mode_frequencies : The (bare, un-regularized) mode spectrum being summed.

    Examples
    --------
    The cutoff-and-subtract result agrees with the known closed form
    :math:`-\\pi c/(24d)`, and (within the well-behaved cutoff range noted
    above) agrees more closely as the cutoff shrinks:

    >>> d = 3.0
    >>> exact = -np.pi * 1.0 / (24 * d)
    >>> bool(abs(casimir_energy_1d(d, cutoff=3e-3 * d) - exact) < 1e-6)
    True
    >>> bool(abs(casimir_energy_1d(d, cutoff=7e-4 * d) - exact) < 1e-8)
    True

    The energy grows less negative (weaker confinement of vacuum energy)
    as the plates separate, giving an attractive force :math:`-dE/dd < 0`:

    >>> bool(casimir_energy_1d(1.0) < casimir_energy_1d(2.0) < 0)
    True
    """
    if cutoff is None:
        cutoff = 1e-3 * d / c
    a = np.pi * c * cutoff / d
    x = np.exp(-a)
    E_reg = (np.pi * c) / (2 * d) * x / (1 - x) ** 2
    divergent = d / (2 * np.pi * c * cutoff**2)
    return float(E_reg - divergent)
