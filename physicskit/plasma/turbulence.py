"""Drift-wave turbulence: a reduced Hasegawa-Mima model.

Gyrokinetic drift-wave turbulence -- the dominant cross-field transport
mechanism in magnetized fusion plasmas -- is normally studied with a
gyrokinetic or gyrofluid code evolving several coupled moments (density,
parallel flow, temperature, ...) along and across the field. The standard
first reduction of that problem, capturing its essential nonlinear
saturation mechanism while remaining a single scalar field, is the
Hasegawa-Mima equation (Hasegawa & Mima, 1977) for the normalized
electrostatic potential :math:`\\phi`:

.. math::

   \\partial_t(\\phi - \\nabla^2\\phi) + \\{\\phi, \\nabla^2\\phi\\} + \\partial_y\\phi = 0,

where :math:`\\{\\phi,\\zeta\\}=(\\partial_x\\phi)(\\partial_y\\zeta)-(\\partial_y\\phi)(\\partial_x\\zeta)`
is the :math:`\\mathbf{E}\\times\\mathbf{B}` advection of the potential
vorticity :math:`q=\\nabla^2\\phi-\\phi` by the electrostatic drift velocity
:math:`\\mathbf{v}_E=\\hat{z}\\times\\nabla\\phi`, and the linear
:math:`\\partial_y\\phi` term is the background density-gradient drift wave
(:math:`y` is the direction of the electron diamagnetic drift). This is a
genuine simplification of full gyrokinetics -- it drops magnetic-field-line
bending, parallel electron dynamics (adiabatic-electron closure), and
temperature-gradient drive -- but it is the standard reduced model used
throughout the drift-wave-turbulence literature to demonstrate the basic
phenomenology: small-amplitude drift-wave noise nonlinearly cascading into
a turbulent field of vortices ("blobs") via the same kind of :math:`E\\times B`
advection that governs 2D Navier-Stokes turbulence.

The pseudo-spectral solver here is self-contained (built directly on
``numpy.fft`` rather than importing :mod:`physicskit.fluids`'s spectral-grid
machinery) so that :mod:`physicskit.plasma` has no cross-package dependency
on :mod:`physicskit.fluids`, even though the underlying numerics -- an
elliptic inversion for a streamfunction-like potential, followed by
pseudo-spectral advection of a vorticity-like field -- are structurally the
same idea used there for 2D Navier-Stokes.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "drift_wave_noise_ic",
    "hasegawa_mima_rhs",
    "simulate_hasegawa_mima",
]


def drift_wave_noise_ic(n: int, length: float, amplitude: float, seed: int = 0) -> np.ndarray:
    """Small-amplitude random-phase noise initial condition for Hasegawa-Mima drift-wave turbulence.

    A featureless, isotropic random field with the specified root-mean-square
    amplitude -- the standard "let it find its own structure" initial
    condition for turbulence simulations, in place of a specific unstable
    linear eigenmode, since real drift-wave turbulence in a fusion device is
    continuously driven by many linearly unstable modes simultaneously
    rather than growing from one clean seed.

    Parameters
    ----------
    n : int
        Number of grid points along each axis.
    length : float
        Physical domain size. Unused, since the noise is independent per
        grid point (white) and so has no length scale; kept for signature
        symmetry with :func:`simulate_hasegawa_mima`.
    amplitude : float
        Root-mean-square amplitude of the seeded potential noise.
    seed : int, default=0
        Random seed.

    Returns
    -------
    ndarray, shape (n, n)
        Initial potential :math:`\\phi(x, y)`.

    See Also
    --------
    simulate_hasegawa_mima : Evolves this initial condition forward in time.

    Examples
    --------
    >>> phi0 = drift_wave_noise_ic(64, 2 * 3.141592653589793, amplitude=0.01, seed=0)
    >>> phi0.shape
    (64, 64)
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (n, n))
    noise -= noise.mean()
    rms = np.sqrt(np.mean(noise**2))
    return amplitude * noise / rms


def _spectral_grid(n: int, length: float) -> tuple:
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    return KX, KY, K2


def hasegawa_mima_rhs(q: np.ndarray, KX: np.ndarray, KY: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Non-dissipative right-hand side of the Hasegawa-Mima potential-vorticity equation, evaluated pseudo-spectrally.

    Evolves the potential vorticity :math:`q=\\nabla^2\\phi-\\phi` (from
    which :math:`\\phi` is recovered via :math:`\\hat{\\phi}=-\\hat{q}/(1+k^2)`,
    the elliptic inversion analogous to the streamfunction Poisson solve of
    2D Navier-Stokes) under :math:`\\mathbf{E}\\times\\mathbf{B}` advection by
    the drift velocity :math:`(u,v)=(-\\partial_y\\phi,\\partial_x\\phi)` and
    the linear drift-wave term :math:`-\\partial_y\\phi`. Excludes the
    dissipative regularization :func:`simulate_hasegawa_mima` adds on top,
    which -- being stiffer at grid scale than this advective term -- is
    integrated separately via an exact integrating factor rather than
    folded into this explicit right-hand side.

    Parameters
    ----------
    q : ndarray, shape (n, n)
        Potential vorticity :math:`q = \\nabla^2\\phi - \\phi`.
    KX, KY, K2 : ndarray, shape (n, n)
        Wavenumber grids from :func:`_spectral_grid`.

    Returns
    -------
    ndarray, shape (n, n)
        :math:`dq/dt`, excluding dissipation.
    """
    q_hat = np.fft.fft2(q)
    phi_hat = -q_hat / (1.0 + K2)
    u = np.real(np.fft.ifft2(-1j * KY * phi_hat))
    v = np.real(np.fft.ifft2(1j * KX * phi_hat))
    dq_dx = np.real(np.fft.ifft2(1j * KX * q_hat))
    dq_dy = np.real(np.fft.ifft2(1j * KY * q_hat))
    dphi_dy = np.real(np.fft.ifft2(1j * KY * phi_hat))
    advection = -(u * dq_dx + v * dq_dy)
    drift = -dphi_dy
    return advection + drift


def simulate_hasegawa_mima(phi0: np.ndarray, dt: float, steps: int, length: float, nu: float = 0.03) -> dict:
    """Time-step the Hasegawa-Mima drift-wave-turbulence equation with pseudo-spectral RK4.

    Starting from a potential field :math:`\\phi_0` (e.g. small-amplitude
    noise from :func:`drift_wave_noise_ic`), converts to the potential
    vorticity :math:`q=\\nabla^2\\phi-\\phi` and advances it one step at a
    time: the non-stiff advection and linear drift-wave terms
    (:func:`hasegawa_mima_rhs`) with explicit RK4, Strang-split around an
    *exact* diffusive decay :math:`\\hat{q}\\mathrel{*}=e^{-\\nu k^2 dt/2}`
    applied before and after -- the same splitting idea
    :mod:`physicskit.fields.solitons`'s KdV solver uses for its stiff
    dispersive term, applied here because a small Laplacian-type
    dissipation :math:`-\\nu\\nabla^2 q` (not part of the ideal
    Hasegawa-Mima equation) is needed to drain enstrophy piling up at the
    grid scale once the flow turns turbulent, and integrating it exactly
    avoids the explicit-RK4 stability restriction that treating it as an
    ordinary right-hand-side term would impose. Reproduces the
    characteristic Hasegawa-Mima phenomenology of small-scale drift waves
    nonlinearly steepening and merging into a field of long-lived coherent
    vortices ("turbulent blobs") that then dominates the cross-field
    transport, in analogy to two-dimensional Navier-Stokes turbulence's
    inverse energy cascade.

    Parameters
    ----------
    phi0 : ndarray, shape (n, n)
        Initial electrostatic potential on a doubly periodic ``[0, length)^2`` domain.
    dt : float
        Time step.
    steps : int
        Number of RK4 steps to advance.
    length : float
        Physical domain size.
    nu : float, default=0.03
        Dissipation coefficient (see above); purely a numerical
        stabilizer, not part of the ideal physics. Too small a value lets
        the (undealiased) pseudo-spectral nonlinear term blow up once the
        flow turns turbulent; the default was chosen to remain stable for
        the noise amplitudes and grid resolutions used throughout this
        module's examples and tests.

    Returns
    -------
    dict
        ``{"phi": final potential, "q": final potential vorticity}``.

    See Also
    --------
    drift_wave_noise_ic : Builds a typical initial condition consumed here.
    hasegawa_mima_rhs : The single-step right-hand side repeated here.

    Examples
    --------
    >>> import numpy as np
    >>> phi0 = drift_wave_noise_ic(48, 2 * np.pi, amplitude=0.01, seed=0)
    >>> result = simulate_hasegawa_mima(phi0, dt=0.02, steps=20, length=2 * np.pi)
    >>> result["phi"].shape
    (48, 48)
    >>> bool(np.all(np.isfinite(result["phi"])))
    True
    """
    n = phi0.shape[0]
    KX, KY, K2 = _spectral_grid(n, length)
    phi0_hat = np.fft.fft2(phi0)
    q_hat = -(1.0 + K2) * phi0_hat
    decay = np.exp(-nu * K2 * dt / 2.0)
    for _ in range(steps):
        q_hat = q_hat * decay
        q = np.real(np.fft.ifft2(q_hat))
        k1 = np.fft.fft2(hasegawa_mima_rhs(q, KX, KY, K2))
        q1 = np.real(np.fft.ifft2(q_hat + dt / 2 * k1))
        k2 = np.fft.fft2(hasegawa_mima_rhs(q1, KX, KY, K2))
        q2 = np.real(np.fft.ifft2(q_hat + dt / 2 * k2))
        k3 = np.fft.fft2(hasegawa_mima_rhs(q2, KX, KY, K2))
        q3 = np.real(np.fft.ifft2(q_hat + dt * k3))
        k4 = np.fft.fft2(hasegawa_mima_rhs(q3, KX, KY, K2))
        q_hat = q_hat + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q_hat = q_hat * decay
    q = np.real(np.fft.ifft2(q_hat))
    phi_hat = -q_hat / (1.0 + K2)
    phi = np.real(np.fft.ifft2(phi_hat))
    return {"phi": phi, "q": q}
