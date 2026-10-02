"""Rayleigh-Bénard convection: the onset of convection in a layer heated from below.

A fluid layer of depth :math:`d` with temperature difference :math:`\\Delta T`
across it starts to convect when the Rayleigh number

.. math::

    Ra = \\frac{g\\alpha\\Delta T d^3}{\\nu\\kappa}

exceeds a critical value set by the boundaries (Rayleigh, Phil. Mag. 32,
529 (1916)). In units of the depth and the thermal diffusion time
:math:`d^2/\\kappa`, the Boussinesq equations for the temperature
perturbation :math:`\\theta` about the linear conduction profile are

.. math::

    \\partial_t\\omega + (\\mathbf u\\cdot\\nabla)\\omega = Pr\\,\\nabla^2\\omega + Pr\\,Ra\\,\\partial_x\\theta,
    \\qquad
    \\partial_t\\theta + (\\mathbf u\\cdot\\nabla)\\theta = w + \\nabla^2\\theta,

with :math:`\\omega = \\partial_x w - \\partial_z u` and
:math:`Pr = \\nu/\\kappa`. Linearizing about rest with
:math:`w \\propto W(z)e^{iax + \\sigma t}` gives

.. math::

    (D^2 - a^2)^3 W = -Ra\\,a^2 W \\quad (\\sigma = 0).

Between stress-free walls :math:`W = \\sin(\\pi z)` solves this exactly:
:math:`Ra(a) = (\\pi^2 + a^2)^3 / a^2`, minimal at :math:`a_c = \\pi/\\sqrt2`
with :math:`Ra_c = 27\\pi^4/4 \\approx 657.5`. Between rigid (no-slip)
walls the onset rises to :math:`Ra_c \\approx 1707.76` at :math:`a_c
\\approx 3.117` (Jeffreys 1928; Pellew and Southwell, Proc. R. Soc. A 176,
312 (1940); Chandrasekhar, *Hydrodynamic and Hydromagnetic Stability*,
1961).

:func:`rayleigh_benard_critical` solves the linear problem for either
boundary condition (exactly, from Chandrasekhar's characteristic
determinant for rigid walls), and :class:`RayleighBenard2D` integrates the full
nonlinear equations between stress-free walls on the doubly periodic
pseudo-spectral grid, by extending the layer to an odd-symmetric, periodic
one of twice the depth. :class:`RayleighBenardWalls2D` does the same between
rigid (no-slip) or stress-free walls, with a Chebyshev expansion in depth,
and its simulated onset reproduces both critical values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "RayleighBenard2D",
    "RayleighBenardWalls2D",
    "rayleigh_benard_critical",
    "rayleigh_benard_growth_rate_free",
    "rayleigh_benard_marginal_rayleigh",
]


def _rigid_determinant(Ra: float, a: float) -> float:
    # even mode W = sum_j A_j cosh(q_j z) on -1/2 < z < 1/2, with q_j^2 = a^2 + lambda_j and
    # lambda_j^3 = -Ra a^2; W = W' = (D^2 - a^2)^2 W = 0 at z = 1/2. Columns j = 1, 2 are complex
    # conjugates, so the determinant is purely imaginary.
    tau = (Ra * a * a) ** (1.0 / 3.0)
    lam = tau * np.array([-1.0, np.exp(1j * np.pi / 3), np.exp(-1j * np.pi / 3)])
    q = np.sqrt(a * a + lam + 0j)
    M = np.array([np.cosh(q / 2), q * np.sinh(q / 2), lam**2 * np.cosh(q / 2)])
    return float(np.linalg.det(M).imag)


def rayleigh_benard_marginal_rayleigh(a: float, boundaries: str = "rigid") -> float:
    """Rayleigh number at which the mode of horizontal wavenumber :math:`a` is marginally stable.

    For ``"free"`` (stress-free) walls this is :math:`(\\pi^2 + a^2)^3/a^2`.
    For ``"rigid"`` (no-slip) walls the lowest, even mode is
    :math:`W = \\sum_{j=0}^2 A_j \\cosh(q_j z)` on :math:`-\\tfrac12 < z < \\tfrac12`,
    with :math:`(q_j^2 - a^2)^3 = -Ra\\,a^2`, and :math:`Ra` is the smallest
    root of the :math:`3\\times3` determinant imposing :math:`W = W' =
    \\theta = 0` at the wall (Chandrasekhar 1961, sec. 15).

    Parameters
    ----------
    a : float
        Horizontal wavenumber (in units of :math:`1/d`).
    boundaries : {"rigid", "free"}, default "rigid"

    Returns
    -------
    float

    Examples
    --------
    >>> round(rayleigh_benard_marginal_rayleigh(3.117), 1)
    1707.8
    """
    if boundaries == "free":
        return float((np.pi**2 + a**2) ** 3 / a**2)
    if boundaries != "rigid":
        raise InvalidParameterError("boundaries must be 'rigid' or 'free'")
    Ra_grid = np.geomspace(100.0, 1e6, 600)
    values = np.array([_rigid_determinant(Ra, a) for Ra in Ra_grid])
    change = np.nonzero(np.sign(values[:-1]) != np.sign(values[1:]))[0]
    if change.size == 0:
        raise InvalidParameterError(f"no marginal Rayleigh number below 1e6 for a = {a}")
    i = change[0]
    return float(brentq(_rigid_determinant, Ra_grid[i], Ra_grid[i + 1], args=(a,), xtol=1e-10))


def rayleigh_benard_critical(boundaries: str = "rigid") -> tuple[float, float]:
    """Critical Rayleigh number and wavenumber for the onset of convection.

    Minimizes :func:`rayleigh_benard_marginal_rayleigh` over :math:`a`.

    Parameters
    ----------
    boundaries : {"rigid", "free"}, default "rigid"
        No-slip or stress-free horizontal walls (both isothermal).

    Returns
    -------
    Ra_c, a_c : float
        :math:`(27\\pi^4/4, \\pi/\\sqrt2)` exactly for ``"free"``; about
        ``(1707.76, 3.117)`` for ``"rigid"``.

    Examples
    --------
    >>> Ra_c, a_c = rayleigh_benard_critical("rigid")
    >>> round(Ra_c, 2), round(a_c, 3)
    (1707.76, 3.116)
    """
    if boundaries == "free":
        return 27.0 * np.pi**4 / 4.0, np.pi / np.sqrt(2.0)
    res = minimize_scalar(lambda a: rayleigh_benard_marginal_rayleigh(a, boundaries), bounds=(2.0, 4.5), method="bounded", options={"xatol": 1e-7})
    return float(res.fun), float(res.x)


def rayleigh_benard_growth_rate_free(Ra: float, a: float, Pr: float = 1.0, m: int = 1) -> float:
    """Exact linear growth rate of the stress-free mode :math:`\\sin(m\\pi z)e^{iax}`.

    With :math:`q^2 = m^2\\pi^2 + a^2`, the growth rate is the larger root of

    .. math::

        \\sigma^2 + (1 + Pr)q^2\\sigma + Pr\\left(q^4 - Ra\\,\\frac{a^2}{q^2}\\right) = 0.

    Parameters
    ----------
    Ra : float
    a : float
        Horizontal wavenumber.
    Pr : float, default 1.0
    m : int, default 1
        Vertical mode number.

    Returns
    -------
    float
        Growth rate, in units of :math:`\\kappa/d^2`; zero at the marginal
        :math:`Ra = q^6/a^2`.

    Examples
    --------
    >>> a = np.pi / np.sqrt(2)
    >>> abs(rayleigh_benard_growth_rate_free(27 * np.pi**4 / 4, a)) < 1e-9
    True
    """
    q2 = (m * np.pi) ** 2 + a**2
    b = (1.0 + Pr) * q2
    c = Pr * (q2**2 - Ra * a**2 / q2)
    return float((-b + np.sqrt(b * b - 4.0 * c)) / 2.0)


class RayleighBenard2D:
    """Nonlinear 2D Rayleigh-Bénard convection between stress-free, isothermal walls.

    The layer :math:`0 < z < 1` is extended to :math:`-1 < z < 1` with
    :math:`\\theta`, :math:`w` and :math:`\\omega` odd in :math:`z`, which
    satisfies :math:`w = \\theta = \\partial_z u = 0` at both walls and makes
    the problem periodic in :math:`z` with period 2. It is then solved
    pseudo-spectrally like :class:`~physicskit.fluids.systems.navier_stokes.NavierStokes2D`,
    with 2/3-rule dealiasing, an exact integrating factor for diffusion,
    RK4 for the remaining terms, and the odd symmetry re-imposed at every
    sample.

    Parameters
    ----------
    Ra : float
        Rayleigh number.
    Pr : float, default 1.0
        Prandtl number.
    aspect : float, default ``2*sqrt(2)``
        Box width over depth; the default fits one critical wavelength
        :math:`2\\pi/a_c` of the stress-free problem.
    nx, nz : int, default 32
        Grid points in :math:`x` and over the doubled depth.

    Attributes
    ----------
    theta, omega : ndarray of float, shape (nz, nx)
        Temperature perturbation and vorticity on the doubled domain.
    x, z : ndarray of float
        Grid coordinates; :math:`z \\in [-1, 1)`.

    Examples
    --------
    >>> rb = RayleighBenard2D(Ra=2000.0, nx=16, nz=16)
    >>> rb.seed_mode(amplitude=1e-3)
    >>> out = rb.run(t_max=0.2, dt=1e-3)
    >>> bool(out["kinetic_energy"][-1] > out["kinetic_energy"][0])  # supercritical: grows
    True
    """

    def __init__(self, Ra: float, Pr: float = 1.0, aspect: float = 2.0 * np.sqrt(2.0), nx: int = 32, nz: int = 32):
        if Ra < 0 or Pr <= 0 or aspect <= 0:
            raise InvalidParameterError("need Ra >= 0, Pr > 0 and aspect > 0")
        self.Ra = float(Ra)
        self.Pr = float(Pr)
        self.aspect = float(aspect)
        self.nx = int(nx)
        self.nz = int(nz)
        self.x = np.linspace(0.0, self.aspect, self.nx, endpoint=False)
        self.z = np.linspace(-1.0, 1.0, self.nz, endpoint=False)
        self.Z, self.X = np.meshgrid(self.z, self.x, indexing="ij")
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.aspect / self.nx)
        kz = 2 * np.pi * np.fft.fftfreq(self.nz, d=2.0 / self.nz)
        self.KZ, self.KX = np.meshgrid(kz, kx, indexing="ij")
        self.K2 = self.KX**2 + self.KZ**2
        self._K2safe = self.K2.copy()
        self._K2safe[0, 0] = 1.0
        self._dealias = (np.abs(self.KX) < np.abs(kx).max() * 2 / 3) & (np.abs(self.KZ) < np.abs(kz).max() * 2 / 3)
        self._mirror = (-np.arange(self.nz)) % self.nz
        self.theta = np.zeros((self.nz, self.nx))
        self.omega = np.zeros((self.nz, self.nx))

    def _odd(self, f: NDArray[np.float64]) -> NDArray[np.float64]:
        return 0.5 * (f - f[self._mirror, :])

    def seed_mode(self, amplitude: float = 1e-3, noise: float = 0.0, seed: int | None = None) -> None:
        """Set :math:`\\theta = A\\sin(\\pi z)\\cos(2\\pi x/\\Gamma)` plus optional odd noise, at rest.

        Parameters
        ----------
        amplitude : float, default 1e-3
        noise : float, default 0.0
            Amplitude of additional random temperature perturbations.
        seed : int, optional
        """
        self.theta = amplitude * np.sin(np.pi * self.Z) * np.cos(2 * np.pi * self.X / self.aspect)
        if noise:
            self.theta = self.theta + self._odd(noise * np.random.default_rng(seed).standard_normal(self.theta.shape))
        self.omega = np.zeros_like(self.theta)

    def velocity(self, omega: NDArray[np.float64] | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Velocity :math:`(u, w)` from the vorticity, via :math:`\\nabla^2\\psi = -\\omega`, :math:`u = \\partial_z\\psi`, :math:`w = -\\partial_x\\psi`."""
        oh = np.fft.fft2(self.omega if omega is None else omega)
        psih = oh / self._K2safe
        psih[0, 0] = 0.0
        u = np.real(np.fft.ifft2(1j * self.KZ * psih))
        w = np.real(np.fft.ifft2(-1j * self.KX * psih))
        return u, w

    def _explicit(self, oh: NDArray[np.complex128], th: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        # advection, buoyancy torque and the background-gradient source w, in Fourier space
        psih = oh / self._K2safe
        psih[0, 0] = 0.0
        wh = -1j * self.KX * psih
        u = np.real(np.fft.ifft2(1j * self.KZ * psih))
        w = np.real(np.fft.ifft2(wh))
        ox = np.real(np.fft.ifft2(1j * self.KX * oh))
        oz = np.real(np.fft.ifft2(1j * self.KZ * oh))
        tx = np.real(np.fft.ifft2(1j * self.KX * th))
        tz = np.real(np.fft.ifft2(1j * self.KZ * th))
        do = -np.fft.fft2(u * ox + w * oz) * self._dealias + self.Pr * self.Ra * 1j * self.KX * th
        dt_ = -np.fft.fft2(u * tx + w * tz) * self._dealias + wh
        return do, dt_

    def nusselt(self) -> float:
        """Nusselt number :math:`Nu = 1 + \\langle w\\theta\\rangle` (conductive heat flux is 1)."""
        _, w = self.velocity()
        return float(1.0 + np.mean(w * self.theta))

    def kinetic_energy(self) -> float:
        """Mean kinetic energy :math:`\\tfrac12\\langle u^2 + w^2\\rangle`."""
        u, w = self.velocity()
        return float(0.5 * np.mean(u**2 + w**2))

    def run(self, t_max: float, dt: float = 1e-3, sample_every: int = 10) -> dict[str, NDArray[np.float64]]:
        """Advance the convection with RK4.

        Parameters
        ----------
        t_max : float
            Time to integrate, in thermal diffusion times :math:`d^2/\\kappa`.
        dt : float, default 1e-3
        sample_every : int, default 10
            Steps between samples.

        Returns
        -------
        dict
            Sampled ``"t"``, ``"kinetic_energy"``, ``"nusselt"`` and the
            amplitude ``"theta_mode"`` of the seeded :math:`\\sin(\\pi z)
            \\cos(2\\pi x/\\Gamma)` mode.
        """
        n_steps = int(round(t_max / dt))
        mode = np.sin(np.pi * self.Z) * np.cos(2 * np.pi * self.X / self.aspect)
        mode_norm = np.sum(mode**2)
        t, ke, nu, amp = [0.0], [self.kinetic_energy()], [self.nusselt()], [np.sum(self.theta * mode) / mode_norm]
        # integrating factor: diffusion exactly, the rest by RK4
        Eo, Eo2 = np.exp(-self.Pr * self.K2 * dt), np.exp(-self.Pr * self.K2 * dt / 2)
        Et, Et2 = np.exp(-self.K2 * dt), np.exp(-self.K2 * dt / 2)
        oh, th = np.fft.fft2(self.omega), np.fft.fft2(self.theta)
        for step in range(1, n_steps + 1):
            a_o, a_t = self._explicit(oh, th)
            b_o, b_t = self._explicit(Eo2 * (oh + dt / 2 * a_o), Et2 * (th + dt / 2 * a_t))
            c_o, c_t = self._explicit(Eo2 * oh + dt / 2 * b_o, Et2 * th + dt / 2 * b_t)
            d_o, d_t = self._explicit(Eo * oh + dt * Eo2 * c_o, Et * th + dt * Et2 * c_t)
            oh = Eo * oh + dt / 6 * (Eo * a_o + 2 * Eo2 * (b_o + c_o) + d_o)
            th = Et * th + dt / 6 * (Et * a_t + 2 * Et2 * (b_t + c_t) + d_t)
            if step % sample_every == 0 or step == n_steps:
                # re-impose the odd symmetry that roundoff slowly breaks
                self.omega = self._odd(np.real(np.fft.ifft2(oh)))
                self.theta = self._odd(np.real(np.fft.ifft2(th)))
                oh, th = np.fft.fft2(self.omega), np.fft.fft2(self.theta)
            if step % sample_every == 0:
                t.append(step * dt)
                ke.append(self.kinetic_energy())
                nu.append(self.nusselt())
                amp.append(np.sum(self.theta * mode) / mode_norm)
        return {"t": np.array(t), "kinetic_energy": np.array(ke), "nusselt": np.array(nu), "theta_mode": np.array(amp)}


def _chebyshev(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Gauss-Lobatto points x_j = cos(pi j / n) and differentiation matrix (Trefethen, Spectral Methods in MATLAB, ch. 6)
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(n + 1)
    dX = x[:, None] - x[None, :]
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(axis=1))
    return x, D


class RayleighBenardWalls2D:
    """2D Rayleigh-Bénard convection between rigid or stress-free walls, Fourier-Chebyshev.

    The streamfunction :math:`\\psi` (:math:`u = \\partial_z\\psi`,
    :math:`w = -\\partial_x\\psi`, vorticity :math:`\\omega = -\\nabla^2\\psi`) and
    the temperature perturbation :math:`\\theta` are expanded in Fourier modes
    :math:`e^{ikx}` across a periodic box and collocated on Chebyshev points
    in :math:`0 \\le z \\le 1`, where

    .. math::

        \\partial_t\\nabla^2\\psi = Pr\\,\\nabla^4\\psi - Pr\\,Ra\\,\\partial_x\\theta
            + (\\mathbf u\\cdot\\nabla)\\omega,
        \\qquad
        \\partial_t\\theta = \\nabla^2\\theta + w - (\\mathbf u\\cdot\\nabla)\\theta.

    Diffusion is treated implicitly (Crank-Nicolson) and the rest explicitly
    (second-order Adams-Bashforth). The walls are isothermal,
    :math:`\\theta = 0`, impermeable, :math:`\\psi = 0`, and either no-slip,
    :math:`\\partial_z\\psi = 0` (``"rigid"``), or stress-free,
    :math:`\\partial_z^2\\psi = 0` (``"free"``), imposed by replacing the
    collocation equations at the two points next to each wall (Canuto et al.,
    *Spectral Methods*, 2006, sec. 3.7).

    Parameters
    ----------
    Ra : float
    Pr : float, default 1.0
    aspect : float, optional
        Box width; defaults to one critical wavelength :math:`2\\pi/a_c` for
        the chosen walls.
    boundaries : {"rigid", "free"}, default "rigid"
    nx : int, default 32
        Fourier points across the box.
    nz : int, default 24
        Chebyshev polynomial degree (``nz + 1`` points).

    Examples
    --------
    >>> rb = RayleighBenardWalls2D(Ra=2500.0, nx=16, nz=16)
    >>> rb.seed_mode(1e-4)
    >>> out = rb.run(t_max=0.3, dt=2e-3)
    >>> bool(out["theta_mode"][-1] > out["theta_mode"][0])  # above 1708: grows
    True
    """

    def __init__(self, Ra: float, Pr: float = 1.0, aspect: float | None = None, boundaries: str = "rigid", nx: int = 32, nz: int = 24):
        if boundaries not in ("rigid", "free"):
            raise InvalidParameterError("boundaries must be 'rigid' or 'free'")
        if Ra < 0 or Pr <= 0:
            raise InvalidParameterError("need Ra >= 0 and Pr > 0")
        self.Ra, self.Pr, self.boundaries = float(Ra), float(Pr), boundaries
        a_c = rayleigh_benard_critical(boundaries)[1]
        self.aspect = 2 * np.pi / a_c if aspect is None else float(aspect)
        self.nx, self.nz = int(nx), int(nz)
        xc, Dc = _chebyshev(self.nz)
        self.z = (xc + 1.0) / 2.0  # z = 1 at index 0, z = 0 at index nz
        self.D = 2.0 * Dc
        self.D2 = self.D @ self.D
        self.x = np.linspace(0.0, self.aspect, self.nx, endpoint=False)
        self.k = 2 * np.pi * np.fft.rfftfreq(self.nx, d=self.aspect / self.nx)
        self._dealias = np.abs(self.k) < (2.0 / 3.0) * np.abs(self.k).max() + 1e-12
        self.psi = np.zeros((self.k.size, self.nz + 1), dtype=complex)
        self.theta = np.zeros((self.k.size, self.nz + 1), dtype=complex)
        self._dt: float | None = None

    def _setup(self, dt: float) -> None:
        I = np.eye(self.nz + 1)
        D, D2 = self.D, self.D2
        self._L2, self._L4, self._lhs_psi, self._lhs_theta = [], [], [], []
        bc_row = D if self.boundaries == "rigid" else D2
        for k in self.k:
            L2 = D2 - k * k * I
            L4 = L2 @ L2
            A = L2 - 0.5 * dt * self.Pr * L4
            A[0], A[-1] = I[0], I[-1]  # psi = 0
            A[1], A[-2] = bc_row[0], bc_row[-1]  # dpsi/dz = 0 or d2psi/dz2 = 0
            B = I - 0.5 * dt * L2
            B[0], B[-1] = I[0], I[-1]  # theta = 0
            self._L2.append(L2)
            self._L4.append(L4)
            self._lhs_psi.append(np.linalg.inv(A))
            self._lhs_theta.append(np.linalg.inv(B))
        self._dt = dt

    def seed_mode(self, amplitude: float = 1e-3) -> None:
        """Set :math:`\\theta = A\\sin(\\pi z)\\cos(2\\pi x/\\Gamma)` with the fluid at rest."""
        self.psi[:] = 0.0
        self.theta[:] = 0.0
        self.theta[1] = 0.5 * self.nx * amplitude * np.sin(np.pi * self.z)

    def _physical(self, f_hat):
        return np.fft.irfft(f_hat, n=self.nx, axis=0)

    def fields(self) -> dict[str, NDArray[np.float64]]:
        """Real-space ``"u"``, ``"w"``, ``"theta"`` and ``"omega"``, shape ``(nx, nz + 1)``."""
        ik = 1j * self.k[:, None]
        omega = -(self.psi @ self.D2.T - (self.k**2)[:, None] * self.psi)
        return {
            "u": self._physical(self.psi @ self.D.T),
            "w": self._physical(-ik * self.psi),
            "theta": self._physical(self.theta),
            "omega": self._physical(omega),
        }

    def _explicit(self) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        ik = 1j * self.k[:, None]
        omega_hat = -(self.psi @ self.D2.T - (self.k**2)[:, None] * self.psi)
        u = self._physical(self.psi @ self.D.T)
        w = self._physical(-ik * self.psi)
        adv_omega = u * self._physical(ik * omega_hat) + w * self._physical(omega_hat @ self.D.T)
        adv_theta = u * self._physical(ik * self.theta) + w * self._physical(self.theta @ self.D.T)
        cut = self._dealias[:, None]
        N_psi = np.fft.rfft(adv_omega, axis=0) * cut - self.Pr * self.Ra * ik * self.theta
        N_theta = -np.fft.rfft(adv_theta, axis=0) * cut - ik * self.psi
        return N_psi, N_theta

    def nusselt(self) -> float:
        """Nusselt number :math:`1 + \\langle w\\theta\\rangle`, averaged over the layer (Clenshaw-Curtis)."""
        f = self.fields()
        flux = np.mean(f["w"] * f["theta"], axis=0)
        # trapezoid on the nonuniform Chebyshev grid is accurate enough for a smooth flux profile
        return float(1.0 + np.trapezoid(flux[::-1], self.z[::-1]))

    def run(self, t_max: float, dt: float = 2e-3, sample_every: int = 10) -> dict[str, NDArray[np.float64]]:
        """Advance with CN-AB2.

        Returns
        -------
        dict
            Sampled ``"t"``, ``"nusselt"`` and ``"theta_mode"``, the amplitude of
            the first Fourier mode of :math:`\\theta` at mid-depth.
        """
        if self._dt != dt:
            self._setup(dt)
        mid = int(np.argmin(np.abs(self.z - 0.5)))
        n_steps = int(round(t_max / dt))
        t, nu, amp = [0.0], [self.nusselt()], [abs(self.theta[1, mid]) * 2 / self.nx]
        prev = None
        for step in range(1, n_steps + 1):
            N_psi, N_theta = self._explicit()
            if prev is None:
                ex_psi, ex_theta = N_psi, N_theta
            else:
                ex_psi, ex_theta = 1.5 * N_psi - 0.5 * prev[0], 1.5 * N_theta - 0.5 * prev[1]
            prev = (N_psi, N_theta)
            for i in range(self.k.size):
                rhs = self.psi[i] @ (self._L2[i] + 0.5 * dt * self.Pr * self._L4[i]).T + dt * ex_psi[i]
                rhs[[0, 1, -2, -1]] = 0.0
                self.psi[i] = self._lhs_psi[i] @ rhs
                rhs = self.theta[i] + 0.5 * dt * (self.theta[i] @ self._L2[i].T) + dt * ex_theta[i]
                rhs[[0, -1]] = 0.0
                self.theta[i] = self._lhs_theta[i] @ rhs
            if step % sample_every == 0:
                t.append(step * dt)
                nu.append(self.nusselt())
                amp.append(abs(self.theta[1, mid]) * 2 / self.nx)
        return {"t": np.array(t), "nusselt": np.array(nu), "theta_mode": np.array(amp)}
