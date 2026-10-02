"""Statistically stationary, forced 2D turbulence and its :math:`k^{-5/3}` energy spectrum.

Kolmogorov's 1941 argument needs a steady energy flux :math:`\\varepsilon`
through a range of scales, so it is only visible in *forced* turbulence. In
two dimensions the energy injected at a forcing wavenumber :math:`k_f`
cascades to *larger* scales (Kraichnan, Phys. Fluids 10, 1417 (1967)), and
the same dimensional argument gives

.. math::

    E(k) = C\\,\\varepsilon^{2/3} k^{-5/3}, \\qquad k < k_f,

with a Kolmogorov-Kraichnan constant :math:`C \\approx 6` (Boffetta and
Ecke, Annu. Rev. Fluid Mech. 44, 427 (2012)), while enstrophy cascades to
small scales with :math:`E(k) \\propto k^{-3}`.

:class:`ForcedTurbulence2D` integrates

.. math::

    \\partial_t\\omega + (\\mathbf u\\cdot\\nabla)\\omega
    = -\\alpha\\omega - \\nu_p(-\\nabla^2)^p\\omega + f,

pseudo-spectrally with 2/3-rule dealiasing, an exact integrating factor for
the linear terms and RK4 for advection. :math:`f` is white-in-time random
forcing on the shell :math:`|k - k_f| < \\Delta k`, normalized to inject
energy at the fixed rate :math:`\\varepsilon`; the linear drag
:math:`\\alpha` removes the inverse-cascading energy at large scales and the
hyperviscosity :math:`\\nu_p` the enstrophy at the grid scale.

:class:`ForcedTurbulence3D` is the three-dimensional counterpart, where the
energy cascades the other way, from the forcing to small scales, as in
Kolmogorov's original picture.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sp_fft
from numpy.typing import NDArray

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = ["ForcedTurbulence2D", "ForcedTurbulence3D", "kolmogorov_kraichnan_spectrum"]


def kolmogorov_kraichnan_spectrum(k: NDArray[np.float64], epsilon: float, C: float = 6.0) -> NDArray[np.float64]:
    """Inertial-range energy spectrum :math:`E(k) = C\\varepsilon^{2/3}k^{-5/3}`.

    Parameters
    ----------
    k : array_like
        Wavenumbers.
    epsilon : float
        Energy flux through the inertial range.
    C : float, default 6.0
        Kolmogorov constant; about 1.5 in 3D, and about 6 for the 2D inverse
        cascade.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> float(kolmogorov_kraichnan_spectrum(1.0, epsilon=1.0))
    6.0
    """
    k = np.asarray(k, dtype=np.float64)
    return C * epsilon ** (2.0 / 3.0) * k ** (-5.0 / 3.0)


class ForcedTurbulence2D:
    """Forced, dissipative 2D turbulence on a :math:`2\\pi`-periodic square.

    Parameters
    ----------
    n : int, default 128
        Grid points per side.
    kf : float, default 20.0
        Forcing wavenumber. Must lie well inside the dealiased range
        :math:`k < n/3`.
    forcing_width : float, default 1.5
        Half-width :math:`\\Delta k` of the forced shell.
    epsilon : float, default 1.0
        Energy injection rate.
    drag : float, default 0.1
        Linear (Ekman) drag :math:`\\alpha`.
    hyperviscosity_order : int, default 8
        Power :math:`p` of the hyperviscous Laplacian.
    seed : int, optional
        Seed for the random forcing.

    Examples
    --------
    >>> flow = ForcedTurbulence2D(n=32, kf=6.0, seed=0)
    >>> out = flow.run(t_max=0.4, dt=0.01, t_average=0.2)
    >>> out["k"].shape == out["E"].shape
    True
    >>> bool(0.0 < out["energy"][-1] < 1.0)  # about epsilon * t early on
    True
    """

    def __init__(
        self,
        n: int = 128,
        kf: float = 20.0,
        forcing_width: float = 1.5,
        epsilon: float = 1.0,
        drag: float = 0.1,
        hyperviscosity_order: int = 8,
        seed: int | None = None,
    ):
        if not 2.0 < kf < n / 3 - forcing_width:
            raise InvalidParameterError(f"kf must lie inside the dealiased range (2, n/3 - forcing_width), got {kf}")
        if epsilon <= 0 or drag < 0:
            raise InvalidParameterError("epsilon must be positive and drag non-negative")
        self.n = int(n)
        self.kf = float(kf)
        self.forcing_width = float(forcing_width)
        self.epsilon = float(epsilon)
        self.drag = float(drag)
        self.p = int(hyperviscosity_order)
        self._rng = np.random.default_rng(seed)

        k = np.fft.fftfreq(self.n, 1.0 / self.n)
        kr = np.fft.rfftfreq(self.n, 1.0 / self.n)
        self.KX, self.KY = np.meshgrid(kr, k)  # rfft2 layout: last axis is x
        self.K2 = self.KX**2 + self.KY**2
        self._K2safe = self.K2.copy()
        self._K2safe[0, 0] = 1.0
        self.Kmag = np.sqrt(self.K2)
        self._dealias = (np.abs(self.KX) < self.n / 3) & (np.abs(self.KY) < self.n / 3)
        # the kx = 0 column is left unforced so that independent random phases
        # never break the Hermitian symmetry irfft2 assumes there
        self._shell = (np.abs(self.Kmag - self.kf) < self.forcing_width) & (self.KX > 0)
        # hyperviscosity damps the largest dealiased wavenumber at rate 20
        self.nu_p = 20.0 / (self.n / 3) ** (2 * self.p)
        # rfft2 stores half the plane: count interior columns twice
        self._weight = np.ones_like(self.KX)
        self._weight[:, 1:] = 2.0
        if self.n % 2 == 0:
            self._weight[:, -1] = 1.0
        self.omega_hat = np.zeros_like(self.KX, dtype=complex)

    def _nonlinear(self, wh: NDArray[np.complex128]) -> NDArray[np.complex128]:
        n = self.n
        psih = wh / self._K2safe
        psih[0, 0] = 0.0
        u = np.fft.irfft2(1j * self.KY * psih, s=(n, n))
        v = np.fft.irfft2(-1j * self.KX * psih, s=(n, n))
        wx = np.fft.irfft2(1j * self.KX * wh, s=(n, n))
        wy = np.fft.irfft2(1j * self.KY * wh, s=(n, n))
        return -np.fft.rfft2(u * wx + v * wy) * self._dealias

    def _modal_energy(self, wh: NDArray[np.complex128]) -> NDArray[np.float64]:
        e = 0.5 * self._weight * np.abs(wh) ** 2 / self._K2safe / self.n**4
        e[0, 0] = 0.0
        return e

    def kinetic_energy(self) -> float:
        """Kinetic energy per unit area, :math:`\\tfrac12\\langle|\\mathbf u|^2\\rangle`."""
        return float(self._modal_energy(self.omega_hat).sum())

    def spectrum(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Shell-summed energy spectrum :math:`E(k)` of the current state, with :math:`\\sum_k E(k)` the energy.

        Returns
        -------
        k : ndarray of float, shape (n//2,)
            Integer shell wavenumbers.
        E : ndarray of float, shape (n//2,)
        """
        e = self._modal_energy(self.omega_hat)
        kb = np.rint(self.Kmag).astype(int)
        E = np.bincount(kb.ravel(), e.ravel(), minlength=self.n)[: self.n // 2]
        return np.arange(self.n // 2, dtype=np.float64), E

    def vorticity(self) -> NDArray[np.float64]:
        """Real-space vorticity field, shape ``(n, n)``."""
        return np.fft.irfft2(self.omega_hat, s=(self.n, self.n))

    def run(self, t_max: float = 40.0, dt: float = 0.004, t_average: float = 20.0, sample_every: int = 20) -> dict[str, NDArray[np.float64]]:
        """Advance the flow and time-average its spectrum.

        Parameters
        ----------
        t_max : float, default 40.0
            Integration time from the current state.
        dt : float, default 0.004
            Time step.
        t_average : float, default 20.0
            Average the spectrum over the last ``t_average`` time units.
        sample_every : int, default 20
            Steps between energy and spectrum samples.

        Returns
        -------
        dict
            ``"k"``, time-averaged ``"E"``, the sampled ``"t"`` and
            ``"energy"`` series, and the final ``"omega"``.
        """
        n_steps = int(round(t_max / dt))
        lin = -(self.drag + self.nu_p * self.K2**self.p)
        E_full = np.exp(lin * dt)
        E_half = np.exp(lin * dt / 2)
        shell_energy = self._modal_energy((self._shell * self.Kmag).astype(complex)).sum()
        amplitude = np.sqrt(self.epsilon / shell_energy * dt)
        wh = self.omega_hat
        times, energies = [], []
        E_sum = np.zeros(self.n // 2)
        n_avg = 0
        start_avg = n_steps - int(round(t_average / dt))
        for step in range(1, n_steps + 1):
            a = self._nonlinear(wh)
            b = self._nonlinear(E_half * (wh + dt / 2 * a))
            c = self._nonlinear(E_half * wh + dt / 2 * b)
            d = self._nonlinear(E_full * wh + dt * E_half * c)
            wh = E_full * wh + dt / 6 * (E_full * a + 2 * E_half * (b + c) + d)
            # random-phase forcing on the shell; a white-noise increment of size
            # sqrt(dt) injects energy at the mean rate epsilon (Ito)
            phase = np.exp(2j * np.pi * self._rng.random(wh.shape))
            wh = wh + amplitude * self._shell * self.Kmag * phase
            if step % sample_every == 0:
                self.omega_hat = wh
                times.append(step * dt)
                energies.append(self.kinetic_energy())
                if step > start_avg:
                    E_sum += self.spectrum()[1]
                    n_avg += 1
        self.omega_hat = wh
        k = np.arange(self.n // 2)
        return {
            "k": k,
            "E": E_sum / max(n_avg, 1),
            "t": np.array(times),
            "energy": np.array(energies),
            "omega": self.vorticity(),
        }


class ForcedTurbulence3D:
    """Forced, statistically steady 3D turbulence on a :math:`(2\\pi)^3`-periodic cube.

    The incompressible Navier-Stokes equations are integrated in rotational
    form,

    .. math::

        \\partial_t\\hat{\\mathbf u} = \\mathsf P\\,\\widehat{\\mathbf u\\times\\boldsymbol\\omega}
            - \\nu_p k^{2p}\\hat{\\mathbf u} + \\hat{\\mathbf f},

    where :math:`\\mathsf P` projects onto divergence-free fields, with 2/3-rule
    dealiasing, an exact integrating factor for the (hyper)viscosity and a
    second-order Runge-Kutta step. The forcing
    :math:`\\hat{\\mathbf f} = \\varepsilon\\,\\hat{\\mathbf u}/(2E_f)` on the
    shell :math:`0 < k < k_f`, with :math:`E_f` that shell's energy, injects
    energy at exactly the rate :math:`\\varepsilon` (Lundgren 2003; Alvelius
    1999 for random variants). In 3D the energy cascades to small scales,
    and in the inertial range Kolmogorov's 1941 law
    :math:`E(k) = C_K\\varepsilon^{2/3}k^{-5/3}`, :math:`C_K \\approx 1.5`, holds.
    On a :math:`64^3` grid that range spans only :math:`3 \\lesssim k \\lesssim 10`.

    Parameters
    ----------
    n : int, default 64
    kf : float, default 2.5
        Forcing cutoff.
    epsilon : float, default 1.0
    hyperviscosity_order : int, default 2
        Power :math:`p`; ``1`` is ordinary viscosity.
    damping_at_kmax : float, default 20.0
        :math:`\\nu_p k_{\\max}^{2p}`, the damping rate at the dealiasing cutoff.
    seed : int, optional
        Seed for the random initial field.

    Examples
    --------
    >>> flow = ForcedTurbulence3D(n=16, seed=0)
    >>> out = flow.run(t_max=0.2, dt=0.01)
    >>> bool(flow.max_divergence() < 1e-10)
    True
    """

    def __init__(
        self,
        n: int = 64,
        kf: float = 2.5,
        epsilon: float = 1.0,
        hyperviscosity_order: int = 2,
        damping_at_kmax: float = 20.0,
        seed: int | None = None,
    ):
        if epsilon <= 0 or not 1.0 < kf < n / 3:
            raise InvalidParameterError("need epsilon > 0 and 1 < kf < n/3")
        self.n, self.kf, self.epsilon, self.p = int(n), float(kf), float(epsilon), int(hyperviscosity_order)
        k = np.fft.fftfreq(self.n, 1.0 / self.n)
        kr = np.fft.rfftfreq(self.n, 1.0 / self.n)
        self.KX, self.KY, self.KZ = np.meshgrid(k, k, kr, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self._K2safe = self.K2.copy()
        self._K2safe[0, 0, 0] = 1.0
        self.Kmag = np.sqrt(self.K2)
        kmax = self.n / 3
        self._dealias = (np.abs(self.KX) < kmax) & (np.abs(self.KY) < kmax) & (np.abs(self.KZ) < kmax)
        self._force = (self.Kmag < self.kf) & (self.Kmag > 0)
        self.nu_p = damping_at_kmax / kmax ** (2 * self.p)
        self._weight = np.ones_like(self.KZ)
        self._weight[..., 1:] = 2.0
        if self.n % 2 == 0:
            self._weight[..., -1] = 1.0
        rng = np.random.default_rng(seed)
        u = [np.fft.rfftn(rng.standard_normal((self.n,) * 3)) * self._force for _ in range(3)]
        u = self._project(u)
        scale = np.sqrt(0.5 / self._energy(u))
        self.u_hat = [c * scale for c in u]

    def _project(self, u):
        div = (self.KX * u[0] + self.KY * u[1] + self.KZ * u[2]) / self._K2safe
        return [u[0] - self.KX * div, u[1] - self.KY * div, u[2] - self.KZ * div]

    def _energy(self, u, mask=None) -> float:
        total = 0.0
        for c in u:
            e = self._weight * np.abs(c if mask is None else c * mask) ** 2
            total += float(e.sum())
        return 0.5 * total / self.n**6

    def _irfft(self, c):
        return sp_fft.irfftn(c, s=(self.n,) * 3, axes=(0, 1, 2), workers=-1)

    def _rhs(self, u):
        KX, KY, KZ = self.KX, self.KY, self.KZ
        omega = [1j * (KY * u[2] - KZ * u[1]), 1j * (KZ * u[0] - KX * u[2]), 1j * (KX * u[1] - KY * u[0])]
        U = [self._irfft(c) for c in u]
        W = [self._irfft(c) for c in omega]
        cross = [U[1] * W[2] - U[2] * W[1], U[2] * W[0] - U[0] * W[2], U[0] * W[1] - U[1] * W[0]]
        nonlinear = self._project([sp_fft.rfftn(c, axes=(0, 1, 2), workers=-1) * self._dealias for c in cross])
        gain = self.epsilon / (2.0 * self._energy(u, self._force))
        return [nonlinear[i] + gain * u[i] * self._force for i in range(3)]

    def kinetic_energy(self) -> float:
        """:math:`\\tfrac12\\langle|\\mathbf u|^2\\rangle`."""
        return self._energy(self.u_hat)

    def max_divergence(self) -> float:
        """Largest :math:`|\\nabla\\cdot\\mathbf u|` on the grid (zero up to roundoff)."""
        div = 1j * (self.KX * self.u_hat[0] + self.KY * self.u_hat[1] + self.KZ * self.u_hat[2])
        return float(np.abs(self._irfft(div)).max())

    def spectrum(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Shell-summed energy spectrum, with :math:`\\sum_k E(k)` the kinetic energy."""
        e = sum(self._weight * np.abs(c) ** 2 for c in self.u_hat) / (2 * self.n**6)
        kb = np.rint(self.Kmag).astype(int)
        E = np.bincount(kb.ravel(), e.ravel(), minlength=self.n)[: self.n // 2]
        return np.arange(self.n // 2, dtype=np.float64), E

    def dissipation_rate(self) -> float:
        """:math:`\\sum_k 2\\nu_p k^{2p}E(\\mathbf k)`, the rate the (hyper)viscosity removes energy."""
        e = sum(self._weight * np.abs(c) ** 2 for c in self.u_hat) / (2 * self.n**6)
        return float(np.sum(2.0 * self.nu_p * self.K2**self.p * e))

    def run(self, t_max: float = 10.0, dt: float = 0.01, t_average: float = 5.0, sample_every: int = 10) -> dict[str, NDArray[np.float64]]:
        """Advance and time-average the spectrum over the last ``t_average``.

        Returns
        -------
        dict
            ``"k"``, time-averaged ``"E"``, sampled ``"t"``, ``"energy"`` and
            ``"dissipation"``.
        """
        n_steps = int(round(t_max / dt))
        lin = -self.nu_p * self.K2**self.p
        full, half = np.exp(lin * dt), np.exp(lin * dt / 2)
        start_avg = n_steps - int(round(t_average / dt))
        t, energy, diss = [], [], []
        E_sum = np.zeros(self.n // 2)
        n_avg = 0
        u = self.u_hat
        for step in range(1, n_steps + 1):
            a = self._rhs(u)
            mid = [half * (u[i] + 0.5 * dt * a[i]) for i in range(3)]
            b = self._rhs(mid)
            u = [full * u[i] + dt * half * b[i] for i in range(3)]
            if step % sample_every == 0:
                self.u_hat = u
                t.append(step * dt)
                energy.append(self.kinetic_energy())
                diss.append(self.dissipation_rate())
                if step > start_avg:
                    E_sum += self.spectrum()[1]
                    n_avg += 1
        self.u_hat = u
        return {
            "k": np.arange(self.n // 2, dtype=np.float64),
            "E": E_sum / max(n_avg, 1),
            "t": np.array(t),
            "energy": np.array(energy),
            "dissipation": np.array(diss),
        }
