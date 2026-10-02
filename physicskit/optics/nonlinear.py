"""Second-harmonic generation and phase matching.

In a medium with a second-order susceptibility :math:`\\chi^{(2)}`, a field at
frequency :math:`\\omega` drives a polarization at :math:`2\\omega` (Franken
et al., Phys. Rev. Lett. 7, 118 (1961)). For collinear plane waves with slowly
varying amplitudes :math:`A_1` (fundamental) and :math:`A_2` (second
harmonic), normalized so that :math:`|A_1|^2 + |A_2|^2` is the conserved
power, the coupled-amplitude equations are (Armstrong, Bloembergen, Ducuing
and Pershan, Phys. Rev. 127, 1918 (1962); Boyd, *Nonlinear Optics*, ch. 2)

.. math::

    \\frac{dA_1}{dz} = -i\\kappa A_2 A_1^* e^{-i\\Delta k z}, \\qquad
    \\frac{dA_2}{dz} = -i\\kappa A_1^2 e^{i\\Delta k z},
    \\qquad \\Delta k = k_{2\\omega} - 2k_\\omega.

Three regimes are solved in closed form:

- undepleted pump, :math:`|A_2(L)|^2 = \\kappa^2|A_1|^4 L^2
  \\operatorname{sinc}^2(\\Delta k L/2)`, which oscillates with the coherence
  length :math:`L_c = \\pi/\\Delta k`;
- perfect phase matching with pump depletion,
  :math:`|A_2|^2 = |A_0|^2\\tanh^2(\\kappa A_0 z)`;
- quasi-phase matching, where the sign of :math:`\\kappa` is flipped every
  :math:`L_c` and the harmonic grows linearly at :math:`2/\\pi` of the
  phase-matched rate.

Birefringent phase matching uses the extraordinary index
:math:`n_e(\\theta)` to make :math:`n_e^{2\\omega}(\\theta) = n_o^{\\omega}`
(Giordmaine 1962; Maker et al. 1962).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

__all__ = [
    "extraordinary_index",
    "shg_coupled_amplitudes",
    "shg_phase_matched_efficiency",
    "shg_undepleted_power",
    "type_i_phase_matching_angle",
]


def shg_undepleted_power(L: ArrayLike, delta_k: float, kappa: float = 1.0, A1: float = 1.0) -> NDArray[np.float64]:
    """Second-harmonic power for an undepleted pump.

    .. math::

        |A_2(L)|^2 = \\kappa^2 |A_1|^4 L^2 \\operatorname{sinc}^2(\\Delta k L / 2)
        = \\frac{4\\kappa^2|A_1|^4}{\\Delta k^2}\\sin^2\\frac{\\Delta k L}{2}.

    Parameters
    ----------
    L : array_like
        Crystal length(s).
    delta_k : float
        Phase mismatch.
    kappa : float, default 1.0
        Coupling constant.
    A1 : float, default 1.0
        Pump amplitude.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> round(float(shg_undepleted_power(np.pi, delta_k=1.0)), 12)  # one coherence length: 4/dk^2
    4.0
    """
    L = np.asarray(L, dtype=float)
    return kappa**2 * A1**4 * L**2 * np.sinc(delta_k * L / (2 * np.pi)) ** 2


def shg_phase_matched_efficiency(z: ArrayLike, kappa: float = 1.0, A0: float = 1.0) -> NDArray[np.float64]:
    """Conversion efficiency :math:`|A_2|^2/|A_0|^2 = \\tanh^2(\\kappa A_0 z)` with pump depletion.

    Parameters
    ----------
    z : array_like
        Propagation distance(s).
    kappa : float, default 1.0
    A0 : float, default 1.0
        Input fundamental amplitude.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> round(float(shg_phase_matched_efficiency(10.0)), 6)
    1.0
    """
    return np.tanh(kappa * A0 * np.asarray(z, dtype=float)) ** 2


def shg_coupled_amplitudes(
    z: ArrayLike,
    delta_k: float = 0.0,
    kappa: float = 1.0,
    A0: complex = 1.0,
    qpm_period: float | None = None,
) -> dict[str, NDArray]:
    """Integrate the SHG coupled-amplitude equations, with optional quasi-phase matching.

    Parameters
    ----------
    z : array_like
        Increasing output positions, starting at 0.
    delta_k : float, default 0.0
        Phase mismatch :math:`\\Delta k`.
    kappa : float, default 1.0
        Nonlinear coupling.
    A0 : complex, default 1.0
        Fundamental amplitude at :math:`z = 0`; the harmonic starts at 0.
    qpm_period : float, optional
        If given, :math:`\\kappa` changes sign every half period
        (periodically poled crystal); first-order QPM uses
        :math:`2\\pi/\\Delta k`.

    Returns
    -------
    dict
        Complex ``"A1"`` and ``"A2"`` at ``z``.

    Examples
    --------
    >>> out = shg_coupled_amplitudes(np.linspace(0, 2, 5))
    >>> bool(np.allclose(np.abs(out["A2"]) ** 2, np.tanh(np.linspace(0, 2, 5)) ** 2, atol=1e-6))
    True
    """
    z = np.asarray(z, dtype=float)

    def rhs(zz, y):
        A1 = y[0] + 1j * y[1]
        A2 = y[2] + 1j * y[3]
        k = kappa
        if qpm_period is not None and np.floor(2 * zz / qpm_period) % 2 == 1:
            k = -kappa
        dA1 = -1j * k * A2 * np.conj(A1) * np.exp(-1j * delta_k * zz)
        dA2 = -1j * k * A1**2 * np.exp(1j * delta_k * zz)
        return [dA1.real, dA1.imag, dA2.real, dA2.imag]

    A0 = complex(A0)
    max_step = np.inf if qpm_period is None else qpm_period / 40
    if delta_k:
        max_step = min(max_step, np.pi / abs(delta_k) / 20)
    sol = solve_ivp(rhs, (z[0], z[-1]), [A0.real, A0.imag, 0.0, 0.0], t_eval=z, rtol=1e-10, atol=1e-12, max_step=max_step)
    return {"A1": sol.y[0] + 1j * sol.y[1], "A2": sol.y[2] + 1j * sol.y[3]}


def extraordinary_index(theta: ArrayLike, n_o: float, n_e: float) -> NDArray[np.float64]:
    """Index of the extraordinary wave at angle :math:`\\theta` to the optic axis of a uniaxial crystal.

    .. math::

        \\frac{1}{n_e(\\theta)^2} = \\frac{\\cos^2\\theta}{n_o^2} + \\frac{\\sin^2\\theta}{n_e^2}

    Parameters
    ----------
    theta : array_like
        Angle in radians.
    n_o, n_e : float
        Ordinary and principal extraordinary indices.

    Returns
    -------
    ndarray
    """
    theta = np.asarray(theta, dtype=float)
    return 1.0 / np.sqrt(np.cos(theta) ** 2 / n_o**2 + np.sin(theta) ** 2 / n_e**2)


def type_i_phase_matching_angle(n_o_w: float, n_o_2w: float, n_e_2w: float) -> float:
    """Type-I (ooe) phase-matching angle of a negative uniaxial crystal.

    Solves :math:`n_e^{2\\omega}(\\theta) = n_o^{\\omega}`:

    .. math::

        \\sin^2\\theta_m = \\frac{(n_o^{\\omega})^{-2} - (n_o^{2\\omega})^{-2}}
                               {(n_e^{2\\omega})^{-2} - (n_o^{2\\omega})^{-2}}.

    Parameters
    ----------
    n_o_w : float
        Ordinary index at the fundamental.
    n_o_2w, n_e_2w : float
        Ordinary and extraordinary indices at the harmonic.

    Returns
    -------
    float
        :math:`\\theta_m` in radians.

    Raises
    ------
    ValueError
        If the birefringence cannot compensate the dispersion.

    Examples
    --------
    KDP doubling 1064 nm (Sellmeier indices from Zernike, JOSA 54, 1215 (1964)):

    >>> round(float(np.degrees(type_i_phase_matching_angle(1.4942, 1.5129, 1.4709))), 1)
    41.3
    """
    s2 = (n_o_w**-2 - n_o_2w**-2) / (n_e_2w**-2 - n_o_2w**-2)
    if not 0.0 <= s2 <= 1.0:
        raise ValueError("no type-I phase-matching angle exists for these indices")
    return float(np.arcsin(np.sqrt(s2)))
