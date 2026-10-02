"""Interference in layered media: thin films, Fabry-Pérot cavities and 1D photonic crystals.

A plane wave crossing a stack of homogeneous layers is described exactly by
the characteristic matrix of each layer (Abelès, Ann. Phys. (Paris) 5, 596
(1950); Born and Wolf, *Principles of Optics*, sec. 1.6),

.. math::

    M_j = \\begin{pmatrix} \\cos\\delta_j & i\\sin\\delta_j/\\eta_j \\\\
                           i\\eta_j\\sin\\delta_j & \\cos\\delta_j \\end{pmatrix},
    \\qquad \\delta_j = \\frac{2\\pi}{\\lambda} n_j d_j \\cos\\theta_j,

with tilted admittance :math:`\\eta_j = n_j\\cos\\theta_j` (s polarization)
or :math:`n_j/\\cos\\theta_j` (p). The product of the layer matrices gives the
reflection and transmission of the whole stack. Three classic results
follow:

- a single quarter-wave film of index :math:`\\sqrt{n_0 n_s}` cancels
  reflection at one wavelength (antireflection coating);
- two mirrors of reflectance :math:`R` facing each other transmit the Airy
  function :math:`T = 1/(1 + F\\sin^2(\\delta/2))`, :math:`F = 4R/(1-R)^2`
  (Fabry and Pérot, 1899);
- a periodic stack has Bloch modes with :math:`\\cos(K\\Lambda) =
  \\tfrac12\\operatorname{tr} M_{\\text{period}}`, and where
  :math:`|\\tfrac12\\operatorname{tr} M| > 1` light cannot propagate: a
  photonic band gap (Rayleigh 1887; Yablonovitch 1987).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "airy_transmission",
    "bloch_wavenumber",
    "coefficient_of_finesse",
    "fabry_perot_free_spectral_range",
    "finesse",
    "multilayer_response",
    "quarter_wave_band_gap",
    "quarter_wave_stack",
]


def multilayer_response(
    n_layers: ArrayLike,
    d_layers: ArrayLike,
    wavelength: ArrayLike,
    n_incident: complex = 1.0,
    n_substrate: complex = 1.5,
    angle: float = 0.0,
    polarization: str = "s",
) -> dict[str, NDArray]:
    """Reflection and transmission of a stack of thin layers by the characteristic-matrix method.

    Parameters
    ----------
    n_layers : array_like of complex, shape (m,)
        Refractive index of each layer, from the incident side; complex
        values :math:`n + i\\kappa` with :math:`\\kappa > 0` describe
        absorption (time dependence :math:`e^{-i\\omega t}`).
    d_layers : array_like of float, shape (m,)
        Physical thickness of each layer, in the units of ``wavelength``.
    wavelength : array_like of float
        Vacuum wavelength(s).
    n_incident : complex, default 1.0
        Index of the incident medium.
    n_substrate : complex, default 1.5
        Index of the exit medium.
    angle : float, default 0.0
        Angle of incidence in radians.
    polarization : {"s", "p"}, default "s"

    Returns
    -------
    dict
        Complex amplitude coefficients ``"r"``, ``"t"`` and the power
        reflectance ``"R"`` and transmittance ``"T"``, each shaped like
        ``wavelength``.

    Examples
    --------
    Bare glass reflects :math:`((1 - 1.5)/(1 + 1.5))^2 = 4\\%`; a quarter-wave
    film of index :math:`\\sqrt{1.5}` removes it at the design wavelength:

    >>> round(float(multilayer_response([], [], 550.0)["R"]), 4)
    0.04
    >>> n_ar = np.sqrt(1.5)
    >>> float(multilayer_response([n_ar], [550.0 / (4 * n_ar)], 550.0)["R"]) < 1e-12
    True
    """
    if polarization not in ("s", "p"):
        raise ValueError("polarization must be 's' or 'p'")
    # the characteristic matrix below is written for e^{+i omega t}, where
    # absorption is n - i kappa, so work with conjugated indices
    n_layers = np.conj(np.atleast_1d(np.asarray(n_layers, dtype=complex)))
    d_layers = np.atleast_1d(np.asarray(d_layers, dtype=float))
    if n_layers.shape != d_layers.shape:
        raise ValueError("n_layers and d_layers must have the same length")
    lam = np.asarray(wavelength, dtype=float)
    shape = lam.shape
    lam = lam.reshape(-1)
    n0 = np.conj(complex(n_incident))
    ns = np.conj(complex(n_substrate))
    s0 = n0 * np.sin(angle)  # conserved n sin(theta)

    def admittance(n):
        cos_t = np.sqrt(1 - (s0 / n) ** 2 + 0j)
        return (n * cos_t if polarization == "s" else n / cos_t), cos_t

    eta0, _ = admittance(n0)
    etas, _ = admittance(ns)
    M11 = np.ones_like(lam, dtype=complex)
    M12 = np.zeros_like(lam, dtype=complex)
    M21 = np.zeros_like(lam, dtype=complex)
    M22 = np.ones_like(lam, dtype=complex)
    for n, d in zip(n_layers, d_layers):
        eta, cos_t = admittance(n)
        delta = 2 * np.pi * n * d * cos_t / lam
        c, s = np.cos(delta), np.sin(delta)
        a11, a12, a21, a22 = c, 1j * s / eta, 1j * eta * s, c
        M11, M12, M21, M22 = M11 * a11 + M12 * a21, M11 * a12 + M12 * a22, M21 * a11 + M22 * a21, M21 * a12 + M22 * a22
    B = M11 + M12 * etas
    C = M21 + M22 * etas
    r = (eta0 * B - C) / (eta0 * B + C)
    t = 2 * eta0 / (eta0 * B + C)
    R = np.abs(r) ** 2
    T = np.real(etas) / np.real(eta0) * np.abs(t) ** 2
    return {"r": np.conj(r).reshape(shape), "t": np.conj(t).reshape(shape), "R": R.reshape(shape), "T": T.reshape(shape)}


def quarter_wave_stack(n_high: float, n_low: float, n_pairs: int, design_wavelength: float, cavity: bool = False) -> tuple[NDArray, NDArray]:
    """Layer indices and thicknesses of a quarter-wave (Bragg) mirror, or of a Fabry-Pérot cavity.

    Parameters
    ----------
    n_high, n_low : float
        Indices of the two materials; the stack starts with ``n_high``.
    n_pairs : int
        Number of high/low pairs.
    design_wavelength : float
        Wavelength at which every layer is a quarter wave thick.
    cavity : bool, default False
        If True, return two such mirrors separated by a half-wave
        (:math:`\\lambda/2n_{\\text{low}}`) spacer of the low index: a
        Fabry-Pérot cavity with a resonance at ``design_wavelength``.

    Returns
    -------
    n_layers, d_layers : ndarray

    Examples
    --------
    >>> n, d = quarter_wave_stack(2.3, 1.38, 2, 600.0)
    >>> n.real.tolist(), (n.real * d).tolist()
    ([2.3, 1.38, 2.3, 1.38], [150.0, 150.0, 150.0, 150.0])
    """
    n_mirror = np.tile([n_high, n_low], n_pairs).astype(complex)
    d_mirror = design_wavelength / (4 * n_mirror.real)
    if not cavity:
        return n_mirror, d_mirror
    n = np.concatenate((n_mirror, [n_low], n_mirror[::-1]))
    d = np.concatenate((d_mirror, [design_wavelength / (2 * n_low)], d_mirror[::-1]))
    return n, d


def coefficient_of_finesse(R: ArrayLike) -> NDArray:
    """:math:`F = 4R/(1 - R)^2`.

    Parameters
    ----------
    R : array_like
        Mirror power reflectance, :math:`0 \\le R < 1`.

    Returns
    -------
    ndarray
    """
    R = np.asarray(R, dtype=float)
    return 4 * R / (1 - R) ** 2


def finesse(R: ArrayLike) -> NDArray:
    """Reflectance finesse :math:`\\mathcal F = \\pi\\sqrt R/(1 - R)`: free spectral range over linewidth.

    Parameters
    ----------
    R : array_like
        Mirror power reflectance.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> round(float(finesse(0.9)), 2)
    29.8
    """
    R = np.asarray(R, dtype=float)
    return np.pi * np.sqrt(R) / (1 - R)


def airy_transmission(delta: ArrayLike, R: float) -> NDArray:
    """Fabry-Pérot (Airy) transmission of two lossless mirrors of reflectance :math:`R`.

    .. math::

        T = \\frac{1}{1 + F\\sin^2(\\delta/2)}, \\qquad \\delta = \\frac{4\\pi n L\\cos\\theta}{\\lambda},

    where :math:`\\delta` is the round-trip phase.

    Parameters
    ----------
    delta : array_like
        Round-trip phase.
    R : float
        Mirror reflectance.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> float(airy_transmission(0.0, 0.99))  # every resonance transmits fully
    1.0
    """
    return 1.0 / (1.0 + coefficient_of_finesse(R) * np.sin(np.asarray(delta) / 2) ** 2)


def fabry_perot_free_spectral_range(length: float, n: float = 1.0, wavelength: float | None = None) -> float:
    """Free spectral range :math:`c/2nL` in frequency, or :math:`\\lambda^2/2nL` in wavelength.

    Parameters
    ----------
    length : float
        Mirror spacing :math:`L`.
    n : float, default 1.0
        Index between the mirrors.
    wavelength : float, optional
        If given, return the range in wavelength units near this
        wavelength; otherwise in units of :math:`c` per length unit.

    Returns
    -------
    float
    """
    if wavelength is None:
        return 1.0 / (2 * n * length)
    return wavelength**2 / (2 * n * length)


def bloch_wavenumber(n_layers: ArrayLike, d_layers: ArrayLike, wavelength: ArrayLike) -> NDArray[np.complex128]:
    """Bloch wavenumber :math:`K` of an infinite periodic stack at normal incidence.

    For one period of layers, :math:`\\cos(K\\Lambda) = \\tfrac12
    \\operatorname{tr} M_{\\text{period}}`; for two layers this is

    .. math::

        \\cos(K\\Lambda) = \\cos\\delta_1\\cos\\delta_2
        - \\tfrac12\\left(\\frac{n_1}{n_2} + \\frac{n_2}{n_1}\\right)\\sin\\delta_1\\sin\\delta_2.

    Parameters
    ----------
    n_layers, d_layers : array_like
        Indices and thicknesses of the layers of one period.
    wavelength : array_like
        Vacuum wavelength(s).

    Returns
    -------
    ndarray of complex
        :math:`K\\Lambda`, with real part in :math:`[0, \\pi]` and a nonzero
        imaginary part (the decay per period) inside band gaps.

    Examples
    --------
    >>> KL = bloch_wavenumber([1.0, 1.0], [0.25, 0.25], 1.0)  # uniform medium: K = 2 pi n / lambda
    >>> round(float(KL.real), 6) == round(np.pi, 6)
    True
    """
    n_layers = np.atleast_1d(np.asarray(n_layers, dtype=float))
    d_layers = np.atleast_1d(np.asarray(d_layers, dtype=float))
    lam = np.asarray(wavelength, dtype=float)
    M11 = np.ones_like(lam, dtype=complex)
    M12 = np.zeros_like(lam, dtype=complex)
    M21 = np.zeros_like(lam, dtype=complex)
    M22 = np.ones_like(lam, dtype=complex)
    for n, d in zip(n_layers, d_layers):
        delta = 2 * np.pi * n * d / lam
        c, s = np.cos(delta), np.sin(delta)
        a11, a12, a21, a22 = c, 1j * s / n, 1j * n * s, c
        M11, M12, M21, M22 = M11 * a11 + M12 * a21, M11 * a12 + M12 * a22, M21 * a11 + M22 * a21, M21 * a12 + M22 * a22
    half_trace = np.real(0.5 * (M11 + M22))
    KL = np.arccos(half_trace + 0j)
    return KL.real + 1j * np.abs(KL.imag)


def quarter_wave_band_gap(n_high: float, n_low: float) -> float:
    """Relative width :math:`\\Delta\\omega/\\omega_0` of the first gap of a quarter-wave stack.

    .. math::

        \\frac{\\Delta\\omega}{\\omega_0} = \\frac4\\pi\\arcsin\\frac{n_H - n_L}{n_H + n_L}

    (Yariv and Yeh, *Optical Waves in Crystals*, 1984, sec. 6.3).

    Parameters
    ----------
    n_high, n_low : float

    Returns
    -------
    float

    Examples
    --------
    >>> round(quarter_wave_band_gap(2.3, 1.38), 4)
    0.3217
    """
    return float(4 / np.pi * np.arcsin(abs(n_high - n_low) / (n_high + n_low)))
