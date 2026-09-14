r"""Nuclear binding energy and reaction Q-values.

Masses and energies here are in **MeV**, the universal convention for the
semi-empirical mass formula's fitted coefficients.

- :func:`semf_binding_energy`, :func:`binding_energy_per_nucleon` -- the
  Weizsacker (liquid-drop) semi-empirical mass formula.
- :func:`q_value` -- the energy released (or absorbed) in a nuclear reaction.
"""

from __future__ import annotations

__all__ = ["semf_binding_energy", "binding_energy_per_nucleon", "q_value"]

_A_V = 15.75
_A_S = 17.8
_A_C = 0.711
_A_A = 23.7
_A_P = 11.18


def semf_binding_energy(Z, A):
    r"""Semi-empirical (Weizsacker) total nuclear binding energy, in MeV.

    .. math::

        B(Z,A) = a_V A - a_S A^{2/3} - a_C\frac{Z(Z-1)}{A^{1/3}}
        - a_A\frac{(A-2Z)^2}{A} + \delta(A,Z)

    with the standard coefficients :math:`a_V=15.75`, :math:`a_S=17.8`,
    :math:`a_C=0.711`, :math:`a_A=23.7` MeV, and the pairing term
    :math:`\delta=+a_PA^{-1/2}` for even-even nuclei,
    :math:`\delta=-a_PA^{-1/2}` for odd-odd nuclei, and :math:`\delta=0`
    for odd-:math:`A` nuclei, with :math:`a_P\approx11.18` MeV.

    Parameters
    ----------
    Z : int
        Atomic (proton) number.
    A : int
        Mass number.

    Returns
    -------
    float
        Total binding energy, in MeV.

    Examples
    --------
    >>> B = semf_binding_energy(26, 56)  # Iron-56
    >>> 8.0 < B / 56 < 9.0
    True
    """
    N = A - Z
    if Z % 2 == 0 and N % 2 == 0:
        delta = _A_P / A**0.5
    elif Z % 2 == 1 and N % 2 == 1:
        delta = -_A_P / A**0.5
    else:
        delta = 0.0
    return _A_V * A - _A_S * A ** (2 / 3) - _A_C * Z * (Z - 1) / A ** (1 / 3) - _A_A * (A - 2 * Z) ** 2 / A + delta


def binding_energy_per_nucleon(Z, A):
    """Binding energy per nucleon, ``semf_binding_energy(Z, A) / A``, in MeV.

    Examples
    --------
    >>> bpn = binding_energy_per_nucleon(26, 56)
    >>> 8.0 < bpn < 9.0
    True
    """
    return semf_binding_energy(Z, A) / A


def q_value(reactant_masses, product_masses):
    r"""Reaction Q-value, :math:`Q=\sum m_{\rm reactants}-\sum m_{\rm products}`.

    A positive ``Q`` means the reaction is exothermic (releases energy);
    a negative ``Q`` means it is endothermic and requires that much
    additional kinetic energy to proceed. Masses (or mass-energies) may be
    in any consistent unit.

    Parameters
    ----------
    reactant_masses : iterable of float
        Masses of the reactants.
    product_masses : iterable of float
        Masses of the products.

    Returns
    -------
    float

    Examples
    --------
    >>> round(q_value([2.014102, 3.016049], [4.002602, 1.008665]), 6)  # D + T -> He4 + n
    0.018884
    """
    return sum(reactant_masses) - sum(product_masses)
