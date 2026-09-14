r"""Toy quark confinement and QCD string breaking.

Uses the same natural-unit convention (:math:`\hbar=c=1`) as
:mod:`physicskit.particle.kinematics`.

A pair of quarks confined by a linear potential :math:`V(r)=\kappa r`
(the standard long-distance behavior of the QCD static potential,
confirmed by lattice QCD, with :math:`\kappa` the string tension,
:math:`\kappa\approx0.9\ {\rm GeV/fm}\approx0.18\ {\rm GeV}^2` for real
QCD) is receding at a prescribed constant speed. The color flux tube
("string") connecting them stores energy :math:`\kappa r(t)`; once that
exceeds the threshold to pair-produce a light :math:`q\bar q` pair from
the vacuum, the string breaks.

- :func:`string_tension_energy` -- the stored linear-potential energy.
- :func:`string_break_chain` -- a sequence of such breaks as the
  original pair keeps separating, sharing the growing total extent
  evenly among the (growing number of) string segments.

**Simplifications, clearly flagged**: the two original quarks are given
a *prescribed* constant recession velocity rather than one obtained by
solving their equation of motion under the confining force plus
radiation reaction (a full dynamical treatment is out of scope for a
toy model); the newly created quark-antiquark pairs are bookkept only
through the energy balance :math:`\kappa r=2m_qN` (each of the ``N``
current segments must independently store enough energy to pair-produce
its own light quark pair) -- individual momentum conservation for the
produced pairs, and their own subsequent dynamics, are not tracked.
"""

from __future__ import annotations

import numpy as np

__all__ = ["string_tension_energy", "string_break_chain"]


def string_tension_energy(r, kappa):
    r"""Energy stored in a confining linear-potential string, :math:`V(r)=\kappa r`.

    Parameters
    ----------
    r : array_like
        Quark-antiquark separation.
    kappa : float
        String tension.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> float(string_tension_energy(2.0, 0.9))
    1.8
    """
    return kappa * np.abs(np.asarray(r, dtype=float))


def string_break_chain(t, v, kappa, m_q, r0=0.0, n_breaks=3):
    r"""Sequence of string-breaking events as a receding quark pair separates.

    The original quark pair separates at the prescribed constant speed
    ``v``, :math:`r(t)=r_0+vt`. The ``k``-th break occurs when the total
    stored string energy :math:`\kappa r(t)` first reaches :math:`2m_qk`
    -- the energy needed for *each* of the ``k`` string segments existing
    just before that break to independently pair-produce a light
    :math:`q\bar q` pair at threshold (each of the ``k`` segments is
    assumed, for this bookkeeping, to carry an equal share
    :math:`\kappa r(t)/k` of the total stored energy). After the break
    there are ``k+1`` segments.

    Parameters
    ----------
    t : array_like
        Times at which to evaluate the state of the system.
    v : float
        Constant recession speed of the original quark pair, :math:`0<v<1`.
    kappa : float
        String tension.
    m_q : float
        Constituent mass of the light quark pair created at each break
        (the pair-production threshold for one string segment is
        :math:`2m_q`).
    r0 : float, default=0.0
        Initial quark-antiquark separation at :math:`t=0`.
    n_breaks : int, default=3
        Maximum number of sequential breaks to compute.

    Returns
    -------
    dict
        ``t``, ``r`` (total extent), ``energy_total`` (:math:`\kappa r`),
        ``n_segments`` (number of string segments at each time),
        ``energy_per_segment``, ``break_times``, ``r_break_unit``
        (:math:`2m_q/\kappa`, the extent of a single fresh segment at
        the instant it breaks), ``kappa``, ``m_q``.

    Examples
    --------
    >>> import numpy as np
    >>> sim = string_break_chain(np.linspace(0, 10, 5), v=0.5, kappa=1.0, m_q=1.0, n_breaks=2)
    >>> sim["break_times"]
    array([4., 8.])
    """
    if not (0.0 < v < 1.0):
        raise ValueError(f"v={v} must satisfy 0 < v < 1 (a sub-luminal recession speed).")
    t = np.asarray(t, dtype=float)
    r = r0 + v * t
    energy_total = kappa * r
    break_times = np.array([(2.0 * m_q * k / kappa - r0) / v for k in range(1, n_breaks + 1)])
    break_times = break_times[break_times >= 0.0]
    n_segments = 1 + np.searchsorted(break_times, t, side="right")
    energy_per_segment = energy_total / n_segments
    return {
        "t": t,
        "r": r,
        "energy_total": energy_total,
        "n_segments": n_segments,
        "energy_per_segment": energy_per_segment,
        "break_times": break_times,
        "r_break_unit": 2.0 * m_q / kappa,
        "kappa": kappa,
        "m_q": m_q,
    }
