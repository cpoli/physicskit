r"""Toy branching cascades (particle showers, parton showers) and a
schematic detector geometry for displaying their final states.

Uses the same natural-unit convention (:math:`\hbar=c=1`) as
:mod:`physicskit.particle.kinematics`.

- :class:`ShowerParticle` -- a node of a branching-cascade tree.
- :func:`simple_shower` -- a generic toy branching cascade (particle
  collision -> shower).
- :func:`parton_shower` -- a DGLAP-flavored toy branching cascade with
  quark/gluon flavor bookkeeping.
- :func:`flatten_shower`, :func:`shower_leaves` -- tree-traversal helpers.
- :func:`cluster_into_jets` -- group final-state momenta into jets by angle.
- :func:`charged_track_points` -- a charged (or neutral) particle's 2D
  trajectory in a uniform axial magnetic field, for a schematic detector
  event display.

**Shared branching model** (:func:`simple_shower` and
:func:`parton_shower`): each particle carries a "virtuality" mass (its
:class:`~physicskit.particle.kinematics.FourVector` invariant mass,
standing in for how far off its true mass shell it is); a branching is
computed as an *exact* two-body decay (reusing
:func:`~physicskit.particle.decays.two_body_decay`) in the parent's own
rest frame with an isotropically sampled angle, then boosted back to the
lab frame with :func:`~physicskit.particle.kinematics.boost_generic` --
so energy-momentum is conserved at every vertex to machine precision,
for free, by reusing already-tested kinematics building blocks. Daughter
virtualities shrink by a fixed fraction each generation, and branching
stops once either the energy or the virtuality falls below a cutoff
(the shower's "hadronization scale"). This is a deliberately simplified
stand-in for a real fragmentation function / DGLAP splitting kernel
(which would sample an energy-sharing fraction ``z`` and an evolution
variable from Sudakov form factors) -- but the isotropic rest-frame
angle, combined with the boost back to the lab frame, still produces a
genuinely collimated, energy-ordered cascade (small opening angles for
high-energy branches), a real physical consequence of the boost rather
than something imposed by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from physicskit.particle.decays import two_body_decay
from physicskit.particle.kinematics import FourVector, boost_generic

__all__ = [
    "ShowerParticle",
    "simple_shower",
    "parton_shower",
    "flatten_shower",
    "shower_leaves",
    "cluster_into_jets",
    "charged_track_points",
]


@dataclass
class ShowerParticle:
    """One node of a branching-cascade tree.

    Attributes
    ----------
    four_vector : FourVector
        This particle's lab-frame four-momentum.
    generation : int
        Branching depth (0 for the initial/primary particle).
    flavor : str, default=""
        Parton flavor label (``"q"``, ``"qbar"``, ``"g"``); unused (left
        empty) by :func:`simple_shower`.
    parent : ShowerParticle, optional
        The particle this one was produced from (``None`` for the root).
    children : list of ShowerParticle
        The two daughters, if this particle branched further (empty for
        a final-state / leaf particle).
    """

    four_vector: FourVector
    generation: int
    flavor: str = ""
    parent: Optional[ShowerParticle] = None
    children: list = field(default_factory=list)

    @property
    def is_leaf(self):
        """Whether this particle is final-state (did not branch further)."""
        return len(self.children) == 0


def _split_isotropic(parent_four_vector, m1, m2, rng):
    """Exact two-body split of ``parent_four_vector`` (in its own rest frame,
    isotropic angle), boosted back to the lab frame."""
    cos_theta = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    p1_rest, p2_rest = two_body_decay(parent_four_vector.mass, m1, m2, cos_theta, phi)
    beta_vec = parent_four_vector.p_vec / parent_four_vector.E
    return boost_generic(p1_rest, beta_vec), boost_generic(p2_rest, beta_vec)


def _build_shower(E0, m0_fraction, daughter_fraction, E_threshold, max_generations, rng, flavor_rule=None, flavor0=""):
    m0 = m0_fraction * E0
    p0 = FourVector(E0, 0.0, 0.0, float(np.sqrt(max(E0**2 - m0**2, 0.0))))
    root = ShowerParticle(p0, generation=0, flavor=flavor0)

    def branch(node):
        if node.generation >= max_generations or node.four_vector.E < E_threshold or node.four_vector.mass < 1e-9:
            return
        m_parent = node.four_vector.mass
        m1 = m2 = daughter_fraction * m_parent
        if m1 + m2 >= m_parent:
            return
        p1, p2 = _split_isotropic(node.four_vector, m1, m2, rng)
        f1, f2 = flavor_rule(node.flavor, rng) if flavor_rule is not None else ("", "")
        c1 = ShowerParticle(p1, node.generation + 1, flavor=f1, parent=node)
        c2 = ShowerParticle(p2, node.generation + 1, flavor=f2, parent=node)
        node.children = [c1, c2]
        branch(c1)
        branch(c2)

    branch(root)
    return root


def simple_shower(E0, m0_fraction=0.3, daughter_fraction=0.42, E_threshold=1.0, max_generations=8, rng=None):
    r"""A toy branching cascade: a single high-energy particle splitting
    repeatedly until its descendants fall below an energy threshold.

    Each split is an exact 1-to-2 branching (see the module docstring
    for the shared construction); the initial particle is given a toy
    "virtuality" :math:`m_0=` ``m0_fraction`` :math:`{}\times E_0`
    representing how far off-shell it is when it enters the cascade,
    and each generation's daughters inherit a virtuality shrunk by
    ``daughter_fraction`` (< 0.5, so it always halves-or-better,
    guaranteeing the branching stays kinematically allowed).

    Parameters
    ----------
    E0 : float
        Energy of the initial incident particle (along the beam/z axis).
    m0_fraction : float, default=0.3
        Initial virtuality as a fraction of ``E0``.
    daughter_fraction : float, default=0.42
        Fraction of the parent's virtuality inherited by each daughter.
    E_threshold : float, default=1.0
        Energy below which a particle is treated as final-state (stops branching).
    max_generations : int, default=8
        Hard cap on branching depth (safety net against numerical edge cases).
    rng : numpy.random.Generator, optional
        Random number generator; a fresh default one is used if omitted.

    Returns
    -------
    ShowerParticle
        The root of the cascade tree (the initial particle; its
        descendants are reached via ``.children``).

    Examples
    --------
    >>> import numpy as np
    >>> root = simple_shower(100.0, E_threshold=5.0, rng=np.random.default_rng(0))
    >>> leaves = shower_leaves(root)
    >>> from physicskit.particle.kinematics import invariant_mass
    >>> bool(abs(invariant_mass([leaf.four_vector for leaf in leaves]) - root.four_vector.mass) < 1e-6)
    True
    """
    rng = np.random.default_rng() if rng is None else rng
    return _build_shower(E0, m0_fraction, daughter_fraction, E_threshold, max_generations, rng)


def parton_shower(E0, flavor0="q", m0_fraction=0.3, daughter_fraction=0.42, E_threshold=1.0, gluon_splitting_prob=0.2, max_generations=10, rng=None):
    r"""A toy DGLAP-flavored parton shower: quarks radiate gluons, gluons
    split into gluon pairs or quark-antiquark pairs.

    Uses the same exact-two-body-split-and-boost construction as
    :func:`simple_shower` (see the module docstring), with a simplified
    flavor rule standing in for the DGLAP splitting functions
    :math:`P_{qq},P_{gg},P_{qg}`:

    - a quark or antiquark always radiates a gluon (``q -> q g``,
      ``qbar -> qbar g``), the only flavor-conserving 1-to-2 splitting
      available to it;
    - a gluon splits into two gluons with probability
      ``1-gluon_splitting_prob`` (``g -> g g``) or into a quark-antiquark
      pair with probability ``gluon_splitting_prob`` (``g -> q qbar``).

    Parameters
    ----------
    E0 : float
        Energy of the initial parton.
    flavor0 : str, default="q"
        Flavor of the initial parton (``"q"``, ``"qbar"``, or ``"g"``).
    m0_fraction, daughter_fraction, E_threshold, max_generations, rng
        See :func:`simple_shower`.
    gluon_splitting_prob : float, default=0.2
        Probability that a gluon branching produces a quark-antiquark
        pair rather than two gluons.

    Returns
    -------
    ShowerParticle
        The root of the shower tree.

    Examples
    --------
    >>> import numpy as np
    >>> root = parton_shower(100.0, E_threshold=5.0, rng=np.random.default_rng(0))
    >>> leaves = shower_leaves(root)
    >>> from physicskit.particle.kinematics import invariant_mass
    >>> bool(abs(invariant_mass([leaf.four_vector for leaf in leaves]) - root.four_vector.mass) < 1e-6)
    True
    """
    rng = np.random.default_rng() if rng is None else rng

    def flavor_rule(flavor, rng):
        if flavor == "g":
            if rng.uniform() < gluon_splitting_prob:
                return "q", "qbar"
            return "g", "g"
        return flavor, "g"

    return _build_shower(E0, m0_fraction, daughter_fraction, E_threshold, max_generations, rng, flavor_rule=flavor_rule, flavor0=flavor0)


def flatten_shower(root):
    """All nodes of a shower tree (root included), in no particular order.

    Parameters
    ----------
    root : ShowerParticle

    Returns
    -------
    list of ShowerParticle
    """
    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.children)
    return out


def shower_leaves(root):
    """The final-state (non-branching) particles of a shower tree.

    Parameters
    ----------
    root : ShowerParticle

    Returns
    -------
    list of ShowerParticle
    """
    return [node for node in flatten_shower(root) if node.is_leaf]


def cluster_into_jets(leaves, n_jets=2, n_iter=25, seed=0):
    """Group final-state shower particles into jets by momentum direction.

    A minimal, dependency-free k-means clustering on the unit momentum
    directions (i.e. by angular proximity) -- a schematic stand-in for a
    real sequential jet algorithm (:math:`k_T`, anti-:math:`k_T`, ...),
    adequate for visually grouping a toy shower's final state into a
    couple of back-to-back jet cones.

    Parameters
    ----------
    leaves : sequence of ShowerParticle
        Final-state particles, e.g. from :func:`shower_leaves`.
    n_jets : int, default=2
        Number of jets to form (capped at ``len(leaves)``).
    n_iter : int, default=25
        Number of k-means (Lloyd) iterations.
    seed : int, default=0
        Seed for the deterministic random initialization.

    Returns
    -------
    list of list of ShowerParticle
        One list of member particles per jet.

    Examples
    --------
    >>> import numpy as np
    >>> root = parton_shower(100.0, E_threshold=5.0, rng=np.random.default_rng(0))
    >>> jets = cluster_into_jets(shower_leaves(root), n_jets=2)
    >>> len(jets) == 2
    True
    """
    directions = np.array([[leaf.four_vector.px, leaf.four_vector.py, leaf.four_vector.pz] for leaf in leaves])
    norm = np.linalg.norm(directions, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    unit = directions / norm
    n_jets = min(n_jets, len(leaves))
    rng = np.random.default_rng(seed)
    centers = unit[rng.choice(len(unit), size=n_jets, replace=False)].copy()
    labels = np.zeros(len(unit), dtype=int)
    for _ in range(n_iter):
        labels = np.argmax(unit @ centers.T, axis=1)
        for k in range(n_jets):
            members = unit[labels == k]
            if len(members) > 0:
                c = members.mean(axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm > 0:
                    centers[k] = c / c_norm
    return [[leaves[i] for i in range(len(leaves)) if labels[i] == k] for k in range(n_jets)]


def charged_track_points(four_vector, charge, B, vertex=(0.0, 0.0), n_points=100, path_length=1.0):
    r"""2D transverse-plane trajectory of a particle in a uniform axial magnetic field.

    A charged particle with transverse momentum :math:`p_T` follows a
    circular arc of radius

    .. math::

        r = \frac{p_T}{qB}

    the standard relation for the curvature of a charged track in a
    uniform axial field (see e.g. the Particle Data Group's "Passage of
    particles through matter" review); here in schematic natural units
    where :math:`q` is in units of the elementary charge and :math:`B`
    is scaled so that :math:`r` comes out directly in the detector's
    length unit (the familiar practical-unit form is :math:`r[{\rm m}]
    =p_T[{\rm GeV}/c]/(0.3\,B[{\rm T}]\,q[e])`). A neutral particle
    (``charge=0``) instead follows a straight line.

    Parameters
    ----------
    four_vector : FourVector
        The particle's four-momentum.
    charge : float
        Charge, in units of the elementary charge (0 for neutral).
    B : float
        Magnetic field along :math:`+z`, out of the transverse plane.
        The Lorentz force :math:`q\,\mathbf{v}\times\mathbf{B}` then bends
        positive charges clockwise (for ``B > 0``) and negative charges
        counterclockwise, as seen looking down the :math:`z` axis.
    vertex : array_like of shape (2,), default=(0, 0)
        Starting point of the track.
    n_points : int, default=100
        Number of points to sample along the trajectory.
    path_length : float, default=1.0
        For a neutral track, the length of the straight segment drawn;
        for a charged track, the arc length traced out (capped
        implicitly by how many points are requested along it).

    Returns
    -------
    ndarray of shape (n_points, 2)
        Points along the trajectory, starting at ``vertex`` and moving
        off in ``four_vector``'s initial transverse direction.

    Examples
    --------
    >>> from physicskit.particle.kinematics import FourVector
    >>> p = FourVector(10.0, 3.0, 0.0, 5.0)
    >>> pts = charged_track_points(p, charge=0.0, B=1.0, path_length=2.0)
    >>> np.allclose(pts[0], [0.0, 0.0]) and np.allclose(pts[-1], [2.0, 0.0])
    True
    """
    vertex = np.asarray(vertex, dtype=float)
    px, py = four_vector.px, four_vector.py
    pT = float(np.hypot(px, py))
    if charge == 0.0 or pT == 0.0:
        direction = np.array([px, py]) / pT if pT > 0.0 else np.array([1.0, 0.0])
        s = np.linspace(0.0, path_length, n_points)
        return vertex + np.outer(s, direction)
    r = pT / (abs(charge) * B)
    phi0 = np.arctan2(py, px)
    curve_sign = -float(np.sign(charge * B))  # +1: counterclockwise; q v x B turns q*B > 0 clockwise
    center = vertex + r * np.array([-np.sin(phi0), np.cos(phi0)]) * curve_sign
    max_angle = min(path_length / r, 2.0 * np.pi)
    thetas = np.linspace(0.0, max_angle, n_points)
    v0 = vertex - center
    cs, sn = np.cos(curve_sign * thetas), np.sin(curve_sign * thetas)
    xs = center[0] + v0[0] * cs - v0[1] * sn
    ys = center[1] + v0[0] * sn + v0[1] * cs
    return np.column_stack([xs, ys])
