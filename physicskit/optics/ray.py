"""Geometric ray optics: paraxial ray transfer (ABCD) matrices and optical systems.

In the paraxial approximation, a light ray at a given plane along the
optical axis is fully described by two numbers -- its height above the
axis, :math:`y`, and the angle it makes with the axis, :math:`\\theta`
(in radians, small-angle/paraxial regime) -- collected into a state vector
:math:`(y, \\theta)^T`. Every simple optical element (propagation through
free space, refraction at an interface, a thin or thick lens, a curved
mirror, a graded-index medium, ...) acts on this vector as a linear map, a
:math:`2\\times 2` "ABCD" matrix:

.. math::

    \\begin{pmatrix} y_{\\text{out}} \\\\ \\theta_{\\text{out}} \\end{pmatrix}
    = \\begin{pmatrix} A & B \\\\ C & D \\end{pmatrix}
      \\begin{pmatrix} y_{\\text{in}} \\\\ \\theta_{\\text{in}} \\end{pmatrix}.

This formalism -- developed piecemeal through the 19th century (Gauss's
theory of optical systems) and formalized for laser resonator design by
Kogelnik and Li in the 1960s -- reduces the analysis of an arbitrarily long
chain of lenses, mirrors, and gaps to ordinary matrix multiplication: the
matrix of a compound system is just the product of the matrices of its
elements, applied in the order light encounters them (rightmost first).
The same matrices reappear in :mod:`physicskit.optics.gaussian` to
propagate the complex beam parameter of a Gaussian laser beam, and their
trace controls whether a laser cavity is a stable resonator.
"""

from __future__ import annotations

import numpy as np

__all__ = [
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
]


def free_space(d):
    """ABCD matrix for propagation through a distance ``d`` of free space (or any homogeneous medium).

    .. math::

        M = \\begin{pmatrix} 1 & d \\\\ 0 & 1 \\end{pmatrix}

    A ray's height changes in proportion to its angle and the distance
    traveled; its angle is unchanged.

    Parameters
    ----------
    d : float
        Propagation distance.

    Returns
    -------
    ndarray of shape (2, 2)

    Examples
    --------
    >>> free_space(2.0)
    array([[1., 2.],
           [0., 1.]])
    """
    return np.array([[1.0, float(d)], [0.0, 1.0]])


def thin_lens(f):
    """ABCD matrix for a thin lens of focal length ``f``.

    .. math::

        M = \\begin{pmatrix} 1 & 0 \\\\ -1/f & 1 \\end{pmatrix}

    Parameters
    ----------
    f : float
        Focal length (positive for a converging lens, negative for
        diverging).

    Returns
    -------
    ndarray of shape (2, 2)

    Examples
    --------
    >>> thin_lens(0.5)
    array([[ 1.,  0.],
           [-2.,  1.]])
    """
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def flat_interface(n1, n2):
    """ABCD matrix for refraction at a flat interface, index ``n1`` to ``n2``.

    .. math::

        M = \\begin{pmatrix} 1 & 0 \\\\ 0 & n_1/n_2 \\end{pmatrix}

    Parameters
    ----------
    n1 : float
        Refractive index of the incident medium.
    n2 : float
        Refractive index of the transmitted medium.

    Returns
    -------
    ndarray of shape (2, 2)

    Examples
    --------
    >>> flat_interface(1.0, 1.5)
    array([[1.        , 0.        ],
           [0.        , 0.66666667]])
    """
    return np.array([[1.0, 0.0], [0.0, n1 / n2]])


def curved_interface(R, n1, n2):
    """ABCD matrix for refraction at a spherical interface of radius ``R``, index ``n1`` to ``n2``.

    .. math::

        M = \\begin{pmatrix} 1 & 0 \\\\ (n_1-n_2)/(R n_2) & n_1/n_2 \\end{pmatrix}

    Parameters
    ----------
    R : float
        Radius of curvature of the interface, positive if the center of
        curvature lies on the outgoing (transmitted) side of the surface.
    n1 : float
        Refractive index of the incident medium.
    n2 : float
        Refractive index of the transmitted medium.

    Returns
    -------
    ndarray of shape (2, 2)

    See Also
    --------
    flat_interface : The :math:`R \\to \\infty` limit.

    Examples
    --------
    >>> np.allclose(curved_interface(1.0e12, 1.0, 1.5), flat_interface(1.0, 1.5), atol=1e-6)
    True
    """
    return np.array([[1.0, 0.0], [(n1 - n2) / (R * n2), n1 / n2]])


def thick_lens(R1, R2, t, n, n_ext=1.0):
    """ABCD matrix for a thick lens: two curved interfaces separated by thickness ``t``.

    Composes, in the order light passes through them, refraction into the
    lens at the first surface, propagation across the lens body, and
    refraction back out at the second surface:

    .. math::

        M = M_{R_2}\\,M_t\\,M_{R_1}, \\qquad
        M_{R_1} = \\text{curved\\_interface}(R_1, n_{\\text{ext}}, n), \\\\
        M_t = \\text{free\\_space}(t) \\ \\text{(in medium } n\\text{)}, \\qquad
        M_{R_2} = \\text{curved\\_interface}(R_2, n, n_{\\text{ext}}).

    Parameters
    ----------
    R1 : float
        Radius of curvature of the first (entrance) surface.
    R2 : float
        Radius of curvature of the second (exit) surface.
    t : float
        Center thickness of the lens.
    n : float
        Refractive index of the lens material.
    n_ext : float, default=1.0
        Refractive index of the surrounding medium (air, by default).

    Returns
    -------
    ndarray of shape (2, 2)

    Examples
    --------
    A symmetric biconvex lens in air; the resulting matrix has unit
    determinant, as any lossless ABCD system must:

    >>> M = thick_lens(R1=0.1, R2=-0.1, t=0.01, n=1.5)
    >>> round(float(np.linalg.det(M)), 8)
    1.0
    """
    m1 = curved_interface(R1, n_ext, n)
    m2 = free_space(t)
    m3 = curved_interface(R2, n, n_ext)
    return m3 @ m2 @ m1


def spherical_mirror(R):
    """ABCD matrix for reflection from a spherical mirror of radius of curvature ``R``.

    .. math::

        M = \\begin{pmatrix} 1 & 0 \\\\ -2/R & 1 \\end{pmatrix}

    Parameters
    ----------
    R : float
        Radius of curvature, positive for a concave mirror as seen by the
        incoming ray (i.e. a focusing mirror).

    Returns
    -------
    ndarray of shape (2, 2)

    Examples
    --------
    >>> spherical_mirror(2.0)
    array([[ 1.,  0.],
           [-1.,  1.]])
    """
    return np.array([[1.0, 0.0], [-2.0 / R, 1.0]])


def grin_medium(n0, n2_coeff, d):
    """ABCD matrix for a graded-index (GRIN) medium of length ``d``.

    For the standard quadratic radial index profile

    .. math::

        n(r) = n_0\\left(1 - \\frac{n_2 r^2}{2}\\right),

    paraxial rays oscillate sinusoidally about the axis, giving

    .. math::

        A = D = \\cos(\\sqrt{n_2}\\,d), \\qquad
        B = \\frac{\\sin(\\sqrt{n_2}\\,d)}{n_0\\sqrt{n_2}}, \\qquad
        C = -n_0\\sqrt{n_2}\\,\\sin(\\sqrt{n_2}\\,d).

    Input and output ray angles are measured *outside* the rod, in a
    medium of index 1: this is the inside-the-rod solution
    (:math:`A=D=\\cos`, :math:`B=\\sin/\\sqrt{n_2}`,
    :math:`C=-\\sqrt{n_2}\\sin`) sandwiched between the flat entrance and
    exit faces, ``flat_interface(n0, 1) @ M_inside @ flat_interface(1, n0)``,
    which is where the :math:`n_0` factors come from. As ``n2_coeff``
    :math:`\\to 0`, :math:`A=D\\to 1`, :math:`C\\to 0`, and
    :math:`B \\to d/n_0` -- the familiar reduced thickness of a
    homogeneous slab of index :math:`n_0` in air, equal to
    :func:`free_space` only for :math:`n_0 = 1`.

    Parameters
    ----------
    n0 : float
        On-axis refractive index.
    n2_coeff : float
        Quadratic index-gradient coefficient :math:`n_2` (units of
        1/length^2). Must be non-negative; ``0`` gives a homogeneous
        medium.
    d : float
        Length of the GRIN medium.

    Returns
    -------
    ndarray of shape (2, 2)

    Examples
    --------
    A tiny gradient barely distinguishable from a homogeneous medium of
    index 1 reduces to plain free-space propagation:

    >>> np.allclose(grin_medium(n0=1.0, n2_coeff=1e-8, d=2.0), free_space(2.0), atol=1e-4)
    True
    """
    if n2_coeff <= 0.0:
        return np.array([[1.0, float(d) / n0], [0.0, 1.0]])
    sqrt_n2 = np.sqrt(n2_coeff)
    arg = sqrt_n2 * d
    A = np.cos(arg)
    B = np.sin(arg) / (n0 * sqrt_n2)
    C = -n0 * sqrt_n2 * np.sin(arg)
    return np.array([[A, B], [C, A]])


class OpticalElement:
    """A single named optical element wrapping one ABCD matrix.

    Parameters
    ----------
    matrix : array_like, shape (2, 2)
        The element's ray transfer matrix, e.g. from :func:`thin_lens` or
        :func:`free_space`.
    name : str, default=""
        Human-readable label (e.g. ``"f=50mm lens"``).
    length : float, default=0.0
        Physical length occupied by this element along the optical axis
        (zero for a "thin" element such as a lens or mirror).

    Examples
    --------
    >>> elem = OpticalElement(thin_lens(0.05), name="focusing lens", length=0.0)
    >>> elem.name
    'focusing lens'
    """

    def __init__(self, matrix, name="", length=0.0):
        self.matrix = np.asarray(matrix, dtype=float)
        self.name = name
        self.length = length


class OpticalSystem:
    """An ordered sequence of :class:`OpticalElement` forming a compound optical system.

    Parameters
    ----------
    elements : list of OpticalElement
        The elements in the order light passes through them: ``elements[0]``
        is hit first.

    Examples
    --------
    A single thin lens followed by propagation over its focal length
    focuses any parallel ray bundle back to the axis:

    >>> f = 0.1
    >>> sys = OpticalSystem([
    ...     OpticalElement(thin_lens(f), name="lens"),
    ...     OpticalElement(free_space(f), name="propagate to focus"),
    ... ])
    >>> trajectory = sys.trace_ray(y0=0.01, theta0=0.0)
    >>> abs(float(trajectory[-1, 0])) < 1e-12
    True
    """

    def __init__(self, elements):
        self.elements = list(elements)

    def system_matrix(self):
        """Total ABCD matrix of the system, :math:`M = M_n \\cdots M_2 M_1`.

        ``elements[0]`` is applied first (it is the rightmost factor), so
        it acts on the incoming ray state before any later element.

        Returns
        -------
        ndarray of shape (2, 2)
        """
        M = np.eye(2)
        for element in self.elements:
            M = element.matrix @ M
        return M

    def trace_ray(self, y0, theta0):
        """Trace a single ray through every element, recording its state at each step.

        Parameters
        ----------
        y0 : float
            Initial height.
        theta0 : float
            Initial angle, in radians.

        Returns
        -------
        ndarray of shape (n_elements + 1, 2)
            Row 0 is the input state ``[y0, theta0]``; row ``i`` (for
            ``i >= 1``) is the state after passing through
            ``elements[0], ..., elements[i-1]``.
        """
        n = len(self.elements)
        states = np.empty((n + 1, 2))
        state = np.array([float(y0), float(theta0)])
        states[0] = state
        for i, element in enumerate(self.elements):
            state = element.matrix @ state
            states[i + 1] = state
        return states

    @property
    def stability_parameter(self):
        """Resonator stability parameter :math:`(A+D)/2` of the system matrix.

        Returns
        -------
        float
        """
        M = self.system_matrix()
        return (M[0, 0] + M[1, 1]) / 2.0

    def is_stable(self):
        """Whether the system satisfies the resonator stability condition :math:`|A+D| \\le 2`.

        Returns
        -------
        bool
        """
        return cavity_stability(self.system_matrix())


def cavity_round_trip_matrix(elements):
    """Round-trip ABCD matrix of a laser cavity, given its elements in traversal order.

    Equivalent to ``OpticalSystem(elements).system_matrix()``; provided as
    a standalone function for cavities analyzed without constructing a full
    :class:`OpticalSystem`.

    Parameters
    ----------
    elements : list of OpticalElement
        The elements encountered over one full round trip, in order.

    Returns
    -------
    ndarray of shape (2, 2)

    See Also
    --------
    cavity_stability : Test the resulting matrix for resonator stability.

    Examples
    --------
    >>> M = cavity_round_trip_matrix([
    ...     OpticalElement(spherical_mirror(2.0), name="M1"),
    ...     OpticalElement(free_space(1.0), name="gap"),
    ...     OpticalElement(spherical_mirror(2.0), name="M2"),
    ...     OpticalElement(free_space(1.0), name="gap"),
    ... ])
    >>> cavity_stability(M)
    True
    """
    M = np.eye(2)
    for element in elements:
        M = element.matrix @ M
    return M


def cavity_stability(M):
    """Resonator stability test :math:`|A+D| \\le 2` (equivalently :math:`|\\operatorname{tr} M| \\le 2`).

    A laser cavity with round-trip matrix ``M`` supports stable,
    non-diverging paraxial ray bundles if and only if this holds.

    Parameters
    ----------
    M : array_like, shape (2, 2)
        A round-trip ABCD matrix, e.g. from :func:`cavity_round_trip_matrix`
        or :meth:`OpticalSystem.system_matrix`.

    Returns
    -------
    bool

    Examples
    --------
    >>> cavity_stability(np.array([[1.0, 0.0], [0.0, 1.0]]))
    True
    >>> cavity_stability(np.array([[3.0, 0.0], [0.0, 3.0]]))
    False
    """
    M = np.asarray(M, dtype=float)
    return bool(abs(np.trace(M)) <= 2.0)
