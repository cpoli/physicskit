"""Single-particle motion in electromagnetic fields: the Boris pusher and guiding-center drifts.

Two complementary pictures of charged-particle motion in a magnetized
plasma live here. The *full-orbit* picture integrates the exact Lorentz
force :math:`m\\dot{\\mathbf{v}} = q(\\mathbf{E} + \\mathbf{v}\\times\\mathbf{B})`
with :func:`boris_push` / :func:`boris_integrate`, Jay Boris's 1970
energy-conserving leapfrog scheme -- still the workhorse integrator of
every particle-in-cell (PIC) plasma code. The *guiding-center* picture
instead averages over the fast gyration and tracks only the slow drift of
the orbit's center: :func:`exb_drift`, :func:`grad_b_drift`, and
:func:`curvature_drift` give that drift velocity directly, while
:func:`magnetic_moment` and :func:`mirror_force` capture the adiabatic
invariant :math:`\\mu = m v_\\perp^2 / 2B` responsible for magnetic-mirror
confinement.

The Boris kernels are compiled with Numba for particle-in-cell-scale
performance; every other function here is a closed-form NumPy expression.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from physicskit import constants as _const

__all__ = [
    "QE",
    "ME",
    "MP",
    "cyclotron_frequency",
    "larmor_radius",
    "magnetic_moment",
    "boris_push",
    "boris_integrate",
    "exb_drift",
    "grad_b_drift",
    "curvature_drift",
    "mirror_force",
    "magnetic_mirror_bounce",
]

QE = _const.ELEMENTARY_CHARGE
"""Elementary charge, in Coulombs."""

ME = _const.ELECTRON_MASS
"""Electron mass, in kilograms."""

MP = _const.PROTON_MASS
"""Proton mass, in kilograms."""


def cyclotron_frequency(q: float, m: float, B: float) -> float:
    """Angular cyclotron (gyration) frequency :math:`\\omega_c = qB/m`.

    Parameters
    ----------
    q : float
        Particle charge in Coulombs (signed).
    m : float
        Particle mass in kilograms.
    B : float
        Magnetic field magnitude in Tesla.

    Returns
    -------
    float
        Signed angular gyrofrequency in rad/s (negative for negatively
        charged particles, reflecting the opposite sense of rotation).

    See Also
    --------
    larmor_radius : The orbit radius set by this frequency.

    Examples
    --------
    A proton in a 1 T field gyrates about 15.3 MHz:

    >>> round(cyclotron_frequency(QE, MP, 1.0) / (2 * 3.141592653589793) / 1e6, 2)
    15.25
    """
    return q * B / m


def larmor_radius(v_perp: float, q: float, m: float, B: float) -> float:
    """Larmor (gyro) radius :math:`r_L = m v_\\perp / (|q| B)`.

    Parameters
    ----------
    v_perp : float
        Speed perpendicular to the magnetic field, in m/s.
    q : float
        Particle charge in Coulombs (sign is ignored).
    m : float
        Particle mass in kilograms.
    B : float
        Magnetic field magnitude in Tesla.

    Returns
    -------
    float
        Gyroradius in meters.

    Examples
    --------
    >>> round(larmor_radius(1e5, QE, MP, 1.0), 6)
    0.001044
    """
    return m * v_perp / (abs(q) * B)


def magnetic_moment(v_perp: float, m: float, B: float) -> float:
    """The first adiabatic invariant :math:`\\mu = m v_\\perp^2 / (2B)`.

    Conserved for a charged particle whose gyration is fast compared to
    any change in the field it sees, :math:`\\mu` acts as a magnetic
    "potential energy per unit field": as the particle drifts into
    stronger :math:`B`, :math:`v_\\perp` must grow to keep :math:`\\mu`
    fixed, converting parallel kinetic energy into perpendicular kinetic
    energy. That conversion is the mechanism behind :func:`mirror_force`.

    Parameters
    ----------
    v_perp : float
        Speed perpendicular to the magnetic field, in m/s.
    m : float
        Particle mass in kilograms.
    B : float
        Magnetic field magnitude in Tesla.

    Returns
    -------
    float
        Magnetic moment in Joules/Tesla.

    See Also
    --------
    mirror_force : The parallel force derived from this invariant.

    Examples
    --------
    >>> round(magnetic_moment(1e5, MP, 1.0) * 1e18, 4)
    8.3631
    """
    return 0.5 * m * v_perp**2 / B


@njit(cache=True)
def _boris_kernel(x, y, z, vx, vy, vz, q, m, Ex, Ey, Ez, Bx, By, Bz, dt):
    qmdt2 = (q / m) * (dt / 2.0)
    vmx = vx + qmdt2 * Ex
    vmy = vy + qmdt2 * Ey
    vmz = vz + qmdt2 * Ez
    tx, ty, tz = qmdt2 * Bx, qmdt2 * By, qmdt2 * Bz
    t2 = tx * tx + ty * ty + tz * tz
    denom = 1.0 + t2
    sx, sy, sz = 2.0 * tx / denom, 2.0 * ty / denom, 2.0 * tz / denom
    vpx = vmx + (vmy * tz - vmz * ty)
    vpy = vmy + (vmz * tx - vmx * tz)
    vpz = vmz + (vmx * ty - vmy * tx)
    vplusx = vmx + (vpy * sz - vpz * sy)
    vplusy = vmy + (vpz * sx - vpx * sz)
    vplusz = vmz + (vpx * sy - vpy * sx)
    vnx = vplusx + qmdt2 * Ex
    vny = vplusy + qmdt2 * Ey
    vnz = vplusz + qmdt2 * Ez
    xn = x + vnx * dt
    yn = y + vny * dt
    zn = z + vnz * dt
    return xn, yn, zn, vnx, vny, vnz


@njit(cache=True)
def _boris_integrate_kernel(x, y, z, vx, vy, vz, q, m, Ex, Ey, Ez, Bx, By, Bz, dt, steps):
    xs = np.empty(steps + 1)
    ys = np.empty(steps + 1)
    zs = np.empty(steps + 1)
    vxs = np.empty(steps + 1)
    vys = np.empty(steps + 1)
    vzs = np.empty(steps + 1)
    xs[0], ys[0], zs[0] = x, y, z
    vxs[0], vys[0], vzs[0] = vx, vy, vz
    for i in range(steps):
        x, y, z, vx, vy, vz = _boris_kernel(x, y, z, vx, vy, vz, q, m, Ex, Ey, Ez, Bx, By, Bz, dt)
        xs[i + 1], ys[i + 1], zs[i + 1] = x, y, z
        vxs[i + 1], vys[i + 1], vzs[i + 1] = vx, vy, vz
    return xs, ys, zs, vxs, vys, vzs


def boris_push(pos: np.ndarray, vel: np.ndarray, q: float, m: float, E: np.ndarray, B: np.ndarray, dt: float) -> tuple:
    """Advance a charged particle one leapfrog step with the Boris integrator.

    The Boris (1970) scheme splits each step into an electric
    half-acceleration, an exact rotation about :math:`\\mathbf{B}` (via the
    Boris "E-cross-B" trick, avoiding any explicit trigonometry), and a
    second electric half-acceleration. The rotation step preserves speed
    exactly in a pure magnetic field, which is why the scheme conserves
    kinetic energy over arbitrarily many gyro-orbits where a naive
    forward-Euler or even RK4 update would spiral outward.

    Parameters
    ----------
    pos : ndarray, shape (3,)
        Particle position :math:`[x, y, z]` in meters.
    vel : ndarray, shape (3,)
        Particle velocity :math:`[v_x, v_y, v_z]` in m/s.
    q : float
        Particle charge in Coulombs.
    m : float
        Particle mass in kilograms.
    E : ndarray, shape (3,)
        Electric field :math:`[E_x, E_y, E_z]` in V/m.
    B : ndarray, shape (3,)
        Magnetic field :math:`[B_x, B_y, B_z]` in Tesla.
    dt : float
        Time step in seconds.

    Returns
    -------
    pos_new, vel_new : ndarray, shape (3,)
        Position and velocity after one step of size ``dt``.

    See Also
    --------
    boris_integrate : Repeated application of this step over many time steps.

    Examples
    --------
    A proton launched perpendicular to a uniform field gyrates without
    gaining or losing speed:

    >>> import numpy as np
    >>> r0 = np.array([0.0, 0.0, 0.0])
    >>> v0 = np.array([1e5, 0.0, 0.0])
    >>> E = np.zeros(3)
    >>> B = np.array([0.0, 0.0, 1.0])
    >>> r1, v1 = boris_push(r0, v0, q=QE, m=MP, E=E, B=B, dt=1e-9)
    >>> round(float(np.linalg.norm(v1)), 6)
    100000.0
    """
    xn, yn, zn, vnx, vny, vnz = _boris_kernel(
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(vel[0]),
        float(vel[1]),
        float(vel[2]),
        float(q),
        float(m),
        float(E[0]),
        float(E[1]),
        float(E[2]),
        float(B[0]),
        float(B[1]),
        float(B[2]),
        float(dt),
    )
    return np.array([xn, yn, zn]), np.array([vnx, vny, vnz])


def boris_integrate(pos0: np.ndarray, vel0: np.ndarray, q: float, m: float, E: np.ndarray, B: np.ndarray, dt: float, steps: int) -> tuple:
    """Integrate a charged particle's trajectory in *uniform* E and B fields with the Boris pusher.

    Repeatedly applies :func:`boris_push` inside a single Numba-compiled
    loop, avoiding Python-level overhead per step -- the throughput this
    buys is what makes the Boris scheme practical for particle-in-cell
    codes tracking millions of particles.

    Parameters
    ----------
    pos0 : ndarray, shape (3,)
        Initial position in meters.
    vel0 : ndarray, shape (3,)
        Initial velocity in m/s.
    q : float
        Particle charge in Coulombs.
    m : float
        Particle mass in kilograms.
    E : ndarray, shape (3,)
        Uniform electric field in V/m.
    B : ndarray, shape (3,)
        Uniform magnetic field in Tesla.
    dt : float
        Time step in seconds.
    steps : int
        Number of steps to advance.

    Returns
    -------
    pos_hist : ndarray, shape (steps + 1, 3)
        Position at every step, including the initial condition.
    vel_hist : ndarray, shape (steps + 1, 3)
        Velocity at every step, including the initial condition.

    See Also
    --------
    boris_push : The single-step update repeated here.
    exb_drift : The closed-form drift velocity this trajectory averages to
        when a non-zero E is present.

    Examples
    --------
    Kinetic energy is conserved to machine precision over many gyrations
    in a pure magnetic field:

    >>> import numpy as np
    >>> B = np.array([0.0, 0.0, 1.0])
    >>> E = np.zeros(3)
    >>> omega_c = cyclotron_frequency(QE, MP, 1.0)
    >>> dt = (2 * np.pi / omega_c) / 200
    >>> pos_hist, vel_hist = boris_integrate(
    ...     np.zeros(3), np.array([1e5, 0.0, 0.0]), QE, MP, E, B, dt, steps=2000
    ... )
    >>> speeds = np.linalg.norm(vel_hist, axis=1)
    >>> bool(np.max(np.abs(speeds - 1e5)) < 1e-6)
    True
    """
    xs, ys, zs, vxs, vys, vzs = _boris_integrate_kernel(
        float(pos0[0]),
        float(pos0[1]),
        float(pos0[2]),
        float(vel0[0]),
        float(vel0[1]),
        float(vel0[2]),
        float(q),
        float(m),
        float(E[0]),
        float(E[1]),
        float(E[2]),
        float(B[0]),
        float(B[1]),
        float(B[2]),
        float(dt),
        int(steps),
    )
    pos_hist = np.stack([xs, ys, zs], axis=1)
    vel_hist = np.stack([vxs, vys, vzs], axis=1)
    return pos_hist, vel_hist


def exb_drift(E: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The :math:`\\mathbf{E}\\times\\mathbf{B}` drift velocity :math:`\\mathbf{v}_E = (\\mathbf{E}\\times\\mathbf{B})/B^2`.

    Unlike every other guiding-center drift, this one is independent of
    charge, mass, and energy: electrons and ions drift together at the
    same velocity, so :math:`\\mathbf{E}\\times\\mathbf{B}` drift carries no
    net current and instead advects the whole plasma as a fluid --
    the drift underlying tokamak radial-electric-field rotation and
    magnetospheric convection alike.

    Parameters
    ----------
    E : ndarray, shape (3,)
        Electric field in V/m.
    B : ndarray, shape (3,)
        Magnetic field in Tesla.

    Returns
    -------
    ndarray, shape (3,)
        Drift velocity in m/s.

    Examples
    --------
    >>> import numpy as np
    >>> E = np.array([0.0, 1e3, 0.0])
    >>> B = np.array([0.0, 0.0, 1.0])
    >>> exb_drift(E, B)
    array([1000.,    0.,    0.])
    """
    return np.cross(E, B) / np.dot(B, B)


def grad_b_drift(v_perp: float, q: float, m: float, B: np.ndarray, grad_B: np.ndarray) -> np.ndarray:
    """The grad-B drift :math:`\\mathbf{v}_{\\nabla B} = \\dfrac{m v_\\perp^2}{2qB^3}(\\mathbf{B}\\times\\nabla B)`.

    A particle gyrating in a field whose magnitude varies across the
    orbit sees a tighter turn (smaller Larmor radius) on the strong-field
    side than the weak-field side, so the orbit fails to close and the
    guiding center creeps sideways. Because the drift is inversely
    proportional to charge, electrons and ions drift in *opposite*
    directions, producing a net current -- the origin of the ring current
    in planetary magnetospheres.

    Parameters
    ----------
    v_perp : float
        Speed perpendicular to the field, in m/s.
    q : float
        Particle charge in Coulombs (signed).
    m : float
        Particle mass in kilograms.
    B : ndarray, shape (3,)
        Local magnetic field vector in Tesla.
    grad_B : ndarray, shape (3,)
        Gradient of the field magnitude, :math:`\\nabla |\\mathbf{B}|`, in Tesla/meter.

    Returns
    -------
    ndarray, shape (3,)
        Drift velocity in m/s.

    See Also
    --------
    curvature_drift : The companion drift from field-line curvature,
        which combines with this one in any real toroidal field.

    Examples
    --------
    >>> import numpy as np
    >>> B = np.array([0.0, 0.0, 1.0])
    >>> grad_B = np.array([0.1, 0.0, 0.0])
    >>> drift = grad_b_drift(1e5, QE, MP, B, grad_B)
    >>> round(float(drift[1]), 4)
    5.2198
    """
    Bmag = np.linalg.norm(B)
    return (m * v_perp**2) / (2.0 * q * Bmag**3) * np.cross(B, grad_B)


def curvature_drift(v_par: float, q: float, m: float, B: np.ndarray, R_c: np.ndarray) -> np.ndarray:
    """The curvature drift :math:`\\mathbf{v}_R = \\dfrac{m v_\\parallel^2}{q}\\dfrac{\\mathbf{R}_c\\times\\mathbf{B}}{R_c^2 B^2}`.

    A particle streaming along a curved field line feels a centrifugal
    force in the guiding-center frame, which -- crossed with
    :math:`\\mathbf{B}` -- produces a drift perpendicular to the plane of
    curvature. In a low-beta toroidal equilibrium this combines with
    :func:`grad_b_drift` (curvature and field-strength gradients point the
    same way when :math:`\\nabla\\times\\mathbf{B}=0` in vacuum) into the
    single :math:`\\nabla B` + curvature drift responsible for charge
    separation and the resulting E-cross-B rotation in a tokamak.

    Parameters
    ----------
    v_par : float
        Speed parallel to the field, in m/s.
    q : float
        Particle charge in Coulombs (signed).
    m : float
        Particle mass in kilograms.
    B : ndarray, shape (3,)
        Local magnetic field vector in Tesla.
    R_c : ndarray, shape (3,)
        Radius-of-curvature vector, pointing from the field line's local
        center of curvature to the particle, magnitude :math:`R_c` in meters.

    Returns
    -------
    ndarray, shape (3,)
        Drift velocity in m/s.

    Examples
    --------
    >>> import numpy as np
    >>> B = np.array([0.0, 0.0, 1.0])
    >>> R_c = np.array([1.0, 0.0, 0.0])
    >>> drift = curvature_drift(1e5, QE, MP, B, R_c)
    >>> round(float(drift[1]), 2)
    -104.4
    """
    Bmag = np.linalg.norm(B)
    Rc2 = np.dot(R_c, R_c)
    return (m * v_par**2 / q) * np.cross(R_c, B) / (Rc2 * Bmag**2)


def mirror_force(mu: float, grad_B_parallel: float) -> float:
    """The mirror force :math:`F_\\parallel = -\\mu\\, \\partial B/\\partial \\ell` along a field line.

    As a particle's guiding center moves into a region of stronger field
    (larger :math:`\\partial B/\\partial\\ell`), the conservation of
    :func:`magnetic_moment` forces :math:`v_\\perp` to grow at the expense
    of :math:`v_\\parallel`; this is the reaction force decelerating the
    parallel motion. If the field is strong enough, :math:`v_\\parallel`
    reaches zero before the particle passes the throat and it reflects --
    magnetic mirror confinement, simulated end to end in
    :func:`magnetic_mirror_bounce`.

    Parameters
    ----------
    mu : float
        Magnetic moment in J/T, from :func:`magnetic_moment`.
    grad_B_parallel : float
        Gradient of the field magnitude along the field line, in Tesla/meter.

    Returns
    -------
    float
        Force in Newtons, directed to push the particle toward weaker field.

    Examples
    --------
    >>> mirror_force(mu=1e-17, grad_B_parallel=2.0)
    -2e-17
    """
    return -mu * grad_B_parallel


def magnetic_mirror_bounce(z0: float, v_par0: float, v_perp0: float, m: float, B_func, dz: float = 1e-6, dt: float = 1e-10, steps: int = 20000) -> tuple:
    """Simulate 1D guiding-center bounce motion between the throats of a magnetic mirror.

    Integrates :math:`m\\dot{v}_\\parallel = -\\mu\\, dB/dz` with :math:`\\mu`
    fixed at its initial value (the adiabatic invariant), using a
    symmetric leapfrog step. A particle with too little pitch angle to
    reflect before reaching the mirror throat's peak field instead falls
    into the *loss cone* and would be lost from confinement in a real
    device; :func:`mirror_force` supplies the underlying force law.

    Parameters
    ----------
    z0 : float
        Initial position along the field line, in meters.
    v_par0 : float
        Initial parallel velocity, in m/s.
    v_perp0 : float
        Initial perpendicular velocity, in m/s (sets :math:`\\mu` via :func:`magnetic_moment`).
    m : float
        Particle mass in kilograms.
    B_func : callable
        Field-strength profile ``B_func(z) -> float`` along the field line, in Tesla.
    dz : float, default=1e-6
        Finite-difference step used to evaluate :math:`dB/dz`, in meters.
    dt : float, default=1e-10
        Time step in seconds.
    steps : int, default=20000
        Number of leapfrog steps to advance.

    Returns
    -------
    z_hist : ndarray, shape (steps + 1,)
        Position along the field line at every step.
    v_par_hist : ndarray, shape (steps + 1,)
        Parallel velocity at every step.

    Examples
    --------
    A particle launched from the mirror midplane with enough perpendicular
    energy reflects before reaching the throat, reversing the sign of its
    parallel velocity:

    >>> import numpy as np
    >>> B_func = lambda z: 1.0 + 4.0 * (z / 0.05) ** 2
    >>> z_hist, v_par_hist = magnetic_mirror_bounce(
    ...     z0=0.0, v_par0=2e4, v_perp0=8e4, m=MP, B_func=B_func, steps=6000
    ... )
    >>> bool(v_par_hist[0] > 0 and v_par_hist[-1] < 0)
    True
    """
    z = z0
    v_par = v_par0
    B0 = B_func(z0)
    mu = magnetic_moment(v_perp0, m, B0)
    z_hist = np.empty(steps + 1)
    v_par_hist = np.empty(steps + 1)
    z_hist[0], v_par_hist[0] = z, v_par
    for i in range(steps):
        grad_B = (B_func(z + dz) - B_func(z - dz)) / (2.0 * dz)
        a = mirror_force(mu, grad_B) / m
        v_par = v_par + a * dt
        z = z + v_par * dt
        z_hist[i + 1] = z
        v_par_hist[i + 1] = v_par
    return z_hist, v_par_hist
