"""Plasma-wakefield-style particle acceleration: a test charge surfing a prescribed traveling wave.

A full plasma-wakefield-accelerator simulation would derive the
accelerating field self-consistently from a driver beam or laser pulse
depositing momentum into the plasma via PIC (as :mod:`physicskit.plasma.kinetic`
does for electrostatic waves, just relativistically and in the wake of a
driver). That is a large undertaking; the standard pedagogical
simplification -- used throughout introductory wakefield-acceleration
treatments -- is instead to *prescribe* the wakefield as a traveling
longitudinal electric-field wave :math:`E_z(x,t)=E_0\\cos[k(x-v_{ph}t)]`
(a rigid wake moving at the driver's phase velocity :math:`v_{ph}`, not fed
back on by the accelerated particle) and inject a single test charge into
it with :func:`physicskit.plasma.single_particle.boris_push` (setting
:math:`\\mathbf{B}=0`, since this toy model has no transverse focusing
field). A particle riding near the wave's phase velocity feels a
sign-definite accelerating field for as long as it stays within one
quarter-wavelength of the field's zero-crossing (the "accelerating bucket"),
gaining energy at the expense of the (externally prescribed, infinite-energy-
reservoir) wave, exactly the qualitative picture of plasma wakefield
acceleration without any of its self-consistent field generation.
"""

from __future__ import annotations

import numpy as np

from physicskit.plasma.single_particle import boris_push

__all__ = [
    "wakefield_e_field",
    "simulate_wakefield_acceleration",
]


def wakefield_e_field(x: np.ndarray, t: float, E0: float, k: float, v_phase: float) -> np.ndarray:
    """Prescribed traveling-wave longitudinal wakefield :math:`E_z(x,t) = E_0\\cos[k(x - v_{ph}t)]`.

    A rigid sinusoidal accelerating structure moving at the fixed phase
    velocity `v_phase`, standing in for the self-consistent plasma
    wakefield a real driver beam or laser wake would generate (see the
    module docstring for the simplification this represents).

    Parameters
    ----------
    x : ndarray or float
        Position(s) at which to evaluate the field.
    t : float
        Time.
    E0 : float
        Peak field amplitude.
    k : float
        Wakefield wavenumber.
    v_phase : float
        Phase velocity of the traveling wave.

    Returns
    -------
    ndarray or float
        :math:`E_z(x, t)`, same shape as `x`.

    See Also
    --------
    simulate_wakefield_acceleration : Injects a test particle into this field.

    Examples
    --------
    >>> round(float(wakefield_e_field(0.0, t=0.0, E0=1.0, k=1.0, v_phase=1.0)), 6)
    1.0
    """
    return E0 * np.cos(k * (x - v_phase * t))


def simulate_wakefield_acceleration(x0: float, v0: float, q: float, m: float, E0: float, k: float, v_phase: float, dt: float, steps: int) -> dict:
    """Integrate a test charge's 1D motion in the prescribed traveling wakefield, with the Boris pusher.

    Calls :func:`physicskit.plasma.single_particle.boris_push` once per
    step with :math:`\\mathbf{B}=0` and :math:`\\mathbf{E}=(E_z(x,t),0,0)`
    re-evaluated at the particle's *current* position and time each
    step -- unlike :func:`physicskit.plasma.single_particle.boris_integrate`,
    which assumes a field uniform in space and time, this wakefield varies
    in both, so the field must be recomputed every step rather than
    baked into a single Numba-compiled loop. The pusher itself remains
    exactly energy-conserving in a pure magnetic field and exact-leapfrog
    in a pure electric field; it is otherwise an ordinary (non-relativistic)
    classical integrator, so results should be read qualitatively once
    the particle's speed approaches a meaningful fraction of `v_phase`
    (real wakefield acceleration is an intrinsically relativistic problem).

    Parameters
    ----------
    x0 : float
        Initial position along the wave.
    v0 : float
        Initial longitudinal velocity.
    q : float
        Particle charge.
    m : float
        Particle mass.
    E0 : float
        Peak wakefield amplitude, from :func:`wakefield_e_field`.
    k : float
        Wakefield wavenumber.
    v_phase : float
        Wakefield phase velocity.
    dt : float
        Time step.
    steps : int
        Number of steps to advance.

    Returns
    -------
    dict
        ``{"t": ndarray shape (steps+1,), "x": position history, "v":
        longitudinal velocity history, "kinetic_energy": :math:`\\tfrac12 m v^2`
        history}``.

    See Also
    --------
    wakefield_e_field : The prescribed field this particle rides.

    Examples
    --------
    A particle launched exactly at the wave's phase velocity, sitting on
    the accelerating part of the field, gains kinetic energy:

    >>> import numpy as np
    >>> result = simulate_wakefield_acceleration(
    ...     x0=0.0, v0=0.9, q=1.0, m=1.0, E0=0.05, k=1.0, v_phase=1.0, dt=0.01, steps=2000
    ... )
    >>> bool(result["kinetic_energy"][-1] > result["kinetic_energy"][0])
    True
    """
    pos = np.array([x0, 0.0, 0.0])
    vel = np.array([v0, 0.0, 0.0])
    B = np.zeros(3)

    n = steps + 1
    x_hist = np.empty(n)
    v_hist = np.empty(n)
    x_hist[0], v_hist[0] = pos[0], vel[0]

    t = 0.0
    for i in range(steps):
        Ez = wakefield_e_field(pos[0], t, E0, k, v_phase)
        E = np.array([Ez, 0.0, 0.0])
        pos, vel = boris_push(pos, vel, q, m, E, B, dt)
        t += dt
        x_hist[i + 1], v_hist[i + 1] = pos[0], vel[0]

    t_hist = np.arange(n) * dt
    kinetic_energy = 0.5 * m * v_hist**2
    return {"t": t_hist, "x": x_hist, "v": v_hist, "kinetic_energy": kinetic_energy}
