"""Soliton-bearing nonlinear field equations: KdV, the nonlinear Schrodinger equation, and Sine-Gordon.

Each equation is integrated with a scheme suited to its structure:

- **Korteweg-de Vries** -- Strang-split pseudo-spectral: the stiff linear
  dispersion :math:`\\partial_x^3` is advanced exactly via the FFT, and the
  non-stiff advection :math:`6u\\partial_x u` via RK4.
- **Nonlinear Schrodinger** -- split-step Fourier: exact for the linear
  (kinetic) part and exact for the nonlinear (pointwise phase-rotation) part.
- **Sine-Gordon** -- explicit leapfrog finite differences on the wave equation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "kdv_soliton",
    "kdv_step",
    "kdv_evolve",
    "kdv_evolve_frames",
    "nls_bright_soliton",
    "nls_dark_soliton",
    "nls_evolve",
    "nls_evolve_frames",
    "sine_gordon_kink",
    "sine_gordon_evolve",
    "sine_gordon_evolve_frames",
]


# -- Korteweg-de Vries --------------------------------------------------------


def kdv_soliton(x: np.ndarray, c: float, x0: float = 0.0) -> np.ndarray:
    """Exact single-soliton solution of :math:`u_t + 6uu_x + u_{xxx} = 0` at :math:`t=0`.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    c : float
        Soliton speed (equal to twice its amplitude: amplitude :math:`=c/2`).
    x0 : float, default=0.0
        Initial center position.

    Returns
    -------
    ndarray
        :math:`u(x, 0) = \\tfrac{c}{2}\\,\\mathrm{sech}^2\\!\\big(\\tfrac{\\sqrt{c}}{2}(x-x_0)\\big)`.

    See Also
    --------
    kdv_evolve : Propagate this (or any) initial condition forward in time.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(kdv_soliton(0.0, c=4.0)), 6)
    2.0
    """
    return (c / 2.0) * (1.0 / np.cosh(np.sqrt(c) / 2.0 * (x - x0))) ** 2


def kdv_step(u_hat: np.ndarray, k: np.ndarray, dt: float) -> np.ndarray:
    """Advance the KdV equation in Fourier space by one Strang-split step.

    Parameters
    ----------
    u_hat : ndarray
        Fourier coefficients of :math:`u` (as from ``numpy.fft.fft``).
    k : ndarray
        Angular wavenumbers matching ``u_hat``.
    dt : float
        Time step.

    Returns
    -------
    ndarray
        Fourier coefficients after one step of size ``dt``.
    """
    ik3 = 1j * k**3

    def nonlinear_rhs(uh):
        u = np.real(np.fft.ifft(uh))
        ux = np.real(np.fft.ifft(1j * k * uh))
        return np.fft.fft(-6 * u * ux)

    u_hat = u_hat * np.exp(ik3 * dt / 2)
    k1 = nonlinear_rhs(u_hat)
    k2 = nonlinear_rhs(u_hat + dt / 2 * k1)
    k3 = nonlinear_rhs(u_hat + dt / 2 * k2)
    k4 = nonlinear_rhs(u_hat + dt * k3)
    u_hat = u_hat + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_hat * np.exp(ik3 * dt / 2)


def kdv_evolve(u0: np.ndarray, x: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """Evolve a KdV initial condition forward in time on a periodic domain.

    Parameters
    ----------
    u0 : ndarray
        Initial field, sampled on the periodic grid ``x``.
    x : ndarray
        Uniformly spaced periodic spatial grid.
    dt : float
        Time step.
    steps : int
        Number of steps to advance.

    Returns
    -------
    ndarray
        Field after ``steps * dt`` time units.

    Notes
    -----
    Because KdV solitons are true solitons, two colliding solitons of
    different speed pass through each other **elastically**: each emerges
    from the collision with its original amplitude and speed, shifted only
    in phase -- unlike generic nonlinear waves, which would break up or
    change shape on collision.

    Examples
    --------
    A fast, tall soliton launched behind a slower, shorter one overtakes it;
    both retain their original amplitudes after the collision:

    >>> import numpy as np
    >>> from scipy.signal import find_peaks
    >>> N, L = 512, 60.0
    >>> x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    >>> u0 = kdv_soliton(x, c=9.0, x0=-20) + kdv_soliton(x, c=4.0, x0=-8)
    >>> u = kdv_evolve(u0, x, dt=0.0005, steps=6000)
    >>> peaks, _ = find_peaks(u, height=1.0)
    >>> heights = sorted(u[peaks])
    >>> [round(float(h), 1) for h in heights]
    [2.0, 4.5]
    """
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    u_hat = np.fft.fft(u0)
    for _ in range(steps):
        u_hat = kdv_step(u_hat, k, dt)
    return np.real(np.fft.ifft(u_hat))


def kdv_evolve_frames(u0: np.ndarray, x: np.ndarray, dt: float, steps_per_frame: int, n_frames: int) -> tuple:
    """Evolve a KdV field and record a snapshot every ``steps_per_frame`` steps, for animation.

    A thin looping wrapper around :func:`kdv_evolve` -- no new physics, just
    repeated short calls with each chunk's output fed back in as the next
    chunk's initial condition (valid since :func:`kdv_evolve` is a
    deterministic one-step-at-a-time integrator), so the trajectory can be
    animated frame by frame.

    Parameters
    ----------
    u0 : ndarray
        Initial field.
    x : ndarray
        Uniformly spaced periodic spatial grid.
    dt : float
        Time step.
    steps_per_frame : int
        Number of integration steps between recorded frames.
    n_frames : int
        Number of frames to record after the initial condition.

    Returns
    -------
    frames : ndarray, shape (n_frames + 1, len(x))
        Field at ``t=0`` and after each recorded chunk.
    times : ndarray, shape (n_frames + 1,)

    See Also
    --------
    physicskit.fields.visualizers.animate_field_1d : Render these frames as a line-plot animation.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-30, 30, 256, endpoint=False)
    >>> u0 = kdv_soliton(x, c=4.0, x0=-15)
    >>> frames, times = kdv_evolve_frames(u0, x, dt=0.001, steps_per_frame=200, n_frames=5)
    >>> frames.shape
    (6, 256)
    """
    u = np.array(u0, dtype=float, copy=True)
    frames = [u.copy()]
    times = [0.0]
    for i in range(n_frames):
        u = kdv_evolve(u, x, dt, steps_per_frame)
        frames.append(u.copy())
        times.append((i + 1) * steps_per_frame * dt)
    return np.array(frames), np.array(times)


# -- Nonlinear Schrodinger equation -----------------------------------------


def nls_bright_soliton(x: np.ndarray, t: float, A: float = 1.0, v: float = 0.0, x0: float = 0.0) -> np.ndarray:
    """Exact bright-soliton solution of the focusing NLS equation.

    Solves :math:`i\\psi_t + \\tfrac{1}{2}\\psi_{xx} + |\\psi|^2\\psi = 0`.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    t : float
        Time.
    A : float, default=1.0
        Soliton amplitude.
    v : float, default=0.0
        Soliton velocity.
    x0 : float, default=0.0
        Initial center position.

    Returns
    -------
    ndarray of complex
        :math:`\\psi(x,t) = A\\,\\mathrm{sech}(A(x-x_0-vt))\\, e^{i[v(x-x_0) + (A^2-v^2)t/2]}`.

    See Also
    --------
    nls_dark_soliton : The defocusing-equation counterpart.
    nls_evolve : Propagate this (or any) initial condition numerically.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0])
    >>> abs(nls_bright_soliton(x, t=0.0, A=1.0))
    array([1.])
    """
    envelope = A / np.cosh(A * (x - x0 - v * t))
    phase = v * (x - x0) + (A**2 - v**2) / 2 * t
    return envelope * np.exp(1j * phase)


def nls_dark_soliton(x: np.ndarray, t: float, rho0: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """Exact stationary dark (black) soliton solution of the defocusing NLS equation.

    Solves :math:`i\\psi_t + \\tfrac{1}{2}\\psi_{xx} - |\\psi|^2\\psi = 0`: a
    density notch that vanishes at its center, sitting on a uniform
    background of density ``rho0``.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    t : float
        Time.
    rho0 : float, default=1.0
        Background density far from the soliton.
    x0 : float, default=0.0
        Center position.

    Returns
    -------
    ndarray of complex
        :math:`\\psi(x,t) = \\sqrt{\\rho_0}\\,\\tanh\\!\\big(\\sqrt{\\rho_0/2}\\,(x-x_0)\\big)\\, e^{-i\\rho_0 t}`,
        with healing length :math:`\\xi = 1/\\sqrt{\\rho_0}`.

    See Also
    --------
    nls_bright_soliton : The focusing-equation counterpart.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0])
    >>> abs(nls_dark_soliton(x, t=0.0, rho0=1.0))
    array([0.])
    """
    envelope = np.sqrt(rho0) * np.tanh(np.sqrt(rho0 / 2) * (x - x0))
    return envelope * np.exp(-1j * rho0 * t)


def nls_evolve(psi0: np.ndarray, x: np.ndarray, dt: float, steps: int, g: float = 1.0) -> np.ndarray:
    """Evolve a nonlinear Schrodinger initial condition via split-step Fourier.

    Solves :math:`i\\psi_t + \\tfrac{1}{2}\\psi_{xx} + g|\\psi|^2\\psi = 0`
    on a periodic domain: ``g > 0`` is focusing (bright solitons),
    ``g < 0`` is defocusing (dark solitons).

    Parameters
    ----------
    psi0 : ndarray of complex
        Initial wavefunction, sampled on the periodic grid ``x``.
    x : ndarray
        Uniformly spaced periodic spatial grid.
    dt : float
        Time step.
    steps : int
        Number of steps to advance.
    g : float, default=1.0
        Nonlinearity strength and sign.

    Returns
    -------
    ndarray of complex
        Field after ``steps * dt`` time units.

    Examples
    --------
    A bright soliton propagates without dispersing -- its envelope
    :math:`|\\psi|` is unchanged after 2 time units of evolution:

    >>> import numpy as np
    >>> N, L = 1024, 80.0
    >>> x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    >>> psi0 = nls_bright_soliton(x, t=0.0, A=1.0)
    >>> psi = nls_evolve(psi0, x, dt=0.001, steps=2000, g=1.0)
    >>> shape_error = np.max(np.abs(np.abs(psi) - np.abs(psi0)))
    >>> bool(shape_error < 1e-3)
    True
    """
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    lin_prop = np.exp(-1j * 0.5 * k**2 * dt)
    psi = np.array(psi0, dtype=complex, copy=True)
    for _ in range(steps):
        psi_hat = np.fft.fft(psi)
        psi_hat *= lin_prop
        psi = np.fft.ifft(psi_hat)
        psi *= np.exp(1j * g * np.abs(psi) ** 2 * dt)
    return psi


def nls_evolve_frames(psi0: np.ndarray, x: np.ndarray, dt: float, steps_per_frame: int, n_frames: int, g: float = 1.0) -> tuple:
    """Evolve an NLS field and record a snapshot every ``steps_per_frame`` steps, for animation.

    A thin looping wrapper around :func:`nls_evolve`, analogous to
    :func:`kdv_evolve_frames`: no new physics, just repeated chunked calls
    with the state carried forward, to produce an animatable sequence of frames.

    Parameters
    ----------
    psi0 : ndarray of complex
        Initial wavefunction.
    x : ndarray
        Uniformly spaced periodic spatial grid.
    dt : float
        Time step.
    steps_per_frame : int
        Number of integration steps between recorded frames.
    n_frames : int
        Number of frames to record after the initial condition.
    g : float, default=1.0
        Nonlinearity strength and sign.

    Returns
    -------
    frames : ndarray of complex, shape (n_frames + 1, len(x))
        Field at ``t=0`` and after each recorded chunk.
    times : ndarray, shape (n_frames + 1,)

    See Also
    --------
    physicskit.fields.visualizers.animate_field_1d : Render these frames as a line-plot animation.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-40, 40, 512, endpoint=False)
    >>> psi0 = nls_bright_soliton(x, t=0.0, A=1.0)
    >>> frames, times = nls_evolve_frames(psi0, x, dt=0.001, steps_per_frame=200, n_frames=5, g=1.0)
    >>> frames.shape
    (6, 512)
    """
    psi = np.array(psi0, dtype=complex, copy=True)
    frames = [psi.copy()]
    times = [0.0]
    for i in range(n_frames):
        psi = nls_evolve(psi, x, dt, steps_per_frame, g)
        frames.append(psi.copy())
        times.append((i + 1) * steps_per_frame * dt)
    return np.array(frames), np.array(times)


# -- Sine-Gordon --------------------------------------------------------------


def sine_gordon_kink(x: np.ndarray, t: float, v: float = 0.0, x0: float = 0.0, polarity: int = 1) -> np.ndarray:
    """Exact kink (or antikink) solution of the Sine-Gordon equation :math:`u_{tt} - u_{xx} + \\sin u = 0`.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    t : float
        Time.
    v : float, default=0.0
        Kink velocity (:math:`|v| < 1`, the wave speed of the linearized equation).
    x0 : float, default=0.0
        Initial center position.
    polarity : {1, -1}, default=1
        ``1`` for a kink (field jumps by :math:`2\\pi`), ``-1`` for an antikink.

    Returns
    -------
    ndarray
        :math:`u(x,t) = 4\\arctan\\!\\big(\\exp[\\text{polarity}\\cdot\\gamma(x-x_0-vt)]\\big)`,
        with :math:`\\gamma = 1/\\sqrt{1-v^2}`.

    See Also
    --------
    sine_gordon_evolve : Propagate this (or any) initial condition numerically.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(sine_gordon_kink(0.0, t=0.0)), 6)
    3.141593
    """
    gamma = 1.0 / np.sqrt(1 - v**2)
    return 4 * np.arctan(np.exp(polarity * gamma * (x - x0 - v * t)))


def sine_gordon_evolve(u0: np.ndarray, u0_prev: np.ndarray, x: np.ndarray, dt: float, steps: int) -> tuple:
    """Evolve the Sine-Gordon equation with explicit leapfrog finite differences.

    Parameters
    ----------
    u0 : ndarray
        Field at the starting time.
    u0_prev : ndarray
        Field one step *before* the starting time (needed to seed the
        two-level leapfrog scheme); for a traveling-wave initial condition,
        use the exact solution evaluated at ``t=-dt``.
    x : ndarray
        Uniformly spaced spatial grid, with fixed (Dirichlet) boundaries.
    dt : float
        Time step; must satisfy the CFL condition ``dt <= dx`` (wave speed 1).
    steps : int
        Number of steps to advance.

    Returns
    -------
    u, u_prev : ndarray
        The field at the final step and the step before it (the pair
        needed to continue the integration further).

    Examples
    --------
    A kink launched at speed :math:`v=0.5` arrives at the expected
    Lorentz-contracted position after propagating:

    >>> import numpy as np
    >>> N, L, v, x0 = 4000, 200.0, 0.5, -50.0
    >>> x = np.linspace(-L / 2, L / 2, N)
    >>> dx = x[1] - x[0]
    >>> dt = 0.4 * dx
    >>> u_prev = sine_gordon_kink(x, -dt, v, x0)
    >>> u0 = sine_gordon_kink(x, 0.0, v, x0)
    >>> steps = 1000
    >>> u, _ = sine_gordon_evolve(u0, u_prev, x, dt, steps)
    >>> expected = sine_gordon_kink(x, steps * dt, v, x0)
    >>> bool(np.max(np.abs(u - expected)) < 0.01)
    True
    """
    dx = x[1] - x[0]
    u_prev = np.array(u0_prev, dtype=float, copy=True)
    u = np.array(u0, dtype=float, copy=True)
    for _ in range(steps):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        u_next = 2 * u - u_prev + dt**2 * (lap - np.sin(u))
        u_next[0] = u[0]
        u_next[-1] = u[-1]
        u_prev, u = u, u_next
    return u, u_prev


def sine_gordon_evolve_frames(u0: np.ndarray, u0_prev: np.ndarray, x: np.ndarray, dt: float, steps_per_frame: int, n_frames: int) -> tuple:
    """Evolve a Sine-Gordon field and record a snapshot every ``steps_per_frame`` steps, for animation.

    A thin looping wrapper around :func:`sine_gordon_evolve`, analogous to
    :func:`kdv_evolve_frames`: the ``(u, u_prev)`` leapfrog pair is carried
    forward chunk to chunk, with no change to the underlying physics.

    Parameters
    ----------
    u0 : ndarray
        Field at the starting time.
    u0_prev : ndarray
        Field one step before the starting time (see :func:`sine_gordon_evolve`).
    x : ndarray
        Uniformly spaced spatial grid.
    dt : float
        Time step.
    steps_per_frame : int
        Number of integration steps between recorded frames.
    n_frames : int
        Number of frames to record after the initial condition.

    Returns
    -------
    frames : ndarray, shape (n_frames + 1, len(x))
        Field at ``t=0`` and after each recorded chunk.
    times : ndarray, shape (n_frames + 1,)

    See Also
    --------
    physicskit.fields.visualizers.animate_field_1d : Render these frames as a line-plot animation.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-50, 50, 800)
    >>> dt = 0.4 * (x[1] - x[0])
    >>> u_prev = sine_gordon_kink(x, -dt, v=0.5, x0=-20)
    >>> u0 = sine_gordon_kink(x, 0.0, v=0.5, x0=-20)
    >>> frames, times = sine_gordon_evolve_frames(u0, u_prev, x, dt, steps_per_frame=100, n_frames=4)
    >>> frames.shape
    (5, 800)
    """
    u = np.array(u0, dtype=float, copy=True)
    u_prev = np.array(u0_prev, dtype=float, copy=True)
    frames = [u.copy()]
    times = [0.0]
    for i in range(n_frames):
        u, u_prev = sine_gordon_evolve(u, u_prev, x, dt, steps_per_frame)
        frames.append(u.copy())
        times.append((i + 1) * steps_per_frame * dt)
    return np.array(frames), np.array(times)
