r"""Animations for the ten toy "collider physics" demonstrations built on
top of :mod:`physicskit.particle`'s decay, scattering, shower,
confinement, electroweak, and neutrino-oscillation models.

Every function here returns a :class:`matplotlib.animation.FuncAnimation`
(built with ``blit=False`` throughout, since several of these redraw
variable-length or variable-count artists each frame); save it with
e.g. ``anim.save(path, writer=PillowWriter(fps=10))``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from physicskit.particle.collider import charged_track_points, cluster_into_jets, flatten_shower, shower_leaves
from physicskit.particle.decays import michel_spectrum
from physicskit.particle.electroweak import cp_asymmetry, higgs_potential
from physicskit.particle.electroweak import qed_dsigma_domega_mumu as _qed_dsigma_domega_mumu
from physicskit.particle.neutrinos import oscillation_probability
from physicskit.particle.scattering import ALPHA_FS

__all__ = [
    "animate_particle_cascade",
    "animate_parton_shower",
    "animate_detector_event",
    "animate_string_breaking",
    "animate_higgs_rollover",
    "animate_decay_chain_bars",
    "animate_neutrino_oscillation",
    "animate_qed_angular_distribution",
    "animate_michel_histogram",
    "animate_cp_asymmetry",
]


# ---------------------------------------------------------------------------
# Items 1 & 7: shower / parton-shower tree animation (shared machinery)
# ---------------------------------------------------------------------------


def _layout_shower_positions(root, track_length):
    """Map each non-root node to a (start, end) 2D transverse-plane segment,
    walking outward from the collision vertex at the origin."""
    positions = {}

    def walk(node, start):
        for child in node.children:
            px, py = child.four_vector.px, child.four_vector.py
            pt = float(np.hypot(px, py))
            direction = np.array([px, py]) / pt if pt > 1e-9 else np.array([1.0, 0.0])
            end = start + track_length * direction
            positions[id(child)] = (start, end)
            walk(child, end)

    walk(root, np.array([0.0, 0.0]))
    return positions


def _animate_shower_tree(root, detector_radii, track_length, interval, ax, color_func, extra_static=None):
    positions = _layout_shower_positions(root, track_length)
    nodes = [n for n in flatten_shower(root) if n.parent is not None]
    nodes.sort(key=lambda n: n.generation)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    R_track, R_calo = detector_radii
    ax.add_patch(Circle((0, 0), R_track, fill=False, linestyle="--", color="steelblue"))
    ax.add_patch(Circle((0, 0), R_calo, fill=False, linestyle="-", color="firebrick"))
    if extra_static is not None:
        extra_static(ax, R_calo)
    ax.set_xlim(-R_calo * 1.15, R_calo * 1.15)
    ax.set_ylim(-R_calo * 1.15, R_calo * 1.15)
    ax.set_aspect("equal")
    ax.plot(0, 0, "k*", markersize=10)

    lines = [ax.plot([], [], lw=1.6, color=color_func(n))[0] for n in nodes]
    (markers,) = ax.plot([], [], "o", color="black", markersize=3)

    def update(frame):
        visible = nodes[: frame + 1]
        for line, node in zip(lines, nodes):
            if node in visible:
                s, e = positions[id(node)]
                line.set_data([s[0], e[0]], [s[1], e[1]])
        ends = np.array([positions[id(n)][1] for n in visible]) if visible else np.empty((0, 2))
        if len(ends):
            markers.set_data(ends[:, 0], ends[:, 1])
        return [*lines, markers]

    return FuncAnimation(fig, update, frames=max(len(nodes), 1), interval=interval, blit=False)


def animate_particle_cascade(root, detector_radii=(3.0, 6.0), track_length=0.6, interval=150, ax=None):
    """Animate a toy shower cascade (:func:`physicskit.particle.collider.simple_shower`)
    propagating outward from the collision vertex through a schematic detector.

    Concentric circles mark a "tracker" (inner) and "calorimeter" (outer)
    region; each branch of the shower tree appears as one more track
    segment per frame, in order of production (generation).

    Parameters
    ----------
    root : physicskit.particle.collider.ShowerParticle
        Root of the shower tree, e.g. from
        :func:`physicskit.particle.collider.simple_shower`.
    detector_radii : (float, float), default=(3.0, 6.0)
        Radii of the schematic tracker and calorimeter circles.
    track_length : float, default=0.6
        Schematic (not physically derived) drawn length of each track
        segment -- purely a layout choice, not a decay-length calculation.
    interval : int, default=150
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.particle.collider import simple_shower
    >>> root = simple_shower(50.0, E_threshold=5.0, rng=np.random.default_rng(0))
    >>> anim = animate_particle_cascade(root)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    return _animate_shower_tree(
        root,
        detector_radii,
        track_length,
        interval,
        ax,
        color_func=lambda n: plt.cm.plasma(min(n.generation / 6.0, 1.0)),
    )


def animate_parton_shower(root, detector_radii=(3.0, 6.0), track_length=0.6, n_jets=2, interval=150, ax=None):
    """Animate a toy parton shower (:func:`physicskit.particle.collider.parton_shower`)
    developing into jets.

    Reuses the tree-reveal machinery of :func:`animate_particle_cascade`,
    color-coding branches by parton flavor (quark/antiquark vs. gluon)
    and overlaying the final jet-axis directions found by
    :func:`physicskit.particle.collider.cluster_into_jets`.

    Parameters
    ----------
    root : physicskit.particle.collider.ShowerParticle
        Root of the parton shower tree.
    detector_radii, track_length, interval, ax
        See :func:`animate_particle_cascade`.
    n_jets : int, default=2
        Number of jet cones to cluster the final state into and overlay.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.particle.collider import parton_shower
    >>> root = parton_shower(50.0, E_threshold=5.0, rng=np.random.default_rng(0))
    >>> anim = animate_parton_shower(root)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    flavor_colors = {"q": "royalblue", "qbar": "seagreen", "g": "firebrick", "": "gray"}
    leaves = shower_leaves(root)
    jets = cluster_into_jets(leaves, n_jets=n_jets) if len(leaves) >= n_jets else []

    def extra_static(ax, R_calo):
        for jet in jets:
            if not jet:
                continue
            px = np.mean([leaf.four_vector.px for leaf in jet])
            py = np.mean([leaf.four_vector.py for leaf in jet])
            pt = np.hypot(px, py)
            if pt > 1e-9:
                ax.plot([0, R_calo * px / pt], [0, R_calo * py / pt], "--", color="black", alpha=0.4, lw=1.0)

    return _animate_shower_tree(
        root,
        detector_radii,
        track_length,
        interval,
        ax,
        color_func=lambda n: flavor_colors.get(n.flavor, "gray"),
        extra_static=extra_static,
    )


# ---------------------------------------------------------------------------
# Item 10: detector event display
# ---------------------------------------------------------------------------


def animate_detector_event(four_vectors, charges, B, detector_radii=(3.0, 6.0), n_frames=60, interval=100, ax=None):
    r"""Animate final-state tracks growing outward from the vertex and
    landing in a schematic calorimeter.

    Charged particles curve (radius :math:`r=p_T/(qB)`, see
    :func:`physicskit.particle.collider.charged_track_points`); neutral
    particles fly in straight lines. Each frame reveals one more step
    along every track; a marker is drawn at each track's current
    endpoint, standing in for a calorimeter energy deposit once the
    track reaches the outer radius.

    Parameters
    ----------
    four_vectors : sequence of physicskit.particle.kinematics.FourVector
        Final-state particles.
    charges : sequence of float
        Charge of each particle, in units of the elementary charge.
    B : float
        Magnetic field strength (see
        :func:`~physicskit.particle.collider.charged_track_points`).
    detector_radii : (float, float), default=(3.0, 6.0)
        Radii of the schematic tracker and calorimeter circles.
    n_frames : int, default=60
        Number of animation frames (and points sampled along each track).
    interval : int, default=100
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> from physicskit.particle.kinematics import FourVector
    >>> ps = [FourVector(5.0, 3.0, 0.0, 3.0), FourVector(4.0, 0.0, 3.0, 2.0)]
    >>> anim = animate_detector_event(ps, charges=[1.0, -1.0], B=1.0)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    R_track, R_calo = detector_radii
    tracks = [charged_track_points(p, q, B, n_points=n_frames, path_length=R_calo * 1.3) for p, q in zip(four_vectors, charges)]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.add_patch(Circle((0, 0), R_track, fill=False, linestyle="--", color="steelblue"))
    ax.add_patch(Circle((0, 0), R_calo, fill=False, color="firebrick"))
    ax.set_xlim(-R_calo * 1.15, R_calo * 1.15)
    ax.set_ylim(-R_calo * 1.15, R_calo * 1.15)
    ax.set_aspect("equal")
    ax.plot(0, 0, "k*", markersize=10)

    colors = ["crimson" if q > 0 else ("royalblue" if q < 0 else "gray") for q in charges]
    lines = [ax.plot([], [], lw=1.8, color=c)[0] for c in colors]
    (deposits,) = ax.plot([], [], "s", color="darkorange", markersize=7)

    def update(frame):
        ends = []
        for line, track in zip(lines, tracks):
            pts = track[: frame + 2]
            line.set_data(pts[:, 0], pts[:, 1])
            ends.append(pts[-1])
        ends = np.array(ends)
        deposits.set_data(ends[:, 0], ends[:, 1])
        return [*lines, deposits]

    return FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)


# ---------------------------------------------------------------------------
# Item 2: quark confinement / string breaking
# ---------------------------------------------------------------------------


def animate_string_breaking(sim, interval=120, ax=None):
    r"""Animate a receding quark pair's confining string stretching and
    breaking, from :func:`physicskit.particle.confinement.string_break_chain`.

    A thick line (whose width and color track the stored tension,
    ``energy_per_segment``) stretches between the two receding quark
    markers; each time the simulation records a break, it snaps into
    one more, shorter segment, with a new marker appearing at each
    newly created quark-antiquark endpoint. (The endpoints of the
    ``n_segments(t)`` current strings are drawn evenly spaced across the
    current total extent -- a schematic simplification, see
    :func:`~physicskit.particle.confinement.string_break_chain`'s
    docstring, rather than tracking each new pair's exact, individually
    frozen creation position.)

    Parameters
    ----------
    sim : dict
        Output of :func:`physicskit.particle.confinement.string_break_chain`.
    interval : int, default=120
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.particle.confinement import string_break_chain
    >>> sim = string_break_chain(np.linspace(0, 10, 40), v=0.3, kappa=1.0, m_q=1.0, n_breaks=2)
    >>> anim = animate_string_breaking(sim)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    t, r, n_segments, energy_per_segment = sim["t"], sim["r"], sim["n_segments"], sim["energy_per_segment"]
    m_q = sim["m_q"]
    max_segments = int(n_segments.max())

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2.5))
    else:
        fig = ax.figure
    rmax = r.max() * 0.55 + 1e-9
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("separation")
    ax.set_title("QCD string breaking")

    lines = [ax.plot([], [], lw=3, solid_capstyle="round", color="darkorange")[0] for _ in range(max_segments)]
    (endpoints,) = ax.plot([], [], "o", color="black", markersize=7)

    def update(frame):
        n = int(n_segments[frame])
        half = r[frame] / 2.0
        xs = np.linspace(-half, half, n + 1)
        tension = min(energy_per_segment[frame] / (2.0 * m_q), 1.0)
        for i, line in enumerate(lines):
            if i < n:
                line.set_data([xs[i], xs[i + 1]], [0.0, 0.0])
                line.set_color(plt.cm.inferno(0.2 + 0.6 * tension))
                line.set_linewidth(1.0 + 4.0 * tension)
            else:
                line.set_data([], [])
        endpoints.set_data(xs, np.zeros_like(xs))
        return [*lines, endpoints]

    return FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)


# ---------------------------------------------------------------------------
# Item 3: Higgs mechanism (classical rollover)
# ---------------------------------------------------------------------------


def animate_higgs_rollover(phi_t, a, b, interval=30, ax=None):
    r"""Animate a scalar field rolling from the unstable symmetric point
    into a broken-symmetry vacuum, from
    :func:`physicskit.particle.electroweak.higgs_field_rollover`.

    The static curve is :func:`physicskit.particle.electroweak.higgs_potential`;
    a point (with a fading trail) traces :math:`(\phi(t), V(\phi(t)))` as
    the field rolls down one side of the double well.

    Parameters
    ----------
    phi_t : ndarray
        Field trajectory :math:`\phi(t)`, e.g. the first output of
        :func:`~physicskit.particle.electroweak.higgs_field_rollover`.
    a, b : float
        Potential parameters, see
        :func:`~physicskit.particle.electroweak.higgs_potential`.
    interval : int, default=30
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.particle.electroweak import higgs_field_rollover
    >>> t = np.linspace(0, 40, 100)
    >>> phi, _ = higgs_field_rollover(1e-3, 0.0, a=1.0, b=1.0, t_eval=t, damping=0.05)
    >>> anim = animate_higgs_rollover(phi, a=1.0, b=1.0)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    span = 1.3 * np.max(np.abs(phi_t)) + 0.1
    phi_range = np.linspace(-span, span, 400)
    V = higgs_potential(phi_range, a, b)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(phi_range, V, color="gray", lw=1.5)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi)$")
    ax.set_title("Classical symmetry breaking")
    (point,) = ax.plot([], [], "o", color="crimson", markersize=8)
    (trace,) = ax.plot([], [], color="crimson", lw=1.0, alpha=0.5)

    def update(frame):
        phi = phi_t[: frame + 1]
        V_phi = higgs_potential(phi, a, b)
        trace.set_data(phi, V_phi)
        point.set_data([phi[-1]], [V_phi[-1]])
        return [point, trace]

    return FuncAnimation(fig, update, frames=len(phi_t), interval=interval, blit=False)


# ---------------------------------------------------------------------------
# Item 4: decay chain population animation
# ---------------------------------------------------------------------------


def animate_decay_chain_bars(t, populations, labels=None, interval=150, ax=None):
    """Animate each species' population evolving over time in a decay chain.

    Parameters
    ----------
    t : ndarray of shape (n_t,)
        Times.
    populations : ndarray of shape (n_species, n_t)
        Population of each species over time, e.g. from
        :func:`physicskit.particle.decays.bateman_decay_chain`.
    labels : sequence of str, optional
        Bar labels; defaults to ``"species 1"``, ``"species 2"``, ...
    interval : int, default=150
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.particle.decays import bateman_decay_chain
    >>> t = np.linspace(0, 10, 30)
    >>> N = bateman_decay_chain(1000.0, [0.5, 0.2], t)
    >>> anim = animate_decay_chain_bars(t, N)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    populations = np.asarray(populations)
    n_species = populations.shape[0]
    if labels is None:
        labels = [f"species {i + 1}" for i in range(n_species)]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    bars = ax.bar(labels, populations[:, 0], color=plt.cm.viridis(np.linspace(0, 1, n_species)))
    ax.set_ylim(0, populations.max() * 1.05 + 1e-12)
    ax.set_ylabel("population")
    title = ax.set_title(f"t = {t[0]:.2f}")

    def update(frame):
        for bar, h in zip(bars, populations[:, frame]):
            bar.set_height(h)
        title.set_text(f"t = {t[frame]:.2f}")
        return [*bars, title]

    return FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)


# ---------------------------------------------------------------------------
# Item 5: neutrino oscillations
# ---------------------------------------------------------------------------


def animate_neutrino_oscillation(L, E, theta, delta_m2, interval=60, ax=None):
    r"""Animate :math:`P(\nu_e)` and :math:`P(\nu_\mu)` growing as a
    function of distance traveled, from
    :func:`physicskit.particle.neutrinos.oscillation_probability`.

    Parameters
    ----------
    L : ndarray
        Baselines (distance traveled), in km, in increasing order.
    E : float
        Neutrino energy, in GeV.
    theta : float
        Mixing angle, radians.
    delta_m2 : float
        Mass-squared splitting, in :math:`{\rm eV}^2`.
    interval : int, default=60
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> L = np.linspace(0, 1000, 100)
    >>> anim = animate_neutrino_oscillation(L, E=1.0, theta=0.6, delta_m2=2.5e-3)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    L = np.asarray(L, dtype=float)
    P_mu = oscillation_probability(L, E, theta, delta_m2)
    P_e = 1.0 - P_mu

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.set_xlim(L.min(), L.max())
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("L [km]")
    ax.set_ylabel("probability")
    (line_e,) = ax.plot([], [], color="steelblue", label=r"$P(\nu_e)$")
    (line_mu,) = ax.plot([], [], color="firebrick", label=r"$P(\nu_\mu)$")
    ax.legend(loc="upper right")

    def update(frame):
        line_e.set_data(L[: frame + 1], P_e[: frame + 1])
        line_mu.set_data(L[: frame + 1], P_mu[: frame + 1])
        return [line_e, line_mu]

    return FuncAnimation(fig, update, frames=len(L), interval=interval, blit=False)


# ---------------------------------------------------------------------------
# Item 6: QED angular distribution vs. sqrt(s)
# ---------------------------------------------------------------------------


def animate_qed_angular_distribution(sqrt_s_values, alpha=ALPHA_FS, n_theta=200, interval=200, ax=None):
    r"""Animate the :math:`e^+e^-\to\mu^+\mu^-` angular distribution's
    shape as :math:`\sqrt s` is swept, from
    :func:`physicskit.particle.electroweak.qed_dsigma_domega_mumu`.

    Parameters
    ----------
    sqrt_s_values : ndarray
        Center-of-mass energies to sweep through (one per frame).
    alpha : float, default=ALPHA_FS
        Fine-structure constant.
    n_theta : int, default=200
        Number of polar-angle samples per frame.
    interval : int, default=200
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Polar axes to draw into; new polar axes are created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> sqrt_s = np.linspace(5.0, 20.0, 20)
    >>> anim = animate_qed_angular_distribution(sqrt_s)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
    else:
        fig = ax.figure
    (line,) = ax.plot([], [])
    title = ax.set_title("")

    def update(frame):
        s = sqrt_s_values[frame]
        vals = _qed_dsigma_domega_mumu(np.cos(theta), s, alpha=alpha)
        line.set_data(theta, vals)
        ax.set_ylim(0.0, max(float(vals.max()) * 1.1, 1e-30))
        title.set_text(rf"$\sqrt{{s}}={s:.2f}$")
        return [line, title]

    return FuncAnimation(fig, update, frames=len(sqrt_s_values), interval=interval, blit=False)


# ---------------------------------------------------------------------------
# Item 8: muon decay / Michel spectrum
# ---------------------------------------------------------------------------


def animate_michel_histogram(x_samples, batch_size=20, n_bins=30, interval=100, ax=None):
    r"""Animate a histogram of sampled Michel-spectrum electron energies
    building up event-by-event, converging to
    :func:`physicskit.particle.decays.michel_spectrum`.

    Parameters
    ----------
    x_samples : ndarray
        Sampled scaled electron energies, e.g. from
        :func:`physicskit.particle.decays.sample_michel_electron_energies`.
    batch_size : int, default=20
        Number of additional samples revealed per frame.
    n_bins : int, default=30
        Number of histogram bins.
    interval : int, default=100
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.particle.decays import sample_michel_electron_energies
    >>> xs = sample_michel_electron_energies(500, rng=np.random.default_rng(0))
    >>> anim = animate_michel_histogram(xs, batch_size=100)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    x_theory = np.linspace(0.0, 1.0, 200)
    y_theory = michel_spectrum(x_theory)
    n_frames = int(np.ceil(len(x_samples) / batch_size))

    def update(frame):
        ax.cla()
        ax.plot(x_theory, y_theory, color="black", lw=2, label="Michel spectrum")
        n = min((frame + 1) * batch_size, len(x_samples))
        ax.hist(x_samples[:n], bins=n_bins, range=(0.0, 1.0), density=True, color="steelblue", alpha=0.7, label=f"N={n}")
        ax.set_xlabel(r"$x=2E_e/m_\mu$")
        ax.set_ylabel(r"$d\Gamma/dx$")
        ax.set_ylim(0.0, max(2.2, y_theory.max() * 1.2))
        ax.legend(loc="upper left")
        return []

    return FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)


# ---------------------------------------------------------------------------
# Item 9: CP violation
# ---------------------------------------------------------------------------


def animate_cp_asymmetry(t, delta_m, gamma_s, gamma_l, epsilon, interval=60, ax=None):
    r"""Animate the neutral-meson decay-rate CP asymmetry :math:`A(t)`
    building up, from :func:`physicskit.particle.electroweak.cp_asymmetry`.

    Parameters
    ----------
    t : ndarray
        Times.
    delta_m, gamma_s, gamma_l, epsilon
        See :func:`physicskit.particle.electroweak.meson_decay_rates_cp_eigenstate`.
    interval : int, default=60
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 20, 100)
    >>> anim = animate_cp_asymmetry(t, delta_m=0.5, gamma_s=1.0, gamma_l=0.1, epsilon=0.002 + 0.001j)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    t = np.asarray(t, dtype=float)
    A = cp_asymmetry(t, delta_m, gamma_s, gamma_l, epsilon)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    amax = float(np.max(np.abs(A))) + 1e-12
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(-1.15 * amax, 1.15 * amax)
    ax.set_xlabel("t")
    ax.set_ylabel("A(t)")
    ax.set_title("CP-violating decay-rate asymmetry")
    (line,) = ax.plot([], [], color="darkorange", lw=1.6)

    def update(frame):
        line.set_data(t[: frame + 1], A[: frame + 1])
        return [line]

    return FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)
