import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection

from physicskit.chaos.systems.continuous import DrivenPendulum, Lorenz, MagneticPendulum, RestrictedThreeBody
from physicskit.chaos.systems.maps import BakersMap
from physicskit.chaos.visualizers.dynamic_plots import (
    animate_bakers_map,
    animate_driven_pendulum,
    animate_multi_orbit_map,
    animate_phase_volume_contraction,
    animate_restricted_three_body,
    plot_colored_trajectory,
    plotly_3d_trajectory,
)


def test_plot_colored_trajectory_returns_figure_and_axes_with_line_collection():
    x = np.linspace(0.0, 1.0, 50)
    y = np.sin(x * 10.0)
    values = np.arange(50, dtype=np.float64)

    _fig, ax = plot_colored_trajectory(x, y, values)
    collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(collections) == 1
    # One fewer segment than points (each segment joins consecutive points).
    assert collections[0].get_segments().__len__() == 49


def test_plot_colored_trajectory_sets_axis_limits_to_data_range():
    x = np.array([0.0, 5.0, 2.0])
    y = np.array([0.0, 1.0, -3.0])
    values = np.array([0.0, 1.0, 2.0])

    _, ax = plot_colored_trajectory(x, y, values)
    assert ax.get_xlim() == (0.0, 5.0)
    assert ax.get_ylim() == (-3.0, 1.0)


def test_plot_colored_trajectory_no_colorbar_when_disabled():
    x = np.linspace(0.0, 1.0, 10)
    y = np.linspace(0.0, 1.0, 10)
    values = np.linspace(0.0, 1.0, 10)

    fig, _ = plot_colored_trajectory(x, y, values, colorbar=False)
    assert len(fig.axes) == 1  # just the main axes, no colorbar axes added


def test_plotly_3d_trajectory_contains_all_points():
    states = np.column_stack([np.linspace(0, 1, 20), np.linspace(0, 2, 20), np.linspace(0, 3, 20)])
    fig = plotly_3d_trajectory(states)
    trace = fig.data[0]
    assert len(trace.x) == 20
    np.testing.assert_allclose(trace.z, states[:, 2])


def test_plotly_3d_trajectory_color_by_sets_marker_color_array():
    states = np.random.default_rng(0).uniform(size=(15, 3))
    color_values = np.arange(15, dtype=np.float64)
    fig = plotly_3d_trajectory(states, color_by=color_values)
    trace = fig.data[0]
    np.testing.assert_allclose(trace.line.color, color_values)


def test_animate_phase_volume_contraction_rejects_non_3d_system():
    with pytest.raises(ValueError):
        animate_phase_volume_contraction(MagneticPendulum())


def test_animate_phase_volume_contraction_shrinks_the_cloud():
    """The whole point of the animation: the point cloud's volume must shrink
    over time for a dissipative system, since that is what phase-space
    volume contraction means geometrically. Uses a small ball and a short
    horizon, where the linear (Jacobian-trace) contraction that
    :func:`~physicskit.chaos.utils.metrics.phase_volume_expansion` predicts still
    dominates -- over a long horizon the cloud instead stretches into a
    thin sheet *along* the attractor, and simple size statistics (spread
    along one axis, a covariance-based volume estimate) stop decreasing
    monotonically even though it is still, in the measure-theoretic sense,
    contracting."""

    def _cloud_volume(anim, frame):
        anim._draw_frame(frame)
        x, y, z = anim._fig.axes[0].collections[0]._offsets3d
        cov = np.cov(np.column_stack([x, y, z]).T)
        return float(np.sqrt(np.linalg.det(cov)))

    system = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    anim = animate_phase_volume_contraction(
        system,
        n_points=200,
        ball_radius=0.3,
        t_max=0.5,
        dt=0.005,
        n_frames=10,
        warmup_steps=200,
        seed=0,
    )
    volumes = [_cloud_volume(anim, i) for i in range(10)]
    assert all(v2 < v1 for v1, v2 in zip(volumes, volumes[1:]))


def test_animate_phase_volume_contraction_respects_explicit_center_state():
    system = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    center = np.array([1.0, 1.0, 1.0])
    anim = animate_phase_volume_contraction(system, center_state=center, ball_radius=0.5, n_points=50, t_max=1.0, n_frames=5)
    anim._draw_frame(0)
    x, y, z = anim._fig.axes[0].collections[0]._offsets3d
    points0 = np.column_stack([x, y, z])
    assert np.allclose(points0.mean(axis=0), center, atol=0.5)


def _n_points_drawn(anim):
    return sum(len(line.get_xdata()) for line in anim._fig.axes[0].lines)


def test_animate_multi_orbit_map_grows_every_orbit_together_not_one_at_a_time():
    """Every orbit must gain points on (almost) every frame -- not have whole
    orbits complete one at a time -- so the picture keeps visibly changing
    for the whole animation instead of mostly settling within a few frames."""
    rng = np.random.default_rng(0)
    orbits = [rng.uniform(size=(50, 2)) for _ in range(5)]
    anim = animate_multi_orbit_map(orbits, n_frames=20)

    anim._draw_frame(0)
    counts = [_n_points_drawn(anim)]
    for i in range(1, 20):
        anim._draw_frame(i)
        counts.append(_n_points_drawn(anim))

    # Points accumulate monotonically and every orbit has contributed to the
    # very first frame already (all 5 lines non-empty), not just one.
    assert all(c2 >= c1 for c1, c2 in zip(counts, counts[1:]))
    anim._draw_frame(0)
    non_empty_lines_at_frame_0 = sum(1 for line in anim._fig.axes[0].lines if len(line.get_xdata()) > 0)
    assert non_empty_lines_at_frame_0 == 5


def test_animate_multi_orbit_map_reaches_full_length_on_last_frame():
    orbits = [np.column_stack([np.arange(30), np.arange(30)]) for _ in range(3)]
    anim = animate_multi_orbit_map(orbits, n_frames=10)
    anim._draw_frame(9)
    assert _n_points_drawn(anim) == 30 * 3


def test_animate_multi_orbit_map_handles_orbits_of_different_lengths():
    orbits = [np.zeros((10, 2)), np.zeros((30, 2))]
    anim = animate_multi_orbit_map(orbits, n_frames=5)
    anim._draw_frame(4)
    lengths = [len(line.get_xdata()) for line in anim._fig.axes[0].lines]
    assert lengths == [10, 30]


def _bakers_scatter(anim):
    return anim._fig.axes[0].collections[0]


def test_animate_bakers_map_colors_left_and_right_half_differently():
    n_points = 100
    seed = 0
    system = BakersMap(alpha=0.5)
    anim = animate_bakers_map(
        system,
        n_points=n_points,
        n_iterations=1,
        frames_per_iteration=6,
        stripe_colors=("tab:blue", "tab:red"),
        seed=seed,
    )

    # Reconstruct the same jittered starting grid `animate_bakers_map` builds
    # internally, to know each marker's *original* x -- by frame 0 the
    # stretch phase has already moved the markers, so their current offsets
    # no longer reflect the left/right split the coloring is based on.
    n_side = round(np.sqrt(n_points))
    centers = (np.arange(n_side) + 0.5) / n_side
    grid_x, _ = np.meshgrid(centers, centers)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.15, 0.15, size=(2,) + grid_x.shape) / n_side
    x0 = (grid_x + jitter[0]).ravel()

    anim._draw_frame(0)
    facecolors = _bakers_scatter(anim).get_facecolor()
    unique_colors = {tuple(c) for c in facecolors}
    assert len(unique_colors) == 2
    # Color must actually track the left/right split, not just happen to
    # take two values: every left-of-center point shares one color and every
    # right-of-center point shares the other.
    left_colors = {tuple(c) for c, x in zip(facecolors, x0) if x < 0.5}
    right_colors = {tuple(c) for c, x in zip(facecolors, x0) if x >= 0.5}
    assert len(left_colors) == 1
    assert len(right_colors) == 1
    assert left_colors != right_colors


def test_animate_bakers_map_stretch_phase_extends_past_unit_square():
    """During the stretch phase the cloud should spill out past x = 1, since
    the right-hand branch is drawn stretched into its own, offset slot
    rather than snapping straight to its final (folded-back) position."""
    system = BakersMap(alpha=0.5)
    anim = animate_bakers_map(system, n_points=400, n_iterations=1, frames_per_iteration=9)
    for i in range(4):  # sequentially through frame 3 (mid-stretch, before the cut/pause frame)
        anim._draw_frame(i)
    x = _bakers_scatter(anim).get_offsets()[:, 0]
    assert x.max() > 1.0


def test_animate_bakers_map_returns_to_unit_square_after_full_iteration():
    """At the last frame of an iteration, every point must land exactly on
    the analytic baker's map image of its starting point, back inside the
    unit square (no residual stretch/offset left over)."""
    alpha = 0.5
    system = BakersMap(alpha=alpha)
    n_points = 100
    seed = 0
    anim = animate_bakers_map(system, n_points=n_points, n_iterations=1, frames_per_iteration=6, seed=seed)

    # Reconstruct the same jittered starting grid `animate_bakers_map` builds
    # internally (a perfectly regular grid sits at exact dyadic rationals,
    # which this map's doubling action on `x` would otherwise degenerate
    # within a handful of iterations).
    n_side = round(np.sqrt(n_points))
    centers = (np.arange(n_side) + 0.5) / n_side
    grid_x, grid_y = np.meshgrid(centers, centers)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.15, 0.15, size=(2,) + grid_x.shape) / n_side
    x0 = np.clip(grid_x + jitter[0], 0.0, 1.0 - 1e-12).ravel()
    y0 = np.clip(grid_y + jitter[1], 0.0, 1.0 - 1e-12).ravel()
    left = x0 < alpha
    x_expected = np.where(left, x0 / alpha, (x0 - alpha) / (1.0 - alpha))
    y_expected = np.where(left, alpha * y0, alpha + (1.0 - alpha) * y0)

    for i in range(6):  # sequentially through the last frame of the (only) iteration
        anim._draw_frame(i)
    offsets = _bakers_scatter(anim).get_offsets()
    np.testing.assert_allclose(np.sort(offsets[:, 0]), np.sort(x_expected), atol=1e-10)
    np.testing.assert_allclose(np.sort(offsets[:, 1]), np.sort(y_expected), atol=1e-10)


def test_animate_bakers_map_uses_square_markers():
    system = BakersMap(alpha=0.5)
    anim = animate_bakers_map(system, n_points=100, n_iterations=1, frames_per_iteration=6)
    paths = _bakers_scatter(anim).get_paths()
    assert len(paths) == 1  # scatter shares one path (the marker shape) across all points
    # A square marker path has 4 (or 5, if closed) vertices.
    assert len(paths[0].vertices) in (4, 5)


@pytest.mark.slow
def test_animate_restricted_three_body_saves_gif(tmp_path):
    system = RestrictedThreeBody()
    anim = animate_restricted_three_body(system, dt=0.001, n_steps=800, skip=20)
    out = tmp_path / "cr3bp.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.slow
def test_animate_driven_pendulum_saves_gif(tmp_path):
    system = DrivenPendulum()
    anim = animate_driven_pendulum(system, dt=0.02, n_steps=60)
    out = tmp_path / "driven_pendulum.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0
