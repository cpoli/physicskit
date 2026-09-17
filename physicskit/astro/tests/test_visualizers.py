import numpy as np
import pytest
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection

from physicskit.astro.nbody import NBodySystem, figure_eight_initial_conditions
from physicskit.astro.visualizers import animate_nbody_trajectories


@pytest.mark.slow
def test_animate_nbody_trajectories_figure_eight_saves_gif(tmp_path):
    positions, velocities, masses = figure_eight_initial_conditions()
    system = NBodySystem(positions, velocities, masses)
    history = system.simulate(dt=0.002, n_steps=300)

    anim = animate_nbody_trajectories(history, skip=5)
    out = tmp_path / "nbody_figure_eight.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


def test_animate_nbody_trajectories_uses_one_collection_and_marker_per_body():
    positions, velocities, masses = figure_eight_initial_conditions()
    system = NBodySystem(positions, velocities, masses)
    history = system.simulate(dt=0.002, n_steps=50)

    anim = animate_nbody_trajectories(history)
    ax = anim._fig.axes[0]
    collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(collections) == 3
    assert len(ax.lines) == 3


def test_animate_nbody_trajectories_trail_fades_with_alpha():
    """The trail must actually fade: older segments more transparent than
    the most recent one."""
    positions, velocities, masses = figure_eight_initial_conditions()
    system = NBodySystem(positions, velocities, masses)
    history = system.simulate(dt=0.002, n_steps=100)

    anim = animate_nbody_trajectories(history, trail=100)
    anim._draw_frame(99)
    ax = anim._fig.axes[0]
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    alphas = np.asarray(lc.get_colors())[:, 3]
    assert alphas[0] < alphas[-1]
