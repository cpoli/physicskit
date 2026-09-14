import numpy as np
import pytest

pv_module = pytest.importorskip("pyvista")

from physicskit.chaos.systems.continuous import Lorenz
from physicskit.chaos.visualizers.viewer3d import pyvista_trajectory


def test_pyvista_trajectory_returns_plotter_with_expected_point_count():
    system = Lorenz()
    _, states = system.trajectory(n_steps=200, dt=0.01)
    plotter = pyvista_trajectory(states, off_screen=True)
    try:
        assert isinstance(plotter, pv_module.Plotter)
        image = plotter.screenshot()
        assert image.ndim == 3 and image.shape[2] == 3
    finally:
        plotter.close()


def test_pyvista_trajectory_color_by_speed_runs_without_error():
    system = Lorenz()
    _, states = system.trajectory(n_steps=200, dt=0.01)
    plotter = pyvista_trajectory(states, color_by="speed", off_screen=True)
    try:
        plotter.screenshot()
    finally:
        plotter.close()


def test_pyvista_trajectory_uses_selected_components():
    states = np.column_stack([np.linspace(0, 1, 20), np.linspace(0, 2, 20), np.linspace(0, 3, 20), np.linspace(0, 4, 20)])
    plotter = pyvista_trajectory(states, components=(1, 2, 3), off_screen=True)
    try:
        mesh = plotter.meshes[0]
        np.testing.assert_allclose(mesh.points[:, 0], states[:, 1])
        np.testing.assert_allclose(mesh.points[:, 1], states[:, 2])
        np.testing.assert_allclose(mesh.points[:, 2], states[:, 3])
    finally:
        plotter.close()
