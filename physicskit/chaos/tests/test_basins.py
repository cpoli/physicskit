import numpy as np

from physicskit.chaos.systems.continuous import MagneticPendulum, magnetic_pendulum_rhs
from physicskit.chaos.visualizers.basins import basin_of_attraction


def test_basin_of_attraction_output_shapes():
    system = MagneticPendulum()
    xs, ys, labels = basin_of_attraction(
        magnetic_pendulum_rhs,
        system.params,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        attractors=system.magnet_positions,
        resolution=20,
        dt=0.05,
        n_steps=500,
    )
    assert xs.shape == (20,)
    assert ys.shape == (20,)
    assert labels.shape == (20, 20)
    assert labels.dtype == np.int64


def test_basin_of_attraction_labels_are_valid_attractor_indices():
    system = MagneticPendulum()
    _, _, labels = basin_of_attraction(
        magnetic_pendulum_rhs,
        system.params,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        attractors=system.magnet_positions,
        resolution=20,
        dt=0.05,
        n_steps=500,
    )
    assert np.all(labels >= 0)
    assert np.all(labels < system.magnet_positions.shape[0])


def test_basin_of_attraction_finds_every_attractor_with_symmetric_magnets():
    """With 3 symmetrically-placed magnets and a grid centered on them, every
    magnet should win somewhere on the grid."""
    system = MagneticPendulum()
    _, _, labels = basin_of_attraction(
        magnetic_pendulum_rhs,
        system.params,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        attractors=system.magnet_positions,
        resolution=60,
        dt=0.05,
        n_steps=500,
    )
    assert set(np.unique(labels).tolist()) == {0, 1, 2}


def test_basin_of_attraction_point_at_a_magnet_is_labeled_that_magnet():
    """A grid point starting exactly on top of a magnet (with zero velocity)
    should be classified to that same magnet."""
    system = MagneticPendulum()
    magnet_idx = 0
    magnet = system.magnet_positions[magnet_idx]
    # A tiny grid straddling the magnet position.
    _xs, _ys, labels = basin_of_attraction(
        magnetic_pendulum_rhs,
        system.params,
        x_range=(magnet[0] - 0.01, magnet[0] + 0.01),
        y_range=(magnet[1] - 0.01, magnet[1] + 0.01),
        attractors=system.magnet_positions,
        resolution=3,
        dt=0.05,
        n_steps=500,
    )
    center_label = labels[1, 1]
    assert center_label == magnet_idx
