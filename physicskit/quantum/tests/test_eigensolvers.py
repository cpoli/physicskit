"""Error-path tests for NumerovSolver that test_physics_checks.py's
accuracy checks never exercise: rejecting a non-uniform grid and
rejecting a requested state count that isn't smaller than the number
of interior grid points."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum.core.eigensolvers import NumerovSolver, infinite_well


def test_numerov_solver_rejects_non_uniform_grid():
    x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])  # unevenly spaced
    with pytest.raises(ValueError):
        NumerovSolver(x, infinite_well())


def test_numerov_solver_rejects_n_states_not_smaller_than_interior_points():
    x = np.linspace(0, 1, 6)  # 4 interior points
    solver = NumerovSolver(x, infinite_well())
    with pytest.raises(ValueError):
        solver.solve(n_states=4)
