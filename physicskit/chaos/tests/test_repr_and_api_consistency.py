import numpy as np
import pytest

from physicskit.chaos.core.base_system import DiscreteMap, DynamicalSystem, get_init_params
from physicskit.chaos.exceptions import ChaoskitError, InvalidParameterError
from physicskit.chaos.quantum.billiards import QuantumBilliard
from physicskit.chaos.quantum.maps import QuantumBakersMap, QuantumKickedRotor
from physicskit.chaos.systems.billiards import CircleBilliard, EllipseBilliard, SinaiBilliard
from physicskit.chaos.systems.continuous import Lorenz, MagneticPendulum
from physicskit.chaos.systems.maps import BakersMap, HenonMap, LogisticMap, StandardMap


@pytest.mark.parametrize(
    "obj, expected_substrings",
    [
        (Lorenz(sigma=10.0, rho=28.0), ["Lorenz(", "sigma=10.0", "rho=28.0"]),
        (HenonMap(a=1.4, b=0.3), ["HenonMap(", "a=1.4", "b=0.3"]),
        (SinaiBilliard(cell_size=2.0), ["SinaiBilliard(", "cell_size=2.0"]),
        (MagneticPendulum(friction=0.3), ["MagneticPendulum(", "friction=0.3"]),
        (QuantumKickedRotor(k=2.0, dim=32), ["QuantumKickedRotor(", "k=2.0", "dim=32"]),
        (QuantumBakersMap(dim=40, alpha=0.5), ["QuantumBakersMap(", "dim=40", "alpha=0.5"]),
        (
            QuantumBilliard(CircleBilliard(radius=1.0), resolution=50),
            ["QuantumBilliard(", "resolution=50"],
        ),
    ],
)
def test_repr_includes_class_name_and_init_params(obj, expected_substrings):
    text = repr(obj)
    for substring in expected_substrings:
        assert substring in text


def test_repr_round_trips_through_eval_for_simple_scalar_params():
    """For classes whose __init__ params are all simple scalars, repr()
    should produce a string that reconstructs an equivalent object."""
    original = HenonMap(a=1.4, b=0.3)
    reconstructed = eval(repr(original), {"HenonMap": HenonMap})
    assert reconstructed.a == original.a
    assert reconstructed.b == original.b


@pytest.mark.parametrize(
    "make_map",
    [lambda: StandardMap(), lambda: HenonMap(), lambda: BakersMap(), lambda: LogisticMap()],
)
def test_discrete_map_trajectory_defaults_state0_like_dynamical_system(make_map):
    """DiscreteMap.trajectory() should default state0 via initial_state(),
    exactly like DynamicalSystem.trajectory() already does -- not require it
    positionally."""
    system = make_map()
    traj = system.trajectory(n_iter=5)
    assert traj.shape == (6, system.dim)


def test_invalid_parameter_error_is_both_chaos_and_value_error():
    with pytest.raises(ChaoskitError):
        EllipseBilliard(semi_major=1.0, semi_minor=1.5)
    with pytest.raises(ValueError):
        EllipseBilliard(semi_major=1.0, semi_minor=1.5)
    with pytest.raises(InvalidParameterError):
        EllipseBilliard(semi_major=1.0, semi_minor=1.5)


class _MinimalDynamicalSystem(DynamicalSystem):
    """The smallest possible concrete subclass: overrides only the
    abstract ``rhs``, so calling ``initial_state()`` exercises
    DynamicalSystem's own default (every real system in this package
    overrides it instead, per module docstring)."""

    dim = 2

    def rhs(self, state, t):
        return -state


class _MinimalDiscreteMap(DiscreteMap):
    """The smallest possible concrete subclass: overrides only the
    abstract ``step``, so calling ``initial_state()``/``trajectory()``
    exercises DiscreteMap's own defaults (every real map in this package
    overrides both, per module docstring)."""

    dim = 1

    def step(self, state):
        return state + 1.0


def test_dynamical_system_initial_state_default_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        _MinimalDynamicalSystem().initial_state()


def test_discrete_map_initial_state_default_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        _MinimalDiscreteMap().initial_state()


def test_discrete_map_trajectory_default_implementation_iterates_step():
    system = _MinimalDiscreteMap()
    traj = system.trajectory(state0=np.array([0.0]), n_iter=3)
    np.testing.assert_allclose(traj, [[0.0], [1.0], [2.0], [3.0]])


def test_get_init_params_skips_unreadable_signature_and_missing_attributes():
    """get_init_params must degrade gracefully both when the constructor's
    signature can't be introspected (returns {}) and when a constructor
    parameter isn't stored under its own name on the instance (that one
    param is skipped, not raised)."""

    class Skippy:
        def __init__(self, a, b=1):
            self.a = a
            # b intentionally not stored as self.b

    obj = Skippy(a=1, b=2)
    assert get_init_params(obj) == {"a": 1}

    class Unreadable:
        def __init__(self, a=1):
            self.a = a

    unreadable_obj = Unreadable(a=5)
    Unreadable.__init__ = 5  # no longer a real callable/function
    assert get_init_params(unreadable_obj) == {}


def test_billiard_boundary_arrays_matches_build_boundary_output():
    billiard = CircleBilliard(radius=2.0)
    segments, arcs, arc_full = billiard.boundary_arrays()
    assert segments.shape == (0, 4)
    assert arcs.shape == (1, 5)
    assert arc_full.tolist() == [True]
    np.testing.assert_allclose(arcs[0][:3], [0.0, 0.0, 2.0])
