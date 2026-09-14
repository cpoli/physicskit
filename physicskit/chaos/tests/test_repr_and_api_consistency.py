import pytest

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
