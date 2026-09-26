"""physicskit.chaos: visual analysis and simulation of chaotic dynamical systems and 2D billiards."""

from physicskit.chaos.core.base_system import BilliardSystem, DiscreteMap, DynamicalSystem
from physicskit.chaos.exceptions import ChaoskitError, InvalidParameterError
from physicskit.chaos.quantum.billiards import QuantumBilliard
from physicskit.chaos.quantum.maps import QuantumBakersMap, QuantumKickedRotor
from physicskit.chaos.systems.billiards import (
    BunimovichStadium,
    CircleBilliard,
    EllipseBilliard,
    RectangleBilliard,
    SinaiBilliard,
    TruncatedCircleBilliard,
)
from physicskit.chaos.systems.continuous import (
    Chua,
    DoublePendulum,
    Duffing,
    ForcedVanDerPol,
    Lorenz,
    MagneticPendulum,
    RestrictedThreeBody,
    Rossler,
)
from physicskit.chaos.systems.maps import BakersMap, HenonMap, LogisticMap, SmaleHorseshoe, StandardMap

__version__ = "0.1.0"

__all__ = [
    "BakersMap",
    "BilliardSystem",
    "BunimovichStadium",
    "ChaoskitError",
    "Chua",
    "CircleBilliard",
    "DiscreteMap",
    "DoublePendulum",
    "Duffing",
    "DynamicalSystem",
    "EllipseBilliard",
    "ForcedVanDerPol",
    "HenonMap",
    "InvalidParameterError",
    "LogisticMap",
    "Lorenz",
    "MagneticPendulum",
    "QuantumBakersMap",
    "QuantumBilliard",
    "QuantumKickedRotor",
    "RectangleBilliard",
    "RestrictedThreeBody",
    "Rossler",
    "SinaiBilliard",
    "SmaleHorseshoe",
    "StandardMap",
    "TruncatedCircleBilliard",
    "__version__",
]
