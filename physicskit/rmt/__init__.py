"""physicskit.rmt -- a random matrix theory library organized around Dyson's
threefold way, validated against exact theory and seminal-paper results."""

from . import ensembles, stats, validation
from .spectrum import Spectrum

__all__ = ["ensembles", "stats", "validation", "Spectrum"]
__version__ = "0.1.0"
