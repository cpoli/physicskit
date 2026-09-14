"""Small compatibility shims (numpy>=2.0 renamed trapz to trapezoid)."""

import numpy as np

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
