"""Low-level differential geometry and Numba-accelerated integration.

:mod:`physicskit.relativity.core.tensors` computes Christoffel symbols and curvature
tensors numerically for arbitrary metrics. :mod:`physicskit.relativity.core.geodesics`
integrates Schwarzschild and equatorial Kerr geodesics with fixed-step RK4.
:mod:`physicskit.relativity.core.raytracer` backward ray-traces Schwarzschild null
geodesics to render gravitationally lensed images; :mod:`physicskit.relativity.core.kerr_raytracer`
does the same for Kerr, using the full Carter-formalism equations of motion.

These functions operate on plain NumPy arrays and carry no state; the
stateful, user-facing model classes live in :mod:`physicskit.relativity.chapters`.
"""
