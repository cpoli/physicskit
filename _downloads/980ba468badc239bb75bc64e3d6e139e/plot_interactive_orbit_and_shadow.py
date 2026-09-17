r"""
Interactive Plotly visualizations
======================================

Every other visualizer in physicskit is Matplotlib-based: static, ideal for
publication figures and this documentation's gallery.
:mod:`physicskit.relativity.visualizers.interactive` offers the same
information as pan/zoom/rotate-enabled Plotly figures instead, convenient
for interactively exploring a 3D orbit or a ray-traced shadow image in a
Jupyter notebook. This example builds two such figures: a precessing,
eccentric timelike geodesic around a Schwarzschild black hole of mass
:math:`M`, obtained by integrating the geodesic equation

.. math::

    \frac{d^2 x^\mu}{d\tau^2} + \Gamma^\mu_{\alpha\beta}
        \frac{dx^\alpha}{d\tau}\frac{dx^\beta}{d\tau} = 0

and a backward ray-traced silhouette (shadow) of a rotating Kerr black hole
with spin parameter :math:`a = 0.9M`, then saves each as a standalone
HTML file.
"""

from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole
from physicskit.relativity.visualizers.interactive import interactive_orbit_3d, interactive_shadow_image
from physicskit.relativity.visualizers.shadow_render import render_black_hole_image

# %%
# An interactive 3D precessing orbit
# ------------------------------------
# A bound, eccentric equatorial orbit starting at apoapsis :math:`r_0=15M`
# with tangential velocity reduced 20% below circular, integrated exactly
# (no post-Newtonian approximation) so its relativistic apsidal precession
# is directly visible in the resulting rosette.
bh = SchwarzschildBlackHole(M=1.0)
y0 = bh.eccentric_orbit_initial_state(r0=15.0, eccentricity_boost=0.2)
trajectory = bh.integrate_geodesic(y0, dtau=0.02, n_steps=8000)
fig = interactive_orbit_3d(trajectory, M=1.0)
fig.write_html("orbit_interactive.html")

# %%
# An interactive ray-traced shadow image
# ------------------------------------------
# Backward ray-tracing a camera image around a near-extremal Kerr black
# hole (:math:`a=0.9M`) using the Carter-separated null geodesic equations
# produces the shadow and lensed accretion disk seen below, now pannable
# and zoomable.
result = render_black_hole_image(M=1.0, a=0.9, ny=120, nx=120)
fig = interactive_shadow_image(result)
fig.write_html("shadow_interactive.html")

print("Wrote 2 standalone interactive HTML files.")
