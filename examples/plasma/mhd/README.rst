Magnetohydrodynamics
--------------------

Treats the plasma as a single conducting fluid rather than a collection of
orbiting particles: the characteristic Alfven and magnetosonic wave speeds
(:func:`~physicskit.plasma.mhd.alfven_speed`,
:func:`~physicskit.plasma.mhd.magnetosonic_speeds`); static toroidal
equilibrium via the Grad-Shafranov equation
(:func:`~physicskit.plasma.mhd.solve_grad_shafranov`,
:func:`~physicskit.plasma.mhd.safety_factor_large_aspect_ratio`); and
resistive magnetic reconnection, comparing the classic Sweet-Parker model
against Petschek's faster X-point revision
(:func:`~physicskit.plasma.mhd.sweet_parker_rate`,
:func:`~physicskit.plasma.mhd.petschek_rate`).
