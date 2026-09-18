Single-particle motion
----------------------

Charged-particle motion in electromagnetic fields, in two complementary
pictures: the exact Lorentz-force orbit integrated with the
energy-conserving Boris pusher
(:func:`~physicskit.plasma.single_particle.boris_integrate`), and the
guiding-center picture that averages over the fast gyration to track only
the slow drift of the orbit's center
(:func:`~physicskit.plasma.single_particle.exb_drift`,
:func:`~physicskit.plasma.single_particle.grad_b_drift`,
:func:`~physicskit.plasma.single_particle.curvature_drift`) and the
adiabatic invariant responsible for magnetic mirror confinement
(:func:`~physicskit.plasma.single_particle.magnetic_moment`,
:func:`~physicskit.plasma.single_particle.magnetic_mirror_bounce`).
