Kinetic theory (particle-in-cell)
---------------------------------

An electrostatic particle-in-cell (PIC) solution of the 1D1V
Vlasov-Poisson system: particles stream along exact single-particle
orbits and the self-consistent field is recovered by depositing their
charge onto a grid and solving Poisson's equation each step
(:func:`~physicskit.plasma.kinetic.pic_simulate`). The same
deposit-solve-gather-push pipeline reproduces collisionless Landau
damping (:func:`~physicskit.plasma.kinetic.landau_damping_ic`,
:func:`~physicskit.plasma.kinetic.landau_damping_rate`), its
mirror-image growth mechanism, the two-stream instability
(:func:`~physicskit.plasma.kinetic.two_stream_ic`), and a Langmuir wave
ringing in place at the plasma frequency
(:func:`~physicskit.plasma.kinetic.langmuir_wave_ic`), all without ever
assuming a collision operator -- and, since nothing is dissipated,
runs backwards in time to undo Landau damping exactly. The Weibel/filamentation instability
(:func:`~physicskit.plasma.instabilities.weibel_growth_rate`,
:func:`~physicskit.plasma.instabilities.simulate_weibel_filamentation`)
is treated as a reduced quasi-linear model of the same current-driven
physics, at the opposite (temperature-anisotropy) end of the kinetic
spectrum from a beam-driven instability.
