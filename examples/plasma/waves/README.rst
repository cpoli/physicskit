Cold-plasma waves
-----------------

Linearizes the multi-fluid cold-plasma equations around a uniform
background to get the plasma frequency
(:func:`~physicskit.plasma.waves.plasma_frequency`) and the Stix
dielectric tensor components :math:`S`, :math:`D`, :math:`P`
(:func:`~physicskit.plasma.waves.stix_parameters`), from which every
named cold-plasma wave mode follows as one quartic equation in the
refractive index (:func:`~physicskit.plasma.waves.cold_plasma_dispersion`),
organized by the Clemmow-Mullaly-Allis (CMA) diagram
(:func:`~physicskit.plasma.waves.cma_coordinates`). Beyond this linear,
dispersion-relation picture, the ion-acoustic soliton
(:func:`~physicskit.plasma.waves.ion_acoustic_soliton_profile`,
:func:`~physicskit.plasma.waves.ion_acoustic_soliton_evolve`) is a
genuinely nonlinear traveling wave, propagating without change of shape
under the Washimi-Taniuti Korteweg-de Vries reduction.
