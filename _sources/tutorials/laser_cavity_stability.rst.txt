:orphan:

Laser Cavity Stability and the Fundamental Gaussian Mode
============================================================

This tutorial builds a two-mirror optical resonator with
:mod:`physicskit.optics.ray`, checks its stability against the standard
:math:`g`-parameter criterion, and then finds its self-consistent
fundamental Gaussian mode with :mod:`physicskit.optics.gaussian`.

Building the cavity round trip
----------------------------------

A resonator formed by two spherical mirrors of radius :math:`R_1, R_2`
separated by length :math:`L` is described, one round trip at a time, by
cascading a reflection off each mirror with a free-space propagation
between them. :func:`~physicskit.optics.ray.cavity_round_trip_matrix`
multiplies the element list in order (rightmost element applied first),
so listing ``[M1, d1, M2, d2]`` traces a ray starting just after mirror 1,
to mirror 2, and back to mirror 1:

.. code-block:: python

   import numpy as np
   from physicskit.optics.ray import (
       OpticalElement, spherical_mirror, free_space,
       cavity_round_trip_matrix, cavity_stability,
   )

   R1, R2, L = 2.0, 2.0, 1.0

   elements = [
       OpticalElement(spherical_mirror(R1), name="M1"),
       OpticalElement(free_space(L), name="d1", length=L),
       OpticalElement(spherical_mirror(R2), name="M2"),
       OpticalElement(free_space(L), name="d2", length=L),
   ]
   M = cavity_round_trip_matrix(elements)
   print(M)
   # [[-1.  1.]
   #  [-1.  0.]]

The stability criterion
----------------------------

A resonator is stable -- rays stay bounded after arbitrarily many round
trips -- exactly when :math:`|A+D| \le 2`, checked directly by
:func:`~physicskit.optics.ray.cavity_stability`:

.. code-block:: python

   print(cavity_stability(M))  # True

This is the same condition usually written in terms of the dimensionless
:math:`g`-parameters :math:`g_i = 1 - L/R_i`, stable when
:math:`0 \le g_1 g_2 \le 1`:

.. code-block:: python

   g1, g2 = 1 - L / R1, 1 - L / R2
   print(g1 * g2)  # 0.25 -- inside [0, 1], confirming the same conclusion

Scanning the mirror separation ``L`` from just above 0 out past the
symmetric cavity's confocal-adjacent limit :math:`L = 2R` shows exactly
where the resonator stops being stable:

.. code-block:: python

   def is_stable(L):
       els = [
           OpticalElement(spherical_mirror(R1)),
           OpticalElement(free_space(L), length=L),
           OpticalElement(spherical_mirror(R2)),
           OpticalElement(free_space(L), length=L),
       ]
       return cavity_stability(cavity_round_trip_matrix(els))

   print(is_stable(3.5), is_stable(4.5))  # True False

For this symmetric cavity :math:`g_1g_2 = (1-L/R)^2` crosses 1 exactly at
:math:`L = 2R = 4.0`, matching the True/False boundary found above.

The self-consistent Gaussian mode
--------------------------------------

The fundamental mode of a stable cavity is the Gaussian beam whose complex
parameter :math:`q` reproduces itself after one round trip,
:math:`q = (Aq+B)/(Cq+D)`, i.e. the fixed point of
:func:`~physicskit.optics.gaussian.propagate_q`. Solving the resulting
quadratic :math:`Cq^2 + (D-A)q - B = 0` and keeping the root with
:math:`\operatorname{Im}(q) > 0` (a physical beam, not its mirror image)
gives the mode at the reference plane just after mirror 1:

.. code-block:: python

   from physicskit.optics.gaussian import q_to_beam_params

   A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
   roots = np.roots([C, D - A, -B])
   q = roots[roots.imag > 0][0]
   print(q)  # (0.5+0.866...j)

   wavelength = 1.064e-3  # Nd:YAG, same length units as R1, R2, L
   w, R = q_to_beam_params(q, wavelength)
   print(w, R)  # ~0.0198  2.0

The recovered radius of curvature, :math:`R \approx 2.0`, exactly matches
mirror 1's own radius -- exactly what "self-consistent" means: the
wavefront leaving mirror 1 has the same curvature as mirror 1 itself, so
it retraces its path on reflection.

See Also
--------

- :doc:`/history/optics_breakthroughs` for Kogelnik and Li's 1966 paper
  that introduced this ABCD/:math:`q`-parameter treatment of laser
  resonators.
- :doc:`/api/optics`
