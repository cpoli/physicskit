Relativistic Two-Body Decay Kinematics
==========================================

This tutorial uses :mod:`physicskit.particle.kinematics` and
:mod:`physicskit.particle.decays` to build a two-body decay in a parent
particle's rest frame, boost it into a moving ("lab") frame, and check
two of the most basic relativistic-kinematics invariants: invariant mass
and rapidity additivity.

Decaying a parent particle at rest
--------------------------------------

:func:`~physicskit.particle.decays.two_body_decay` builds both daughters'
four-momenta for a parent of mass ``M`` decaying to daughters of mass
``m1``, ``m2``, emitted at a chosen angle:

.. code-block:: python

   import numpy as np
   from physicskit.particle.kinematics import boost, boost_to_com, invariant_mass, rapidity
   from physicskit.particle.decays import two_body_decay

   M, m1, m2 = 1.0, 0.3, 0.2
   p1, p2 = two_body_decay(M, m1, m2, cos_theta=0.6, phi=0.4)
   print(p1)  # FourVector(E=0.525, px=0.3175, py=0.1342, pz=0.2585)
   print(p2)  # FourVector(E=0.475, px=-0.3175, py=-0.1342, pz=-0.2585)

The two daughters are exactly back-to-back in the parent's rest frame
(``p1.p_vec + p2.p_vec`` is the zero vector), and reassembling their
four-momenta with :func:`~physicskit.particle.kinematics.invariant_mass`
recovers the parent mass exactly:

.. code-block:: python

   print(invariant_mass([p1, p2]))  # 1.0

Boosting into a moving frame
----------------------------------

Physically, a real parent particle is rarely at rest -- it was itself
produced moving, e.g. in a beam or a previous decay.
:func:`~physicskit.particle.kinematics.boost` moves any four-vector into
a frame where the parent travels at velocity ``beta`` along a chosen
axis. Applying the *same* boost to both daughters is exactly what
"boosting the whole decay into the lab frame" means:

.. code-block:: python

   beta_lab = 0.6
   p1_lab = boost(p1, beta_lab, axis="z")
   p2_lab = boost(p2, beta_lab, axis="z")
   print(p1_lab)  # FourVector(E=0.8501, px=0.3175, py=0.1342, pz=0.7169)
   print(p2_lab)  # FourVector(E=0.3999, px=-0.3175, py=-0.1342, pz=0.03312)

Invariant mass is, true to its name, exactly invariant -- reconstructing
it from the boosted daughters still gives the parent mass:

.. code-block:: python

   print(invariant_mass([p1_lab, p2_lab]))  # 1.0 (to floating-point precision)

This is precisely the technique real particle-physics experiments use to
discover unstable particles they cannot detect directly: measure the
momenta of two decay products, and look for a peak in their reconstructed
invariant mass at the parent's mass.

Rapidity is additive under a boost
----------------------------------------

Unlike ordinary velocity, rapidity :math:`y=\tfrac12\ln[(E+p_z)/(E-p_z)]`
adds linearly under a further boost along the same axis -- a boost by
:math:`\beta` shifts every particle's rapidity by exactly
:math:`\operatorname{artanh}\beta`, regardless of its own momentum. That
makes rapidity the natural variable for comparing particles produced in
different reference frames:

.. code-block:: python

   y_rest = rapidity(p1)
   y_lab = rapidity(p1_lab)
   print(y_lab - y_rest)          # 0.6931...
   print(np.arctanh(beta_lab))    # 0.6931... -- an exact match

As a final consistency check,
:func:`~physicskit.particle.kinematics.boost_to_com` recovers the
boost velocity of the daughters' own center-of-momentum frame relative
to the lab -- exactly the ``beta_lab`` we applied:

.. code-block:: python

   print(boost_to_com([p1_lab, p2_lab]))  # [0. 0. 0.6]

See Also
--------

- :doc:`/history/particle_breakthroughs`
- :doc:`/api/particle`
