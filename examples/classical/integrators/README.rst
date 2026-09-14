Integrators
==============

Why ``physicskit.classical`` defaults every conservative system to a symplectic
integrator (RK4's energy drifts monotonically, Verlet/Yoshida4 stay
bounded), and automated timestep selection with
:func:`~physicskit.classical.utils.stepsize.estimate_dt`.
