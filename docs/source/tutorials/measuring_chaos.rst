Measuring Chaos: Lyapunov Exponents and Fractal Dimensions
===========================================================

"Chaotic" is a quantitative claim: nearby trajectories separate
exponentially, at a rate given by the largest Lyapunov exponent
:math:`\lambda_1 > 0`, and the attractor they settle on has a
non-integer dimension. This tutorial measures both with
:mod:`physicskit.chaos`, on three systems of increasing complexity: the
logistic map, the Hénon map and the Lorenz flow. At each step the
numerical answer is checked against a result known in closed form, so
you can tell a real measurement from an artefact of the method.

Everything here is dimensionless. The maps count time in iterations and
the Lorenz equations in their own natural time unit.

The logistic map: from order to chaos
-------------------------------------

The logistic map :math:`x_{n+1} = r x_n (1 - x_n)`
(:class:`~physicskit.chaos.LogisticMap`) is the simplest system with a
period-doubling route to chaos (May, *Nature* 261, 459, 1976). Iterate
it past its transient and count the distinct points it keeps visiting,
then estimate :math:`\lambda` with
:func:`~physicskit.chaos.utils.map_lyapunov_spectrum`, which evolves a
tangent vector along the orbit (the Benettin QR method):

.. code-block:: python

   import numpy as np
   import physicskit as pk
   from physicskit.chaos.utils import map_lyapunov_spectrum

   for r in [2.8, 3.2, 3.5, 3.56, 4.0]:
       logistic = pk.chaos.LogisticMap(r=r)
       orbit = logistic.trajectory(np.array([0.2]), n_iter=2000)
       period = len(np.unique(np.round(orbit[-64:, 0], 6)))
       lam = map_lyapunov_spectrum(logistic, np.array([0.2]), n_iter=20000)[0]
       print(f"r={r:<5} distinct points={period:<3} lambda={lam:+.4f}")

   # r=2.8   distinct points=1   lambda=-0.2231
   # r=3.2   distinct points=2   lambda=-0.9163
   # r=3.5   distinct points=4   lambda=-0.8725
   # r=3.56  distinct points=8   lambda=-0.0771
   # r=4.0   distinct points=64  lambda=+0.6933

The period doubles 1 → 2 → 4 → 8, and :math:`\lambda` stays negative
(stable cycles attract nearby orbits) until the cascade ends and
:math:`\lambda` turns positive. Three of these rows can be checked
exactly:

- At :math:`r = 2.8` the orbit sits on the fixed point
  :math:`x^* = 1 - 1/r`, where :math:`f'(x^*) = 2 - r`, so
  :math:`\lambda = \ln|2 - r| = \ln 0.8 = -0.2231`.
- At :math:`r = 3.2` the 2-cycle has multiplier
  :math:`f'(x_1) f'(x_2) = 4 + 2r - r^2`, so
  :math:`\lambda = \tfrac12 \ln|4 + 2r - r^2| = \tfrac12\ln 0.16 = -0.9163`.
- At :math:`r = 4` the map is conjugate to the tent map, giving
  :math:`\lambda = \ln 2 = 0.6931` (Ulam and von Neumann, 1947).

Near :math:`r = 3.56`, :math:`\lambda` is close to zero because the
cascade is about to reach the Feigenbaum point
:math:`r_\infty = 3.5699...`, where the period becomes infinite.

The Hénon map: a strange attractor in two dimensions
----------------------------------------------------

The Hénon map (:class:`~physicskit.chaos.HenonMap`, Hénon, *Commun. Math.
Phys.* 50, 69, 1976) has two Lyapunov exponents. Its Jacobian has the
constant determinant :math:`-b`, so every area shrinks by :math:`|b|`
per step. That fixes the *sum* of the exponents exactly,
:math:`\lambda_1 + \lambda_2 = \ln b`, which gives a sharp test of the
numerical spectrum:

.. code-block:: python

   from physicskit.chaos.utils import box_counting_dimension

   henon = pk.chaos.HenonMap(a=1.4, b=0.3)
   spectrum = map_lyapunov_spectrum(henon, np.array([0.1, 0.1]), n_iter=50000)
   print(spectrum.round(4), spectrum.sum().round(4), np.log(0.3).round(4))
   # [ 0.4187 -1.6226] -1.204 -1.204

One exponent is positive, so the map stretches, and the sum is negative,
so it contracts area overall. An attractor with both properties is a
fractal. The Kaplan-Yorke conjecture predicts its dimension from the
spectrum alone, :math:`D_{KY} = 1 + \lambda_1/|\lambda_2|` (Kaplan and
Yorke, 1979). Compare that with a direct box count of the attractor:

.. code-block:: python

   kaplan_yorke = 1 + spectrum[0] / abs(spectrum[1])
   orbit = henon.trajectory(np.array([0.1, 0.1]), n_iter=200000)[1000:]
   D0, eps, counts = box_counting_dimension(orbit)
   print(round(kaplan_yorke, 3), round(D0, 3))
   # 1.258 1.303

Both land near the accepted :math:`D \approx 1.26`. The box count comes
out slightly high because a finite orbit covers the fine boxes unevenly.
:func:`~physicskit.chaos.utils.box_counting_dimension` returns
``eps`` and ``counts`` so you can plot :math:`\log N(\epsilon)` against
:math:`\log(1/\epsilon)` and check that the fit really covers a straight
section.

The Lorenz attractor: flows and the zero exponent
-------------------------------------------------

For a continuous flow, :func:`~physicskit.chaos.utils.benettin_lyapunov_spectrum`
integrates the trajectory and its tangent space together. The Lorenz
system (:class:`~physicskit.chaos.Lorenz`) has divergence
:math:`-(\sigma + 1 + \beta)` everywhere, so again the sum is known
exactly, and any bounded, non-fixed-point trajectory of an autonomous
flow must have one exponent equal to zero (the direction along the flow
itself):

.. code-block:: python

   from physicskit.chaos.utils import benettin_lyapunov_spectrum, correlation_dimension

   lorenz = pk.chaos.Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
   spectrum = benettin_lyapunov_spectrum(
       lorenz, np.array([1.0, 1.0, 1.0]), dt=0.01, n_steps=100000, n_transient=2000
   )
   print(spectrum.round(3), spectrum.sum().round(3))
   # [ 9.040e-01 -1.000e-03 -1.457e+01] -13.667

The spectrum :math:`(0.904, 0, -14.57)` matches the literature values
(e.g. Sprott, *Chaos and Time-Series Analysis*, 2003), and the sum
is :math:`-(10 + 1 + 8/3) = -13.667` to the digits shown.

The attractor's correlation dimension :math:`D_2` comes from the
Grassberger-Procaccia correlation sum (*Phys. Rev. Lett.* 50, 346,
1983), as the slope of :math:`\log C(\epsilon)` against
:math:`\log\epsilon`. That slope is only meaningful over the *scaling
region*. At large :math:`\epsilon` the sum saturates at the size of the
attractor, and at small :math:`\epsilon` it runs out of point pairs:

.. code-block:: python

   t, states = lorenz.trajectory(np.array([1.0, 1.0, 1.0]), dt=0.01, n_steps=200000)
   points = states[5000::50][:3000]

   D2_default, _, _ = correlation_dimension(points)
   D2, eps, C = correlation_dimension(points, eps_min=0.5, eps_max=5.0)
   print(round(D2_default, 3), round(D2, 3))
   # 1.681 2.047

With the default range (the 5th to 50th percentile of pair distances)
the fit reaches into the saturated region and returns 1.68. Restricting
it to :math:`0.5 \le \epsilon \le 5` gives 2.05, Grassberger and
Procaccia's own value (:math:`2.05 \pm 0.01`). Before trusting a
dimension estimate, look at the returned ``eps`` and ``C`` on a log-log
plot.

Lyapunov exponents from a single time series
--------------------------------------------

Experiments rarely give you the equations of motion, only a signal.
Takens' theorem (1981) says that delay coordinates
:math:`(x(t), x(t+\tau), x(t+2\tau), \dots)` reconstruct an attractor
equivalent to the original. Rosenstein's method (*Physica D* 65, 117,
1993) then follows nearest neighbours in that reconstruction and
averages their log separation,
:func:`~physicskit.chaos.utils.average_log_divergence`. Its slope is
:math:`\lambda_1`. Here we use only the :math:`x` component of the
Lorenz run:

.. code-block:: python

   from physicskit.chaos.utils import average_log_divergence

   lorenz = pk.chaos.Lorenz()
   t, states = lorenz.trajectory(np.array([1.0, 1.0, 1.0]), dt=0.01, n_steps=65000)
   x = states[5000:, 0]  # observe only x(t), sampled every dt = 0.01

   steps, divergence = average_log_divergence(x, dim=3, tau=10, min_tsep=100, max_iter=300)
   for lo, hi in [(0, 50), (50, 200)]:
       slope = np.polyfit(steps[lo:hi] * 0.01, divergence[lo:hi], 1)[0]
       print(f"fit over steps {lo:>3}-{hi:<3}: lambda = {slope:.3f}")

   # fit over steps   0-50 : lambda = 2.266
   # fit over steps  50-200: lambda = 0.915

The first half time unit is a transient. The neighbour pair starts
separated along every direction, and it takes a while to rotate onto the
most unstable one, so the curve is steeper there. Fitting that part
overestimates :math:`\lambda_1` by more than a factor of two. Fitting
the straight section after it gives 0.915, within 1% of the Benettin
value computed from the equations. The convenience wrapper
:func:`~physicskit.chaos.utils.rosenstein_lyapunov` fits from step 0, so
for a signal like this one plot the curve and choose the linear region
yourself.

Where to go next
----------------

- :func:`~physicskit.chaos.utils.iaaft_surrogate` and
  :func:`~physicskit.chaos.utils.surrogate_test` check whether a
  time series is distinguishable from linearly correlated noise at all,
  which is worth doing before estimating any exponent.
- The chaos :doc:`history </history/chaos_breakthroughs>` traces these
  ideas from Poincaré to Lorenz, and the
  :doc:`gallery </api/gallery/chaos/index>` animates the systems used here.
