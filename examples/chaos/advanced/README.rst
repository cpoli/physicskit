Advanced: Building Custom Systems
---------------------------------

An example showing how to use the low-level, Numba-accelerated integrators
in :mod:`physicskit.chaos.core.integrators` directly -- either by subclassing
:class:`physicskit.chaos.core.base_system.DynamicalSystem` the same way the built-in
systems in :mod:`physicskit.chaos.systems.continuous` do, or by calling the
integrators on a hand-written right-hand-side function with no wrapper class
at all.
