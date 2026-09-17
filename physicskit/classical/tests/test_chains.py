"""Targeted checks for physicskit.classical.systems.chains.SineGordonChain.kink beyond
energy conservation (covered in test_conservation.py): this exists
because of two real bugs found while building the examples gallery --
the original boost formula had a sign error, and using the default
width (2.0, arbitrary) rather than sqrt(k/m) (1.0, for the default
m=1, k=1) gave a profile that isn't actually a valid soliton and
visibly relaxes/radiates instead of propagating cleanly.
"""

from __future__ import annotations

import numpy as np

from physicskit.classical.systems.chains import SineGordonChain


def _front_position(q: np.ndarray) -> float:
    """Interpolated lattice site where q crosses pi (the kink's center)."""
    return float(np.interp(np.pi, q, np.arange(len(q))))


def test_sine_gordon_chain_defaults_to_zero_state_when_q0_p0_omitted():
    system = SineGordonChain(n=10)
    np.testing.assert_allclose(system.q, np.zeros(10))
    np.testing.assert_allclose(system.p, np.zeros(10))


def test_kink_moves_in_the_requested_direction():
    n = 200
    q_right, p_right = SineGordonChain.kink(n, center=80, width=1.0, velocity=0.4, polarity=1)
    q_left, p_left = SineGordonChain.kink(n, center=80, width=1.0, velocity=-0.4, polarity=1)

    right_mover = SineGordonChain(n=n, q0=q_right, p0=p_right)
    left_mover = SineGordonChain(n=n, q0=q_left, p0=p_left)

    res_right = right_mover.integrate((0, 50), dt=0.005, method="yoshida4")
    res_left = left_mover.integrate((0, 50), dt=0.005, method="yoshida4")

    assert _front_position(res_right.q[-1]) > _front_position(res_right.q[0])
    assert _front_position(res_left.q[-1]) < _front_position(res_left.q[0])


def test_kink_with_matched_width_keeps_its_shape():
    """width = sqrt(k/m) = 1 (the default) is the only width for which
    the profile is a genuine static/moving solution; its steepness
    should stay essentially constant as it propagates."""
    n = 200
    q0, p0 = SineGordonChain.kink(n, center=60, width=1.0, velocity=0.3, polarity=1)
    system = SineGordonChain(n=n, q0=q0, p0=p0)
    result = system.integrate((0, 150), dt=0.005, method="yoshida4")

    def measured_width(q):
        return 2.0 / np.max(np.abs(np.diff(q)))

    w0 = measured_width(result.q[0])
    wf = measured_width(result.q[-1])
    assert abs(wf - w0) / w0 < 0.15


def test_kink_with_mismatched_width_visibly_relaxes():
    """A width that does NOT match sqrt(k/m) is not a genuine soliton
    solution; its steepness should visibly change (relax toward the
    natural width) as the system evolves -- the behavior the width=1.0
    fix above avoids."""
    n = 200
    q0, p0 = SineGordonChain.kink(n, center=100, width=3.0, velocity=0.0, polarity=1)
    system = SineGordonChain(n=n, q0=q0, p0=p0)
    result = system.integrate((0, 100), dt=0.005, method="yoshida4")

    def measured_width(q):
        return 2.0 / np.max(np.abs(np.diff(q)))

    w0 = measured_width(result.q[0])
    wf = measured_width(result.q[-1])
    assert abs(wf - w0) / w0 > 0.15


def test_kink_rejects_superluminal_velocity():
    import pytest

    with pytest.raises(ValueError):
        SineGordonChain.kink(100, center=50, width=1.0, velocity=1.0)
    with pytest.raises(ValueError):
        SineGordonChain.kink(100, center=50, width=1.0, velocity=-1.5)
