import numpy as np
import pytest

from physicskit.relativity.core.kerr_raytracer import render_kerr_shadow_image


def test_zero_spin_shadow_matches_schwarzschild_theory():
    outcomes, _hit_radii = render_kerr_shadow_image(
        ny=90,
        nx=90,
        screen_half_width=15.0,
        screen_half_height=15.0,
        r_observer=200.0,
        theta_observer=1.3,
        a=0.0,
        M=1.0,
        r_disk_inner=1.0,
        r_disk_outer=0.0,
    )
    captured_frac = np.mean(outcomes == 1)
    expected = np.pi * (3.0 * np.sqrt(3.0)) ** 2 / (30.0**2)
    assert captured_frac == pytest.approx(expected, rel=0.15)


def test_central_ray_is_captured():
    # a purely radial (alpha=beta=0) ray must plunge straight through the horizon;
    # a 3x3 grid puts the exact center pixel (index [1, 1]) at alpha=beta=0.
    outcomes, _ = render_kerr_shadow_image(
        ny=3,
        nx=3,
        screen_half_width=1.0,
        screen_half_height=1.0,
        r_observer=200.0,
        theta_observer=1.3,
        a=0.7,
        M=1.0,
        r_disk_inner=1.0,
        r_disk_outer=0.0,
    )
    assert outcomes[1, 1] == 1


def test_shadow_area_shrinks_and_shifts_with_spin():
    fractions = []
    shifts = []
    for a in [0.0, 0.5, 0.9]:
        outcomes, _ = render_kerr_shadow_image(
            ny=100,
            nx=100,
            screen_half_width=15.0,
            screen_half_height=15.0,
            r_observer=200.0,
            theta_observer=1.3,
            a=a,
            M=1.0,
            r_disk_inner=1.0,
            r_disk_outer=0.0,
        )
        captured = outcomes == 1
        fractions.append(np.mean(captured))
        cols = np.where(captured.any(axis=0))[0]
        shifts.append((cols.min() + cols.max()) / 2.0 - (outcomes.shape[1] - 1) / 2.0)

    # known qualitative Kerr shadow behavior: shrinks and shifts toward
    # positive alpha (the frame-dragging direction) as spin increases
    assert fractions[0] > fractions[1] > fractions[2]
    assert shifts[0] < shifts[1] < shifts[2]


def test_disk_hits_stay_within_disk_bounds():
    outcomes, hit_radii = render_kerr_shadow_image(
        ny=90,
        nx=90,
        screen_half_width=20.0,
        screen_half_height=20.0,
        r_observer=200.0,
        theta_observer=1.3,
        a=0.9,
        M=1.0,
        r_disk_inner=2.32,
        r_disk_outer=15.0,
    )
    disk_mask = outcomes == 3
    assert np.any(disk_mask)
    assert np.all(hit_radii[disk_mask] >= 2.32 - 1.0e-6)
    assert np.all(hit_radii[disk_mask] <= 15.0 + 1.0e-6)


def test_all_outcomes_are_valid_codes():
    outcomes, _ = render_kerr_shadow_image(
        ny=30,
        nx=30,
        screen_half_width=15.0,
        screen_half_height=15.0,
        r_observer=200.0,
        theta_observer=1.0,
        a=0.5,
        M=1.0,
        r_disk_inner=1.0,
        r_disk_outer=0.0,
    )
    assert np.all((outcomes >= 0) & (outcomes <= 3))
