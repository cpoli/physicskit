from physicskit.statphys.chapters.ehrenfest_urn import EhrenfestUrn


def test_default_starts_full():
    urn = EhrenfestUrn(n_balls=50, seed=0)
    assert urn.n_left == 50


def test_step_changes_n_left_by_one():
    urn = EhrenfestUrn(n_balls=50, seed=1)
    before = urn.n_left
    after = urn.step()
    assert abs(after - before) == 1


def test_n_left_bounded():
    urn = EhrenfestUrn(n_balls=30, seed=2)
    for _ in range(500):
        urn.step()
        assert 0 <= urn.n_left <= 30


def test_entropy_maximal_at_half_full():
    urn = EhrenfestUrn(n_balls=40, n_left_init=20, seed=3)
    S_half = urn.entropy()
    urn2 = EhrenfestUrn(n_balls=40, n_left_init=40, seed=3)
    S_full = urn2.entropy()
    assert S_half > S_full


def test_relaxation_toward_equilibrium():
    urn = EhrenfestUrn(n_balls=200, seed=4)
    history = urn.run(3000)
    early = history["n_left"][:50].mean()
    late = history["n_left"][-500:].mean()
    assert abs(late - 100) < abs(early - 100)


def test_run_history_shapes():
    urn = EhrenfestUrn(n_balls=50, seed=5)
    history = urn.run(100)
    assert history["n_left"].shape == (101,)
    assert history["entropy"].shape == (101,)
    assert history["n_left"][0] == 50
