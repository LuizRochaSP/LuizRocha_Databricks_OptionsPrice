from ibov_barrier.di_curve import DiscountCurve


def test_flat_curve_returns_constant_rate():
    curve = DiscountCurve.from_di1([252, 504, 756], [0.10, 0.10, 0.10])
    for t in [100, 252, 400, 600, 756]:
        assert abs(curve.rate(t) - 0.10) < 1e-10


def test_df_at_vertex_matches_input():
    rates = [0.11, 0.12, 0.13]
    tenors = [252, 504, 756]
    curve = DiscountCurve.from_di1(tenors, rates)
    for t, r in zip(tenors, rates):
        expected_df = (1 + r) ** (-t / 252)
        assert abs(curve.df(t) - expected_df) < 1e-12


def test_df_monotonic():
    curve = DiscountCurve.from_di1([252, 504, 756, 1008], [0.10, 0.11, 0.12, 0.13])
    dfs = [curve.df(t) for t in range(100, 1009, 50)]
    assert all(dfs[i] > dfs[i + 1] for i in range(len(dfs) - 1))


def test_forward_rate_positive():
    curve = DiscountCurve.from_di1([252, 504, 756], [0.10, 0.12, 0.14])
    fwd = curve.forward_rate(252, 504)
    assert fwd > 0


def test_parallel_bump():
    curve = DiscountCurve.from_di1([252, 504], [0.10, 0.10])
    bumped = curve.bump_parallel(100)
    assert abs(bumped.rate(252) - 0.11) < 1e-10
    assert abs(bumped.rate(504) - 0.11) < 1e-10


def test_key_rate_bump_isolated():
    curve = DiscountCurve.from_di1([252, 504, 756], [0.10, 0.11, 0.12])
    bumped = curve.bump_key(504, 100)
    assert abs(bumped.rate(252) - 0.10) < 1e-10
    assert abs(bumped.rate(504) - 0.12) < 1e-10
    assert abs(bumped.rate(756) - 0.12) < 1e-10