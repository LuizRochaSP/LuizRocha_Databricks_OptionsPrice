from ibov_barrier.di_curve import DiscountCurve, price_di_pre_swap


def test_swap_npv_sign():
    curve = DiscountCurve.from_di1([252], [0.10])
    pay_fixed = price_di_pre_swap(curve, 0, 252, 0.12, side="pay_fixed")
    receive_fixed = price_di_pre_swap(curve, 0, 252, 0.12, side="receive_fixed")
    assert pay_fixed.npv < 0
    assert receive_fixed.npv > 0
    assert abs(pay_fixed.npv + receive_fixed.npv) < 1e-6


def test_dv01_finite():
    curve = DiscountCurve.from_di1([252, 504], [0.10, 0.11])
    result = price_di_pre_swap(curve, 0, 504, 0.11, side="pay_fixed")
    assert abs(result.dv01) > 0