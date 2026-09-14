import argparse

from .pricing import BarrierContract, MarketData, greeks, price_down_and_out_call


def main() -> None:
    parser = argparse.ArgumentParser(description="Price an Ibovespa down-and-out call")
    parser.add_argument("--paths", type=int, default=500_000)
    parser.add_argument("--steps", type=int, default=126)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    market = MarketData(spot=150_000, rate=0.12, dividend_yield=0.0, volatility=0.22)
    contract = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)
    result = price_down_and_out_call(market, contract, paths=args.paths, steps=args.steps, seed=args.seed)
    risks = greeks(market, contract, paths=min(args.paths, 200_000), steps=args.steps, seed=args.seed)
    print(f"Barrier price: {result.price:,.2f} points")
    print(f"95% CI: [{result.ci_low:,.2f}, {result.ci_high:,.2f}]")
    print(f"Vanilla call: {result.vanilla_price:,.2f} points")
    print(f"Knock-out probability: {result.knock_out_probability:.2%}")
    for name, value in risks.items():
        print(f"{name}: {value:,.6f}")


if __name__ == "__main__":
    main()

