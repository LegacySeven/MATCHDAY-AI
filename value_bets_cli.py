"""value_bets_cli.py — Terminal report of today's best value bets."""

import json

OUTPUT_PATH = "data/processed/value_bets_today.json"


def main():
    try:
        with open(OUTPUT_PATH) as fh:
            report = json.load(fh)
    except FileNotFoundError:
        print("No predictions yet — run `python live_predictions.py` first.")
        return

    if not report:
        print("No value bets found for today's fixtures.")
        return

    print(f"\n{'=' * 80}")
    print("  VALUE BETS — TOP OPPORTUNITIES TODAY")
    print(f"{'=' * 80}")
    for row in report[:10]:
        print(f"{row['fixture']:<35} | {row['league']:<12} | {row['best_bet']:<12} | edge={row['best_edge']:+.3f}")
    print()


if __name__ == "__main__":
    main()
