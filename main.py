import argparse
from pathlib import Path

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/Telco-Customer-Churn.csv"),
        help="Path to the input CSV file",
    )
    args = parser.parse_args()
    run_pipeline(args.input)


if __name__ == "__main__":
    main()