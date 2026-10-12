from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Telco-Customer-Churn-cleaned.csv"
TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]


def overall_churn_percentage(data):
	churned_customers = data["Churn"].sum()
	total_customers = len(data)
	churn_percentage = churned_customers / total_customers * 100
	print(f"Overall churn: {churn_percentage:.2f}%")


def churn_by_contract(data):
	churn_rates = data.groupby("Contract")["Churn"].mean() * 100
	print("\nChurn by contract type:")
	for contract, rate in churn_rates.items():
		print(f"  {contract}: {rate:.2f}%")


def churn_by_internet_service(data):
	churn_rates = data.groupby("InternetService")["Churn"].mean() * 100
	print("\nChurn by internet service:")
	for service, rate in churn_rates.items():
		print(f"  {service}: {rate:.2f}%")


def churn_by_tenure_range(data):
	tenure_ranges = pd.cut(
		data["tenure"],
		bins=TENURE_BINS,
		labels=TENURE_LABELS,
		include_lowest=True,
	)
	churn_rates = data.groupby(tenure_ranges, observed=False)["Churn"].mean() * 100
	print("\nChurn by tenure range:")
	for tenure_range, rate in churn_rates.items():
		print(f"  {tenure_range}: {rate:.2f}%")


def main():
	data = pd.read_csv(DATA_PATH)
	overall_churn_percentage(data)
	churn_by_contract(data)
	churn_by_internet_service(data)
	churn_by_tenure_range(data)


if __name__ == "__main__":
	main()
