from pathlib import Path

import pandas as pd
import pytest


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"


@pytest.fixture(scope="module")
def churn_data():
    return pd.read_csv(DATA_PATH)


def test_overall_churn_percentage(churn_data):
    percentages = churn_data["Churn"].value_counts(normalize=True).mul(100)

    assert percentages.to_dict() == pytest.approx({"No": 73.46, "Yes": 26.54}, abs=0.01)


def test_churn_percentage_by_contract(churn_data):
    percentages = pd.crosstab(
        churn_data["Contract"], churn_data["Churn"], normalize="index"
    ).mul(100)

    assert percentages["Yes"].to_dict() == pytest.approx(
        {"Month-to-month": 42.71, "One year": 11.27, "Two year": 2.83},
        abs=0.01,
    )


def test_churn_percentage_by_internet_service(churn_data):
    percentages = pd.crosstab(
        churn_data["InternetService"], churn_data["Churn"], normalize="index"
    ).mul(100)

    assert percentages["Yes"].to_dict() == pytest.approx(
        {"DSL": 18.96, "Fiber optic": 41.89, "No": 7.40},
        abs=0.01,
    )


def test_churn_percentage_by_tenure_range(churn_data):
    tenure_ranges = pd.cut(
        churn_data["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        include_lowest=True,
    )
    percentages = pd.crosstab(
        tenure_ranges, churn_data["Churn"], normalize="index"
    ).mul(100)

    assert tenure_ranges.notna().all()
    assert percentages["Yes"].to_dict() == pytest.approx(
        {"0-12": 47.44, "13-24": 28.71, "25-48": 20.39, "49-72": 9.51},
        abs=0.01,
    )