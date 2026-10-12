from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import preprocess_data
from src.validator import validate_data


PROJECT_FOLDER = Path(__file__).parent.parent
DATA_PATH = PROJECT_FOLDER / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_raw_data():
    return pd.read_csv(DATA_PATH)


def test_required_columns_are_validated():
    churn_data = load_raw_data()
    data_without_column = churn_data.drop(columns=["gender"])

    # Check that validation reports the missing column.
    with pytest.raises(ValueError, match="Missing required columns: gender"):
        validate_data(data_without_column)


def test_churn_contains_expected_values_before_encoding():
    churn_data = load_raw_data()

    assert churn_data["Churn"].isin(["Yes", "No"]).all()


def test_duplicate_customer_ids_are_detected():
    churn_data = load_raw_data()
    data_with_duplicate = churn_data.copy()
    first_customer_id = data_with_duplicate.loc[0, "customerID"]
    data_with_duplicate.loc[1, "customerID"] = first_customer_id

    # Check that validation reports the duplicate ID.
    with pytest.raises(ValueError, match="customerID contains duplicate values"):
        validate_data(data_with_duplicate)


def test_total_charges_are_numeric_after_cleaning(tmp_path):
    churn_data = load_raw_data()
    # pytest creates tmp_path as a temporary folder for this test.
    cleaned_file = tmp_path / "cleaned.csv"
    cleaned_data = preprocess_data(churn_data, cleaned_file)

    assert pd.api.types.is_numeric_dtype(cleaned_data["TotalCharges"])


def test_cleaned_data_has_no_negative_charges(tmp_path):
    churn_data = load_raw_data()
    cleaned_file = tmp_path / "cleaned.csv"
    cleaned_data = preprocess_data(churn_data, cleaned_file)

    assert (cleaned_data["MonthlyCharges"] >= 0).all()
    assert (cleaned_data["TotalCharges"].dropna() >= 0).all()