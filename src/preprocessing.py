from pathlib import Path

import pandas as pd


def preprocess_data(data: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    # Clean raw CSV fields and save the processed data.
    processed_data = data.copy()
    duplicate_customer_ids = (
        processed_data.loc[processed_data["customerID"].duplicated(keep=False), "customerID"]
        .drop_duplicates()
        .tolist()
    )
    if duplicate_customer_ids:
        print(f"Duplicated customerID values: {duplicate_customer_ids}")
    else:
        print("no duplicated Customer ID found")

    total_charges_text = processed_data["TotalCharges"].fillna("").astype(str).str.strip()
    total_charges = pd.to_numeric(total_charges_text, errors="coerce")
    missing_charges = total_charges.isna()

    print(f"Missing TotalCharges rows = {int(missing_charges.sum())}")
    if missing_charges.any():
        print(processed_data.loc[missing_charges, ["customerID","TotalCharges"]])

    processed_data["TotalCharges"] = total_charges

    # Convert Churn to binary values
    processed_data["Churn"] = (
        processed_data["Churn"].astype(str).str.strip().replace({"Yes": 1, "No": 0, "1": 1, "0": 0})
    ).astype(int)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(output_path, index=False)
    return processed_data

    