import pandas as pd


REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_data(df: pd.DataFrame) -> None:
    # Check the CSV data and raise an error if it contains invalid values.
    missing_columns = []
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )
    if df.empty:
        raise ValueError("Input dataset must contain at least one row.")

    errors = []

    for column in REQUIRED_COLUMNS:
        if column == "TotalCharges":
            continue

        missing_values = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if missing_values.any():
            errors.append(f"{column} contains missing or blank values")

    if df["customerID"].duplicated().any():
        errors.append("customerID contains duplicate values")

    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    if tenure.isna().any() or not tenure.between(0, 72).all():
        errors.append("tenure must contain numeric values between 0 and 72")
    elif not (tenure % 1 == 0).all():
        errors.append("tenure values must be whole numbers")

    senior_citizen = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    if senior_citizen.isna().any() or not senior_citizen.isin([0, 1]).all():
        errors.append("SeniorCitizen values must be 0 or 1")

    monthly_charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    if monthly_charges.isna().any() or (monthly_charges < 0).any():
        errors.append("MonthlyCharges must contain numeric, nonnegative values")

    total_charges_text = df["TotalCharges"].fillna("").astype(str).str.strip()
    total_charges = pd.to_numeric(total_charges_text, errors="coerce")
    blank_charges = total_charges_text.eq("")
    invalid_charges = total_charges.isna() & ~blank_charges
    if invalid_charges.any():
        errors.append("TotalCharges contains nonnumeric values")

    if not tenure.isna().any() and (blank_charges & tenure.ne(0)).any():
        errors.append("TotalCharges may be blank only when tenure is 0")
    if (total_charges.dropna() < 0).any():
        errors.append("TotalCharges must be nonnegative")

    targets = df["Churn"].astype(str).str.strip()
    allowed_targets = {"Yes", "No", "1", "0"}
    if not targets.isin(allowed_targets).all():
        errors.append("Churn values must be Yes/No or 1/0")
    elif targets.isin({"Yes", "1"}).sum() == 0 or targets.isin({"No", "0"}).sum() == 0:
        errors.append("Churn must contain both target classes")

    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))
