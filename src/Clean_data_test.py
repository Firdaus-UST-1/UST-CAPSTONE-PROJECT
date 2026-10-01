import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\USER\Downloads\UST Capstone\data\raw\Telco-Customer-Churn.csv")

print(f"------------------------------------------------------")

#Check for missing values in the TotalCharges column # will not be used after the conversion to numeric
missing_values = df[pd.to_numeric(df["TotalCharges"], errors="coerce").isna()]
print(f"Check for missing values in the TotalCharges column")
print(f"------------------------------------------------------")
print(missing_values)

#Check for the data type of the TotalCharges column
print(f"------------------------------------------------------")
print(f"Check the data type of the TotalCharges column")
print(f"------------------------------------------------------")
print(df["TotalCharges"].dtype)

#Convert the TotalCharges column to numeric, replacing non-numeric values with NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

#Print the rows with missing values in the TotalCharges column
print(f"------------------------------------------------------")
print(f"Rows with missing values in TotalCharges:")
print(f"------------------------------------------------------")
print(df[df["TotalCharges"].isna()])

#Check for the data type of the TotalCharges column after conversion
print(f"------------------------------------------------------")
print(f"Check for data type of the TotalCharges column after conversion")
print(f"------------------------------------------------------")
print(df["TotalCharges"].dtype)

#Remove the customerID from the feature matrix
df = df.drop(columns=["customerID"])

#Check whether customerID has been removed
print(f"------------------------------------------------------")
print(f"Check whether customerID has been removed")
print(f"------------------------------------------------------")
print(df.head())
print(f"------------------------------------------------------")

#Convert Churn to a binary target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

#Check the converted Churn column
print(f"Check the converted Churn column")
print(f"------------------------------------------------------")
print(df.head())
print(f"------------------------------------------------------")

print(f"Dataset that being kept for processing:")
print(f"------------------------------------------------------")
print(f"SeniorCitizen: {df['SeniorCitizen'].dtype}")
print(f"Tenure: {df['tenure'].dtype}")
print(f"MonthlyCharges: {df['MonthlyCharges'].dtype}")
print(f"TotalCharges: {df['TotalCharges'].dtype}")
print(f"Churn: {df['Churn'].dtype}")
print(f"------------------------------------------------------")

#What percentage of customers churned
print(df["Churn"].value_counts(normalize=True) * 100)
print(f"------------------------------------------------------")

#How does churn differ by Contract type
churn_by_contract = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100
print(churn_by_contract)
print(f"------------------------------------------------------")

#How does churn differ by InternetService
churn_by_internet = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100
print(churn_by_internet)
print(f"------------------------------------------------------")

#How does churn differ by PaymentMethod
churn_by_payment = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100
print(churn_by_payment)
print(f"------------------------------------------------------")


#Save the cleaned dataset to a new CSV file
df.to_csv(r"C:\Users\USER\Downloads\UST Capstone\data\processed\Telco-Customer-Churn-cleaned.csv", index=False)