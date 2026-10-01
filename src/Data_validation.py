import pandas as pd

df = pd.read_csv(r"C:\Users\USER\Downloads\UST Capstone\data\raw\Telco-Customer-Churn.csv")

#to check the number of rows and columns in the dataset
rows, columns = df.shape 

#to check for sum of duplicated values in customer_IDs
duplicated_customers = df["customerID"].duplicated().sum()

#Column containing missing values  
missing_values = df.isnull().sum()

#Column containing blank values
blank_counts = (df.astype(str).apply(lambda x: x.str.strip() == "")).sum()

# Check column behave as expected #to visit
TotalCharges_desc = df["TotalCharges"].describe()

# To check the tenure of the customer
Minimum_tenure = df['tenure'].min()
Maximum_tenure = df['tenure'].max()
Median_tenure = df['tenure'].median()

# Check the data type of all columns
data_types = df.dtypes


print(f"----------------------")
print(f"DATA VALIDATION REPORT")
print(f"----------------------")
print(f"Number of rows: {rows}") #print total number of rows
print(f"----------------------")
print(f"Number of columns: {columns}") #print total number of columns
print(f"----------------------")
print(f"Number of duplicated customer IDs: {duplicated_customers}") #print number of duplicated customer IDs
print(f"----------------------")
print(f"Number of missing values:") #print number of missing values
for column, count in missing_values.items():
    print(f"{column}: {count}")
print(f"----------------------")
print(f"Number of blank values:") #print number of blank values
for column, count in blank_counts.items():
    if count > 0:
        print(f"{column}: {count}")
print(f"----------------------")
print(f"Tenure statistics:") #print the minimum, maximum, and median tenure values
print(f"  Minimum tenure: {Minimum_tenure}")
print(f"  Maximum tenure: {Maximum_tenure}")
print(f"  Median tenure: {Median_tenure}")
print(f"----------------------")
print(f"Data types of columns:") #print the data types of all columns
for column, dtype in data_types.items():
    print(f"{column}: {dtype}")
print(f"----------------------")
