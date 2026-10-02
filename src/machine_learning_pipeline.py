import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(r"C:\Users\USER\Downloads\UST Capstone\data\processed\Telco-Customer-Churn-cleaned.csv")

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=["Churn", "customerID", "TotalCharges"]) #drop the target variable and customerID from the feature matrix
y = df["Churn"]

X = pd.get_dummies(X, drop_first=True)
print(X.dtypes)

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(include=[np.number]).columns

#Start of machine learning. Re-visit for feature selection and model selection. For now, we will use a simple logistic regression model.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#add churn probability to the output
churn_probability = model.predict_proba(X_test)[:, 1]

#Matching the customerID with the test set
test_customer_id = df.loc[X_test.index, "customerID"]
prediction_output = pd.DataFrame({
    "customerID": test_customer_id,
    "Actual": y_test,
    "Predicted": y_pred,
    "Churn_Probability": churn_probability
})
print(prediction_output.head())

#Save the predictions to a CSV file
prediction_output.to_csv(r"C:\Users\USER\Downloads\UST Capstone\output\predictions.csv", index=False)

