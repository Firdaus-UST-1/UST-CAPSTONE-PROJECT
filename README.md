# Telco Customer Churn Prediction

This project predicts whether a telecom customer may leave, based on their account and service information.

## Business Problem

Customer churn can reduce company revenue. The model's predictions can help identify patterns, but cannot guarantee who will leave.

## Project Files

```text
UST Capstone/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── model.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── preprocessing_data.py
│   └── validator.py
├── explanatory/
│   └── explanatory.py
├── Test/
│   └── test.py
├── output/
├── .github/
│   └── workflows/
│       └── ci.yml
├── main.py
├── requirement.txt
└── README.md
```

## Dataset

The input is `data/raw/Telco-Customer-Churn.csv`. It contains 7,043 customers and 21 columns about their accounts, services, bills, and churn. `Churn` is the outcome the model predicts.

**Source and attribution:** The project files do not say who published the dataset or provide its download link or license. Add the correct source and attribution here before sharing the dataset; no source has been guessed.

The original CSV is not changed. A cleaned copy is saved to `data/processed/Telco-Customer-Churn-cleaned.csv`.

## How the Project Works

When you run `main.py`, the project:

1. Reads the raw CSV and leaves it unchanged.
2. Converts `TotalCharges` to numbers. The 11 blank values become missing and stay blank because the model does not use this column. A blank is allowed only when tenure is zero. Converts `Churn` to `1` for Yes and `0` for No. Keeps `customerID` for the results, not as a model input.
3. Checks required columns, missing values, duplicate IDs, tenure, charges, and churn labels. Invalid data stops before training.
4. Splits the data into 80% for training and 20% for testing, using `random_state=10` and keeping the churn proportions similar in both groups (`stratify=Churn`).
5. Trains a Logistic Regression model (`max_iter=1000`) using 18 customer and service fields. Category answers are converted to yes/no indicators, and number fields are put on a similar scale. The model does not use `Churn`, `customerID`, or `TotalCharges` as inputs.
6. Tests predictions on 1,409 customers held aside from training.
7. Saves results in `output/predictions.csv`, including customer ID, actual churn, predicted churn, and estimated churn probability.

The 18 model inputs are `SeniorCitizen`, `tenure`, `MonthlyCharges`, `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, and `PaymentMethod`.

## Results

Results from the included dataset and split settings:

| Measure   |  Score | Meaning                                     |
| --------- | -----: | ------------------------------------------- |
| Accuracy  | 0.7892 | How often the model was correct             |
| Precision | 0.6215 | How often a predicted churner actually left |
| Recall    | 0.5267 | How many customers who left were found      |
| F1 score  | 0.5702 | A combined precision and recall score       |

## Exploratory Analysis

- **Overall:** 26.54% left and 73.46% stayed.
- **By contract:** 42.71% for month-to-month, 11.27% for one-year, and 2.83% for two-year contracts.
- **By internet service:** 41.89% for fiber optic, 18.96% for DSL, and 7.40% for no internet service.
- **By tenure:** 47.44% for 0-12 months, 28.71% for 13-24 months, 20.39% for 25-48 months, and 9.51% for 49-72 months.

Run `python .\explanatory\explanatory.py` to print these summaries. They show patterns, not proof that a service or contract caused a customer to leave.

## Tests

From the project folder, run:

```powershell
python -m pytest Test/test.py
```

The tests check that required columns are validated, raw `Churn` values are `Yes` or `No`, duplicate customer IDs are detected, cleaned `TotalCharges` values are numeric and cleaned charges are not negative.

## Limitations and Next Steps

- `TotalCharges` is not used, and results come from one train/test split.
- The dataset source and license still need to be recorded.

Possible next steps: add the dataset citation, test other models and prediction cutoffs, and add more pipeline tests.
