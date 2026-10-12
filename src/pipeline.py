from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_data
from src.evaluation import evaluate_model, save_outputs
from src.model import train_model
from src.preprocessing import preprocess_data
from src.validator import validate_data


def run_pipeline(input_path: str | Path | None = None):
	project_root = Path(__file__).resolve().parents[1]
	if input_path is None:
		input_path = project_root / "data" / "raw" / "Telco-Customer-Churn.csv"
	else:
		input_path = Path(input_path)
		if not input_path.is_absolute():
			input_path = project_root / input_path
	processed_path = project_root / "data" / "processed" / "Telco-Customer-Churn-cleaned.csv"
	output_dir = project_root / "output"

	rows_loaded = 0
	validation_status = "Not complete"
	cleaning_status = "Not complete"
	training_status = "Not trained"
	evaluation_status = "Not complete"
	predictions_status = "Not saved"
	metrics = None
	error_message = None

	try:
		data = load_data(input_path)
		rows_loaded = len(data)

		processed_data = preprocess_data(data, processed_path)
		cleaning_status = "Complete"

		validate_data(processed_path)
		validation_status = "Complete"

		features = processed_data.drop(columns=["Churn", "customerID", "TotalCharges"])
		target = processed_data["Churn"].astype(int)
		X_train, X_test, y_train, y_test, _, test_ids = train_test_split(
			features,
			target,
			processed_data["customerID"],
			test_size=0.2,
			random_state=10,
			stratify=target,
		)

		preprocessor = ColumnTransformer(
			transformers=[
				(
					"categorical",
					OneHotEncoder(handle_unknown="ignore"),
					X_train.select_dtypes(exclude="number").columns,
				),
				("numeric", StandardScaler(), X_train.select_dtypes(include=["number"]).columns),
			]
		)

		model = train_model(X_train, y_train, preprocessor)
		training_status = "Trained"

		metrics, predictions = evaluate_model(model, X_test, y_test, test_ids)
		evaluation_status = "Complete"

		save_outputs(metrics, predictions, output_dir)
		predictions_status = "Saved"
	except Exception as error:
		error_message = str(error)


	print("========================================")
	print("TELCO CUSTOMER CHURN ANALYSIS")
	print("========================================")
	print(f"Rows loaded = {rows_loaded}")
	print(f"Validation = {validation_status}")
	print(f"Cleaning data = {cleaning_status}")
	print(f"Training model = {training_status}")
	print(f"Evaluating model = {evaluation_status}")

	if metrics is not None:
		print(f"Accuracy = {metrics['accuracy']:.4f}")
		print(f"Precision = {metrics['precision']:.4f}")
		print(f"Recall = {metrics['recall']:.4f}")
		print(f"F1 Score = {metrics['f1']:.4f}")
		print(f"Confusion Matrix = {metrics['confusion_matrix']}")

	if predictions_status == "Saved":
		print(f"Predictions saved to: {output_dir / 'predictions.csv'}")
	else:
		print(f"Predictions saved = {predictions_status}")

	if error_message is not None:
		print(f"Error = {error_message}")
