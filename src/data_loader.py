import pandas as pd


def load_data(file_path):
	# Load a CSV file into a DataFrame.
	data = pd.read_csv(file_path)
	return data