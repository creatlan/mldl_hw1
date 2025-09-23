import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'diabetes.csv'


def load_and_prepare(path: str | Path = DATA_PATH, output_path: str | Path | None = None, overwrite: bool = True):

	path = Path(path)

	df = pd.read_csv(path)

	cols_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
	for col in cols_replace_zero:
		if col in df.columns:
			median = df.loc[df[col] != 0, col].median()
			df[col] = df[col].replace(0, median)

	X = df.drop('Outcome', axis=1)
	y = df['Outcome']

	X_train_raw, X_test_raw, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train_raw)
	X_test = scaler.transform(X_test_raw)

	if output_path is None:
		output_path = Path(__file__).resolve().parents[1] / 'data' / 'diabetes_processed.csv'
	else:
		output_path = Path(output_path)

	output_path.parent.mkdir(parents=True, exist_ok=True)

	if output_path.exists() and not overwrite:
		raise FileExistsError(f"Output file {output_path} already exists (overwrite=False).")

	df.to_csv(output_path, index=False)

	return {
		'X_train': X_train,
		'X_test': X_test,
		'y_train': y_train,
		'y_test': y_test,
		'scaler': scaler,
		'df': df,
	}


if __name__ == '__main__':
    _ = load_and_prepare()
