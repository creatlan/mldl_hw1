import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'diabetes.csv'

def load_and_prepare(path: str = DATA_PATH):
	df = pd.read_csv(path)

	cols_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']  # colls where 0 is invalid
	for col in cols_replace_zero:
		if col in df.columns:
			median = df.loc[df[col] != 0, col].median()
			df[col] = df[col].replace(0, median)

	X = df.drop('Outcome', axis=1)
	y = df['Outcome']

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(
		X_scaled, y, test_size=0.2, random_state=42
	)

	df.to_csv('diabetes_processed.csv', index=False)

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
