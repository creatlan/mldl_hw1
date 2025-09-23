"""Model utilities: training and evaluating the classifier."""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Tuple


def train_model(X_train, y_train) -> LogisticRegression:
	model = LogisticRegression(max_iter=1000)
	model.fit(X_train, y_train)
	return model


def evaluate_model(model, X_test, y_test) -> Tuple[float, list]:
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	cm = confusion_matrix(y_test, y_pred)
	return accuracy, cm


__all__ = ['train_model', 'evaluate_model']
