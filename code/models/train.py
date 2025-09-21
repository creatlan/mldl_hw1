from pathlib import Path
from code.datasets.prepare_data import load_and_prepare
from code.models.models import train_model, evaluate_model
from code.models.save_model import save


def main():
    data = load_and_prepare('code/datasets/diabetes.csv')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    scaler = data['scaler']

    model = train_model(X_train, y_train)
    accuracy, cm = evaluate_model(model, X_test, y_test)

    print(f'Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(cm)

    model_path, scaler_path = save(model, scaler, name='logreg')
    print(f'Model saved to: {model_path}')
    print(f'Scaler saved to: {scaler_path}')


if __name__ == '__main__':
    main()
