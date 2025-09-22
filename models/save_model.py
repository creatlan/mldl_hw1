import joblib
from pathlib import Path

# Save models to code/saved_models directory
MODELS_DIR = Path(__file__).resolve().parents[1] / 'code' / 'saved_models'
MODELS_DIR.mkdir(exist_ok=True)

def save(model, scaler, name: str = 'logreg'):
    model_path = MODELS_DIR / f'{name}.joblib'
    scaler_path = MODELS_DIR / f'{name}_scaler.joblib'
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path
