import joblib
from pathlib import Path

MODELS_DIR = Path("models")

MODELS_DIR.mkdir(exist_ok=True)

def save_model(model, filename):

    joblib.dump(
        model,
        MODELS_DIR / filename,
    )