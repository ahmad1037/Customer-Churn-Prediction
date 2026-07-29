from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_tuned.joblib"
PREPROCESSOR_PATH = BASE_DIR / "artifacts" / "preprocessor.joblib"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)