from src.data_loader import load_data
from src.preprocessing import (
    remove_leakage_columns,
    split_features_target,
    get_feature_types,
    build_preprocessor,
)
from src.train_test_split import split_dataset
import json
import joblib
from pathlib import Path

df = load_data()

df = remove_leakage_columns(df)

X, y = split_features_target(df)

num_cols, cat_cols = get_feature_types(X)

preprocessor = build_preprocessor(
    num_cols,
    cat_cols,
)

X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(X, y)

X_train_processed = preprocessor.fit_transform(X_train)

X_valid_processed = preprocessor.transform(X_valid)

X_test_processed = preprocessor.transform(X_test)


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

joblib.dump(
    preprocessor,
    ARTIFACTS_DIR / "preprocessor.joblib",
)

with open(
    ARTIFACTS_DIR / "numerical_features.json",
    "w",
) as f:
    json.dump(num_cols, f, indent=4)

with open(
    ARTIFACTS_DIR / "categorical_features.json",
    "w",
) as f:
    json.dump(cat_cols, f, indent=4)