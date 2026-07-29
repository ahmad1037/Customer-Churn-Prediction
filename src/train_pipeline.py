from src.prepare_data import *
from src.save_model import save_model
from src.visualization import plot_feature_importance
from src.train_models import (
    MODELS,
    train_models,
)

from src.evaluate import evaluate_models

trained_models = train_models(
    MODELS,
    X_train_processed,
    y_train,
)

results = evaluate_models(
    trained_models,
    X_test_processed,
    y_test,
)

print(results)

results = results.sort_values(
    by="ROC-AUC",
    ascending=False,
)

results.to_csv(
    "reports/model_comparison.csv",
    index=False,
)

best_model_name = results.sort_values(
    by="ROC-AUC",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

save_model(
    best_model,
    "best_model.joblib",
)

feature_names = preprocessor.get_feature_names_out()

plot_feature_importance(
    best_model,
    feature_names
)