from src.prepare_data import *
import shap
import matplotlib.pyplot as plt
from src.save_model import save_model
from src.evaluation import plot_confusion_matrix
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
)
from src.visualization import plot_feature_importance
from src.train_models import (
    MODELS,
    train_models,
)
from src.tune_model import (
    tune_gradient_boosting,
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
print(f"Best model: {best_model_name}")
save_model(
    best_model,
    "best_model.joblib",
)

feature_names = preprocessor.get_feature_names_out()

plot_feature_importance(
    best_model,
    feature_names
)


best_model = tune_gradient_boosting(
    X_train_processed,
    y_train,
    X_test_processed,
    y_test,
)

plot_confusion_matrix(best_model, X_test_processed, y_test)


predictions = best_model.predict(X_test_processed)

print(
    classification_report(
        y_test,
        predictions,
    )
)

RocCurveDisplay.from_estimator(
    best_model,
    X_test_processed,
    y_test,
)

plt.title("ROC Curve")

plt.tight_layout()

plt.savefig(
    "images/roc_curve.png",
    dpi=300,
)

plt.show()

from sklearn.metrics import (
    PrecisionRecallDisplay,
)

PrecisionRecallDisplay.from_estimator(
    best_model,
    X_test_processed,
    y_test,
)

plt.title("Precision-Recall Curve")

plt.tight_layout()

plt.savefig(
    "images/pr_curve.png",
    dpi=300,
)

plt.show()

probabilities = best_model.predict_proba(
    X_test_processed
)[:, 1]

threshold = 0.35

predictions = (
    probabilities >= threshold
).astype(int)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions))
import numpy as np
import shap
import numpy as np
import matplotlib.pyplot as plt
import shap

# Convert sparse matrix to dense
X_test_shap = X_test_processed[:200].toarray()

X_test_shap = np.asarray(
    X_test_shap,
    dtype=np.float64,
)

# Create explainer
explainer = shap.TreeExplainer(
    best_model
)

# Calculate SHAP values
shap_values = explainer.shap_values(
    X_test_shap
)

# Handle binary classification
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# -------------------------
# SHAP SUMMARY PLOT
# -------------------------

shap.summary_plot(
    shap_values,
    X_test_shap,
    feature_names=feature_names,
    show=False,
)

plt.tight_layout()

plt.savefig(
    "images/shap_summary.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# -------------------------
# SHAP WATERFALL PLOT
# -------------------------

# Select one sample
sample_index = 0

# Create SHAP Explanation
explanation = shap.Explanation(
    values=shap_values[sample_index],
    base_values=explainer.expected_value,
    data=X_test_shap[sample_index],
    feature_names=feature_names,
)

# Plot waterfall
shap.plots.waterfall(
    explanation,
    max_display=20,
    show=False,
)

plt.tight_layout()

plt.savefig(
    "images/shap_waterfall.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("SHAP plots saved successfully.")