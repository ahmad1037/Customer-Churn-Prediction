from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import pandas as pd

def evaluate_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:,1]

        results.append({

            "Model": name,

            "Accuracy":
                accuracy_score(y_test, predictions),

            "Precision":
                precision_score(y_test, predictions),

            "Recall":
                recall_score(y_test, predictions),

            "F1":
                f1_score(y_test, predictions),

            "ROC-AUC":
                roc_auc_score(y_test, probabilities),
        })

    return pd.DataFrame(results)