from pathlib import Path

import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from src.evaluate import evaluate_models


def tune_gradient_boosting(
    X_train_processed,
    y_train,
    X_test_processed,
    y_test,
):
    """
    Tune Gradient Boosting Classifier using GridSearchCV.
    """

    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(
            random_state=42
        ),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=2,
    )

    # Train GridSearchCV
    grid_search.fit(
        X_train_processed,
        y_train,
    )

    print("Best Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation ROC-AUC:")
    print(grid_search.best_score_)

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate best model
    evaluation = evaluate_models(
        {
            "Gradient Boosting (Tuned)": best_model
        },
        X_test_processed,
        y_test,
    )

    print("\nEvaluation:")
    print(evaluation)

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save model
    model_path = (
        models_dir
        / "gradient_boosting_tuned.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    print(
        f"\nModel saved to: {model_path}"
    )

    return best_model