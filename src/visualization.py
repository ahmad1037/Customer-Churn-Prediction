import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)


def plot_feature_importance(model, feature_names, top_n=20):

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    importance = importance.sort_values(
        ascending=False
    ).head(top_n)

    plt.figure(figsize=(10, 7))

    importance.sort_values().plot(kind="barh")

    plt.title("Top Feature Importance")

    plt.tight_layout()

    plt.savefig(
        IMAGES_DIR / "feature_importance.png",
        dpi=300
    )

    plt.show()