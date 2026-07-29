from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

def plot_confusion_matrix(model, X_test, y_test):

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues",
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "images/confusion_matrix.png",
        dpi=300,
    )

    plt.show()