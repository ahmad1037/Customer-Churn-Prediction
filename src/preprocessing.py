import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
LEAKAGE_COLUMNS = [
    "CustomerID",
    "Count",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
]
def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=LEAKAGE_COLUMNS)

TARGET = "Churn Value"


def split_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y

def get_feature_types(df):
    numerical_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return numerical_features, categorical_features

def build_numeric_pipeline():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

def build_categorical_pipeline():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

def build_preprocessor(
    numerical_features,
    categorical_features,
):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                build_numeric_pipeline(),
                numerical_features,
            ),
            (
                "cat",
                build_categorical_pipeline(),
                categorical_features,
            ),
        ]
    )

    return preprocessor