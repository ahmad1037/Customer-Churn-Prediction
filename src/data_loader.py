from pathlib import Path
import pandas as pd

from src.config import DATA_DIR

DATA_PATH = DATA_DIR / "raw" / "Telco_customer_churn.csv"


def load_data() -> pd.DataFrame:
    """Load raw Telco customer churn dataset."""
    return pd.read_csv(DATA_PATH)