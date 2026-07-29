from sklearn.model_selection import train_test_split


def split_dataset(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    )