from .feature_engineering import create_features, FEATURE_COLUMNS

def chronological_split(df, train_fraction=0.70, validation_fraction=0.15):
    """Split in time order; the final segment is held out for final testing."""
    featured = create_features(df)
    n = len(featured)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))

    train = featured.iloc[:train_end]
    validation = featured.iloc[train_end:validation_end]
    test = featured.iloc[validation_end:]

    X_train, y_train = train[FEATURE_COLUMNS], train["DOM_MW"]
    X_val, y_val = validation[FEATURE_COLUMNS], validation["DOM_MW"]
    X_test, y_test = test[FEATURE_COLUMNS], test["DOM_MW"]
    return X_train, y_train, X_val, y_val, X_test, y_test
