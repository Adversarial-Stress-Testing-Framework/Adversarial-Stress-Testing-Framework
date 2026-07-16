import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(path="KDDTrain+.txt"):
    return pd.read_csv(path, header=None)


def clean_data(df):
    df = df.dropna(axis=1, how="all")
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=constant_cols)


def encode_and_split(df):
    X = df.iloc[:, :-2].copy()
    y = df.iloc[:, -2].apply(lambda v: 0 if str(v).strip() == "normal" else 1)

    cat_cols = X.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    return X, y


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
