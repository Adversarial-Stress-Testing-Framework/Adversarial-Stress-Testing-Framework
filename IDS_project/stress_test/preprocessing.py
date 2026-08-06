import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# The 41 NSL-KDD feature columns, in file order, followed by the two trailing
# columns (class label and difficulty level). Naming them up front lets the
# constraint engine address features semantically instead of by position, which
# matters because clean_data() drops all-constant columns and shifts the indices.
NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

NSL_KDD_COLUMNS = NSL_KDD_FEATURES + ["label", "difficulty"]


def load_dataset(path="KDDTrain+.txt"):
    df = pd.read_csv(path, header=None)
    if df.shape[1] == len(NSL_KDD_COLUMNS):
        df.columns = NSL_KDD_COLUMNS
    return df


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
