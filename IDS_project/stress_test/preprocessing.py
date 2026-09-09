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


# ---------------------------------------------------------------- CICIDS2017

CICIDS_DIR = "../data/cicids2017"


def load_cicids2017(path=CICIDS_DIR, files=None):
    """Load the CICIDS2017 flow-feature CSVs into one frame.

    The released files need three fixes before anything downstream works, and
    all three are silent failures if missed:

      - every column name is prefixed with a space (' Flow Duration')
      - Flow Bytes/s and Flow Packets/s contain Infinity and NaN, produced by
        flows whose duration rounds to zero
      - the eight day-files must be concatenated, and their column order is not
        identical across all of them

    Pass `files` to load a subset; the default takes every CSV in the directory.
    """
    import glob
    import os

    paths = sorted(glob.glob(os.path.join(path, "*.csv"))) if files is None else list(files)
    if not paths:
        raise FileNotFoundError(
            f"no CSVs under {path!r}. Download MachineLearningCSV.zip from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and extract it there."
        )

    frames = []
    for p in paths:
        d = pd.read_csv(p, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d)

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Infinity is not a value a flow can take; it marks a division by a
    # zero-length duration. Dropping those rows is honest - imputing them would
    # invent traffic - and they are a small fraction of the total.
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return df


def encode_and_split_cicids(df, label_col="Label", benign="BENIGN"):
    """Split CICIDS2017 into features and a binary label.

    Mirrors encode_and_split: 0 for benign, 1 for any attack class.
    """
    y = df[label_col].apply(lambda v: 0 if str(v).strip().upper() == benign else 1)
    X = df.drop(columns=[label_col]).copy()

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return X, y
