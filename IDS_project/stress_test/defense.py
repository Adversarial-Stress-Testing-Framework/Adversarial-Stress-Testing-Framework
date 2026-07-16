import numpy as np


def feature_squeeze(X, bits=4):
    """Reduce numeric precision per-feature to 2**bits levels."""
    levels = 2 ** bits
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    span = np.where(X_max - X_min == 0, 1, X_max - X_min)

    X_norm = (X - X_min) / span
    X_squeezed_norm = np.round(X_norm * (levels - 1)) / (levels - 1)

    return X_squeezed_norm * span + X_min
