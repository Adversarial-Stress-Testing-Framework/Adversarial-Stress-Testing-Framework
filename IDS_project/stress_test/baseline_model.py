from sklearn.svm import LinearSVC


def train_victim_model(X_train, y_train):
    # LinearSVC (liblinear) instead of SVC(kernel="linear") (libsvm): same linear
    # decision boundary and coef_ for FGSM gradients, but scales near-linearly
    # instead of quadratically/cubically, which matters at NSL-KDD's ~126k rows.
    model = LinearSVC(random_state=42, max_iter=5000)
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(model, X_test, y_test):
    y_pred = model.predict(X_test)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())

    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "y_pred": y_pred}
