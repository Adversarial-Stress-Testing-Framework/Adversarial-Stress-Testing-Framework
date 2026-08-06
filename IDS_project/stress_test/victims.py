"""
The victim models under audit.

Three families, chosen so the robustness result cannot be dismissed as a quirk
of any one of them:

  LinearSVC       the existing baseline; differentiable, attackable directly
  RandomForest    no gradient at all, so it can only be reached by transfer;
                  also the model app.py actually ships
  MLP             a small neural net - what production ML-IDS increasingly use,
                  and the case where PGD's extra iterations genuinely matter

Each entry reports whether a direct gradient attack is possible, so the runner
knows to fall back to transfer for the ones where it isn't.
"""

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from stress_test import attacks


RANDOM_STATE = 42


def train_linear_svc(X_train, y_train):
    model = LinearSVC(random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.3,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


VICTIMS = {
    "LinearSVC": {
        "train": train_linear_svc,
        "gradient_fn": attacks.linear_gradient_fn,
    },
    "RandomForest": {
        "train": train_random_forest,
        # Decision trees are piecewise-constant: the gradient is zero almost
        # everywhere and undefined at the splits. There is nothing to descend,
        # so this model is only reachable by transferring another model's work.
        "gradient_fn": None,
    },
    "XGBoost": {
        "train": train_xgboost,
        # Gradient-boosted trees are still trees. XGBoost computes gradients of
        # the loss w.r.t. its own predictions during training, but the model it
        # produces is piecewise-constant in the input, so there is no usable
        # input gradient for an evasion attack either. Transfer only.
        "gradient_fn": None,
    },
    "MLP": {
        "train": train_mlp,
        "gradient_fn": attacks.mlp_gradient_fn,
    },
}


def train_all(X_train, y_train, verbose=True):
    """Train every victim; returns {name: (model, gradient_fn_or_None)}."""
    trained = {}
    for name, spec in VICTIMS.items():
        if verbose:
            print(f"  training {name} ...", flush=True)
        model = spec["train"](X_train, y_train)
        grad_factory = spec["gradient_fn"]
        trained[name] = (model, grad_factory(model) if grad_factory else None)
    return trained
