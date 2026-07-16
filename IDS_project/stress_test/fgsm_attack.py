import numpy as np


def fgsm_perturb(model, X, y, epsilon):
    """
    True FGSM for a linear-kernel SVM. The decision function of a linear SVM
    is f(x) = w.x + b, so d(f)/dx = w for every sample - the gradient sign is
    just sign(w), applied in the direction that pushes each sample toward the
    opposite class (attack samples get pushed toward the 'normal' side).
    """
    w = model.coef_[0]
    grad_sign = np.sign(w)

    y_pred = model.predict(X)
    correctly_flagged_attacks = (y_pred == 1) & (np.asarray(y) == 1)

    X_adv = X.copy()
    # moving against the weight vector direction pushes attack samples
    # toward the normal-class side of the decision boundary
    X_adv[correctly_flagged_attacks] -= epsilon * grad_sign

    return X_adv, correctly_flagged_attacks
