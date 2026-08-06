"""
Defenses under evaluation.

Three, deliberately spanning the range from "does nothing" to "actually works":

  feature_squeeze      the existing input-preprocessing defense. Already
                       diagnosed as class collapse rather than robustness;
                       retained so the comparison stays honest.

  RealizabilityFilter  rejects inputs that could not have come from a real
                       packet capture. Cheap, and it cannot be evaded by a
                       stronger gradient - only by an attacker who respects the
                       domain constraints, which is precisely the regime where
                       evasion rates collapse.

  adversarially_train  retrains the victim on its own adversarial examples,
                       labelled correctly. The standard strong baseline.
"""

import numpy as np

from stress_test import constraints


def feature_squeeze(X, bits=4):
    """Reduce numeric precision per-feature to 2**bits levels."""
    levels = 2 ** bits
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    span = np.where(X_max - X_min == 0, 1, X_max - X_min)

    X_norm = (X - X_min) / span
    X_squeezed_norm = np.round(X_norm * (levels - 1)) / (levels - 1)

    return X_squeezed_norm * span + X_min


class RealizabilityFilter:
    """Flags inputs that violate the absolute domain rules of a network flow.

    Only the constraints checkable on a single sample in isolation are used:
    integrality of counters, non-negativity, rates within [0, 1], and the
    pairwise rate sums. Relative rules - immutability, increase-only - need the
    original flow for comparison and so cannot be enforced at inference time.

    What this catches is the naive attacker who perturbs every feature freely.
    What it does not catch is an attacker who stays inside the realizable set.
    That is the honest framing: the filter is not a robustness guarantee, it is
    a forcing function that pushes the adversary into the constrained regime
    where their success rate is far lower to begin with.
    """

    def __init__(self, projector, tol=1e-6):
        # Built from a fitted ConstraintProjector so the defense enforces exactly
        # the bounds the attack side was held to - learned from real traffic, not
        # assumed. Asserting textbook bounds here flagged 5.2% of genuine flows.
        if projector.pair_caps_ is None:
            raise RuntimeError("pass a fitted ConstraintProjector")

        self.projector = projector
        self.feature_names = list(projector.feature_names)
        self.scaler = projector.scaler
        self.tol = tol

        self._idx = {n: i for i, n in enumerate(self.feature_names)}
        self._int_idx = [
            i for n, i in self._idx.items() if n in constraints.INTEGER_FEATURES
        ]
        self._rate_idx = [
            i for n, i in self._idx.items()
            if constraints.FEATURE_ROLES.get(n) == constraints.DERIVED_RATE
        ]

    def flag(self, X_scaled):
        """True where the sample could not be a real flow."""
        X = self.scaler.inverse_transform(np.asarray(X_scaled, dtype=float))
        bad = np.zeros(len(X), dtype=bool)

        if self._int_idx:
            cols = X[:, self._int_idx]
            # Relative tolerance: src_bytes reaches 1.2e9, where a float64
            # round-trip through the scaler costs more than an absolute 1e-6.
            slack = self.tol * np.maximum(1.0, np.abs(cols))
            bad |= (np.abs(cols - np.rint(cols)) > slack).any(axis=1)
            bad |= (cols < -slack).any(axis=1)

        if self._rate_idx:
            cols = X[:, self._rate_idx]
            bad |= ((cols < -self.tol) | (cols > 1 + self.tol)).any(axis=1)

        for (i, j), cap in self.projector.pair_caps_.items():
            bad |= (X[:, i] + X[:, j]) > cap + self.tol

        return bad

    def predict(self, model, X_scaled):
        """Model predictions, overridden to 'attack' wherever the input is impossible."""
        preds = np.asarray(model.predict(X_scaled)).copy()
        preds[self.flag(X_scaled)] = 1
        return preds


def make_adversarial_augmentation(model, X, y, epsilon, attack_fn,
                                  projector=None, gradient_fn=None, **kwargs):
    """Generate adversarial versions of the attack samples, keeping true labels.

    The label stays 1. That is the entire idea of adversarial training: show the
    model a perturbed attack and insist it is still an attack.
    """
    X_adv, mask = attack_fn(
        model, X, y, epsilon, projector=projector, gradient_fn=gradient_fn, **kwargs
    )
    return X_adv[mask], np.ones(int(mask.sum()), dtype=int)


def adversarially_train(train_fn, X_train, y_train, X_adv, y_adv):
    """Retrain from scratch on the clean training set plus adversarial examples."""
    X_aug = np.vstack([np.asarray(X_train), np.asarray(X_adv)])
    y_aug = np.concatenate([np.asarray(y_train), np.asarray(y_adv)])
    return train_fn(X_aug, y_aug)
