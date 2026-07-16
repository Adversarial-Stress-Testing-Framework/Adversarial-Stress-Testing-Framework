import time
import numpy as np


def detection_rate(tp, fn):
    return tp / (tp + fn) if (tp + fn) else 0.0


def false_positive_rate(fp, tn):
    return fp / (fp + tn) if (fp + tn) else 0.0


def accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    return (tp + tn) / total if total else 0.0


def normal_recall(tn, fp):
    """Recall for the 'normal' class, i.e. 1 - FPR."""
    return tn / (tn + fp) if (tn + fp) else 0.0


def balanced_accuracy(tp, tn, fp, fn):
    """Mean of per-class recall (attack recall / DR and normal recall).
    Unlike raw accuracy, this does not reward a classifier that collapses
    toward predicting the majority class on an imbalanced dataset.
    """
    return (detection_rate(tp, fn) + normal_recall(tn, fp)) / 2


def evasion_rate(fn_adv, total_adv_samples):
    return fn_adv / total_adv_samples if total_adv_samples else 0.0


def robustness_score(baseline_acc, adversarial_acc):
    return baseline_acc - adversarial_acc


def min_perturbation_distance(X_clean, X_adv, evaded_mask):
    if not np.any(evaded_mask):
        return 0.0
    diffs = X_adv[evaded_mask] - X_clean[evaded_mask]
    distances = np.linalg.norm(diffs, axis=1)
    return float(distances.mean())


class LatencyTimer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
