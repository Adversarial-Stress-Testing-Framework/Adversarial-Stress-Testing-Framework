"""
Module 0 experiment: how much does ignoring domain constraints inflate the
measured vulnerability of an ML-based IDS?

Runs the same FGSM attack twice - once free to move every feature in any
direction, once restricted to perturbations a real attacker could actually
produce - and reports the gap. Also runs constrained PGD, which only becomes a
distinct attack from FGSM once the projection step exists.

Run with: python -m stress_test.run_constrained_comparison
(from the IDS_project directory, so KDDTrain+.txt is found)
"""

import json
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from stress_test import (
    preprocessing,
    baseline_model,
    attacks,
    constraints,
    metrics,
)


EPSILON = 0.3
PGD_STEPS = 40
EPSILON_SWEEP = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
REPORT_PATH = "constrained_comparison_report.json"
RANDOM_STATE = 42


def confusion(y_pred, y_true):
    y_true = np.asarray(y_true)
    return {
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
    }


def score(y_pred, y_true, mask=None, X_clean=None, X_adv=None, baseline_acc=None):
    c = confusion(y_pred, y_true)
    out = {
        "DR (Attack Recall)": metrics.detection_rate(c["tp"], c["fn"]),
        "Normal Recall": metrics.normal_recall(c["tn"], c["fp"]),
        "FPR": metrics.false_positive_rate(c["fp"], c["tn"]),
        "Accuracy": metrics.accuracy(c["tp"], c["tn"], c["fp"], c["fn"]),
        "Balanced Accuracy": metrics.balanced_accuracy(c["tp"], c["tn"], c["fp"], c["fn"]),
    }
    if mask is not None:
        evaded = (y_pred == 0) & mask
        out["Evasion Rate"] = metrics.evasion_rate(int(evaded.sum()), int(mask.sum()))
        if X_clean is not None and X_adv is not None:
            out["MPD"] = metrics.min_perturbation_distance(X_clean, X_adv, evaded)
    if baseline_acc is not None:
        out["Robustness Score"] = metrics.robustness_score(baseline_acc, out["Accuracy"])
    return out


def show(title, report):
    print(f"\n=== {title} ===")
    print("  " + "  ".join(f"{k}: {v:.4f}" for k, v in report.items()))


def main():
    df = preprocessing.clean_data(preprocessing.load_dataset("KDDTrain+.txt"))
    X, y = preprocessing.encode_and_split(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_s, X_test_s, scaler = preprocessing.scale_features(X_train, X_test)

    model = baseline_model.train_victim_model(X_train_s, y_train)
    grad_fn = attacks.linear_gradient_fn(model)

    projector = constraints.ConstraintProjector(feature_names, scaler).fit(X_train)
    roles = projector.summary()

    print(f"Features: {len(feature_names)}")
    print(
        f"  immutable: {roles[constraints.IMMUTABLE]}   "
        f"increase-only: {roles[constraints.INCREASE_ONLY]}   "
        f"derived-rate: {roles[constraints.DERIVED_RATE]}"
    )

    # ---------------- baseline ----------------
    y_pred_clean = model.predict(X_test_s)
    baseline = score(y_pred_clean, y_test)
    show("BASELINE (clean)", baseline)
    baseline_acc = baseline["Accuracy"]

    # ---------------- unconstrained FGSM ----------------
    X_unc, mask = attacks.fgsm(model, X_test_s, y_test, EPSILON, gradient_fn=grad_fn)
    unconstrained = score(
        model.predict(X_unc), y_test, mask, X_test_s, X_unc, baseline_acc
    )
    show(f"FGSM, UNCONSTRAINED (eps={EPSILON})", unconstrained)

    viol_unc = constraints.realizability_violations(
        scaler.inverse_transform(X_unc[mask]), feature_names, projector
    )
    print(f"  realizability violations: {viol_unc}")

    # ---------------- constrained FGSM ----------------
    X_con, _ = attacks.fgsm(
        model, X_test_s, y_test, EPSILON, projector=projector, gradient_fn=grad_fn
    )
    constrained = score(
        model.predict(X_con), y_test, mask, X_test_s, X_con, baseline_acc
    )
    show(f"FGSM, CONSTRAINED (eps={EPSILON})", constrained)

    viol_con = constraints.realizability_violations(
        scaler.inverse_transform(X_con[mask]), feature_names, projector
    )
    print(f"  realizability violations: {viol_con}")

    # ---------------- constrained PGD ----------------
    X_pgd, _ = attacks.pgd(
        model, X_test_s, y_test, EPSILON, steps=PGD_STEPS,
        projector=projector, gradient_fn=grad_fn,
    )
    pgd_report = score(
        model.predict(X_pgd), y_test, mask, X_test_s, X_pgd, baseline_acc
    )
    show(f"PGD, CONSTRAINED (eps={EPSILON}, steps={PGD_STEPS})", pgd_report)

    # ---------------- budget sweep ----------------
    # The single-epsilon comparison understates the point. The gap between
    # constrained and unconstrained is widest at small budgets - exactly the
    # regime a realistic attacker operates in - and closes at large ones, where
    # an attacker permitted to rewrite any feature can trivially evade anything.
    sweep = []
    print("\n=== EVASION RATE vs ATTACK BUDGET ===")
    print(f"  {'epsilon':>8}  {'constrained':>12}  {'unconstrained':>14}  {'overstatement':>14}")
    for eps in EPSILON_SWEEP:
        X_u, m_u = attacks.fgsm(model, X_test_s, y_test, eps, gradient_fn=grad_fn)
        X_c, _ = attacks.fgsm(
            model, X_test_s, y_test, eps, projector=projector, gradient_fn=grad_fn
        )
        er_u = metrics.evasion_rate(int(((model.predict(X_u) == 0) & m_u).sum()), int(m_u.sum()))
        er_c = metrics.evasion_rate(int(((model.predict(X_c) == 0) & m_u).sum()), int(m_u.sum()))
        ratio = (er_u / er_c) if er_c > 0 else float("inf")
        sweep.append({
            "epsilon": eps,
            "constrained_evasion_rate": er_c,
            "unconstrained_evasion_rate": er_u,
            "overstatement_factor": ratio,
        })
        print(f"  {eps:>8.1f}  {er_c:>12.4f}  {er_u:>14.4f}  {ratio:>13.1f}x")

    # Attacks that never evade even at the largest budget are ones whose
    # immutable features give them away no matter what the attacker does.
    ceiling = max(s["constrained_evasion_rate"] for s in sweep)
    print(f"\n  Constrained evasion ceiling: {ceiling:.4f}")
    print(f"  -> {1 - ceiling:.2%} of attacks cannot be made to evade at any budget")
    print("     within this constraint set: their immutable features expose them.")

    # ---------------- headline ----------------
    unc_er = unconstrained["Evasion Rate"]
    con_er = max(constrained["Evasion Rate"], pgd_report["Evasion Rate"])
    factor = (unc_er / con_er) if con_er > 0 else float("inf")

    print("\n" + "=" * 68)
    print("  OVERSTATEMENT OF MEASURED VULNERABILITY")
    print(f"    unconstrained evasion rate : {unc_er:.4f}")
    print(f"    constrained evasion rate   : {con_er:.4f}  (best of FGSM / PGD)")
    print(f"    overstatement factor       : {factor:.2f}x  at eps={EPSILON}")
    print("=" * 68)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "epsilon": EPSILON,
        "pgd_steps": PGD_STEPS,
        "n_features": len(feature_names),
        "feature_roles": roles,
        "sections": {
            "baseline_clean": baseline,
            "fgsm_unconstrained": {**unconstrained, "realizability_violations": viol_unc},
            "fgsm_constrained": {**constrained, "realizability_violations": viol_con},
            "pgd_constrained": pgd_report,
        },
        "budget_sweep": sweep,
        "headline": {
            "unconstrained_evasion_rate": unc_er,
            "constrained_evasion_rate": con_er,
            "overstatement_factor": factor,
            "constrained_evasion_ceiling": ceiling,
            "certifiably_unevadable_fraction": 1 - ceiling,
        },
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
