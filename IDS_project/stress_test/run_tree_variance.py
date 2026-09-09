"""
Why are tree-model robustness numbers so seed-sensitive?

The multi-seed run showed RandomForest's constrained evasion ranging from 35.8%
to 80.7% across five splits - a 45 point spread on nothing but the train/test
partition - while LinearSVC and the MLP stayed inside a single point. Before
that instability can be reported as a property of tree ensembles, it has to be
separated from a possible flaw in how we attack them.

Hypothesis: it comes from the substitute. Trees are piecewise-constant, so they
are only reachable by transfer, and how faithfully a substitute happens to
mimic one particular ensemble may vary more by seed than it does for a smooth
model. If that is the cause, per-seed substitute agreement should track
per-seed evasion.

The alternative is that the ensembles themselves differ - a forest grown on one
split may simply be more fragile than one grown on another, independent of the
attack. Recording both lets us tell the two apart.

Run with: python -m stress_test.run_tree_variance
(from the IDS_project directory)
"""

import json
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from stress_test import preprocessing, attacks, constraints, metrics, victims


EPSILON = 0.3
STEPS = 20
RESTARTS = 2
SEEDS = [42, 7, 1337, 2024, 99]
TREES = ("RandomForest", "XGBoost")
REPORT_PATH = "tree_variance_report.json"


def main():
    df = preprocessing.clean_data(preprocessing.load_dataset("KDDTrain+.txt"))
    X, y = preprocessing.encode_and_split(df)
    names = list(X.columns)

    rows = []
    for seed in SEEDS:
        print(f"seed {seed} ...", flush=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        X_trs, X_tes, scaler = preprocessing.scale_features(X_tr, X_te)
        projector = constraints.ConstraintProjector(names, scaler).fit(X_tr)

        victims.RANDOM_STATE = seed
        for name in TREES:
            model = victims.VICTIMS[name]["train"](X_trs, y_tr)
            y_clean = model.predict(X_tes)
            mask = (y_clean == 1) & (np.asarray(y_te) == 1)

            sub = victims.train_substitute(X_trs, model, seed=seed)
            grad = attacks.mlp_gradient_fn(sub)

            # How faithfully does the stand-in reproduce the victim?
            agree_all = float((sub.predict(X_tes) == y_clean).mean())
            # Agreement restricted to the samples actually under attack matters
            # more than overall agreement: those are the only ones the attack
            # has to get right.
            agree_attacked = float((sub.predict(X_tes[mask]) == y_clean[mask]).mean())

            X_adv = attacks.pgd(
                sub, X_tes, y_te, EPSILON, steps=STEPS, restarts=RESTARTS,
                projector=projector, gradient_fn=grad, mask=mask,
                eval_model=model, seed=seed,
            )[0]
            evasion = metrics.evasion_rate(
                int(((model.predict(X_adv) == 0) & mask).sum()), int(mask.sum())
            )

            # A property of the victim alone, independent of any attack: how
            # much of the decision rests on features the attacker may move.
            imp = getattr(model, "feature_importances_", None)
            movable_importance = None
            if imp is not None:
                movable = np.array(
                    [projector.roles.get(n) != constraints.IMMUTABLE for n in names]
                )
                movable_importance = float(imp[movable].sum() / imp.sum())

            rows.append({
                "seed": seed, "model": name,
                "clean_DR": metrics.detection_rate(
                    int(((y_clean == 1) & (y_te == 1)).sum()),
                    int(((y_clean == 0) & (y_te == 1)).sum())),
                "substitute_agreement_all": agree_all,
                "substitute_agreement_attacked": agree_attacked,
                "constrained_evasion": evasion,
                "movable_feature_importance": movable_importance,
                "n_attackable": int(mask.sum()),
            })
            print(f"    {name:<13} evasion={evasion:.3f}  "
                  f"sub_agree(attacked)={agree_attacked:.4f}  "
                  f"movable_importance={movable_importance:.3f}", flush=True)

    # ---------------- correlations ----------------
    print("\n" + "=" * 74)
    print("Does substitute fidelity explain the spread?")
    out = {}
    for name in TREES:
        sub = [r for r in rows if r["model"] == name]
        ev = np.array([r["constrained_evasion"] for r in sub])
        ag = np.array([r["substitute_agreement_attacked"] for r in sub])
        mi = np.array([r["movable_feature_importance"] for r in sub])

        def corr(a, b):
            # Constant input makes Pearson undefined rather than zero.
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        out[name] = {
            "evasion_range": [float(ev.min()), float(ev.max())],
            "evasion_std": float(ev.std(ddof=1)),
            "corr_evasion_vs_substitute_agreement": corr(ev, ag),
            "corr_evasion_vs_movable_importance": corr(ev, mi),
            "substitute_agreement_range": [float(ag.min()), float(ag.max())],
            "movable_importance_range": [float(mi.min()), float(mi.max())],
        }
        o = out[name]
        print(f"\n  {name}")
        print(f"    evasion spread          : {ev.min():.3f} - {ev.max():.3f}  "
              f"(std {ev.std(ddof=1):.3f})")
        print(f"    substitute agreement    : {ag.min():.4f} - {ag.max():.4f}")
        print(f"    movable importance      : {mi.min():.3f} - {mi.max():.3f}")
        print(f"    corr(evasion, sub agree): {o['corr_evasion_vs_substitute_agreement']:+.3f}")
        print(f"    corr(evasion, movable)  : {o['corr_evasion_vs_movable_importance']:+.3f}")

    print("\n  Reading: a strong negative correlation with substitute agreement")
    print("  supports the attack-artefact explanation. A strong positive")
    print("  correlation with movable importance points at the victim instead.")
    print("=" * 74)

    with open(REPORT_PATH, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epsilon": EPSILON, "steps": STEPS, "restarts": RESTARTS,
            "seeds": SEEDS, "per_run": rows, "analysis": out,
        }, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
