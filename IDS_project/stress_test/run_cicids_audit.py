"""
The same audit, on CICIDS2017.

Nothing here is dataset-specific except which loader and which specification
are named. That is the point: if adding a dataset needed engine changes, the
framework would be a study of NSL-KDD with extra steps.

The prediction this tests: the overstatement factor should be LARGER on
CICIDS2017 than on NSL-KDD. CICIDS has 78 features against 40, and far more of
them are statistics of each other - packet-length and inter-arrival min/mean/max
triples, per-second rates, variances. An unconstrained attack moves those
independently and will emit flows whose smallest packet exceeds their largest.
NSL-KDD had almost no such structure to violate.

If the factor comes out SMALLER, that is the more interesting result and needs
explaining rather than burying.

Run with: python -m stress_test.run_cicids_audit
(from the IDS_project directory, with the CSVs in ../data/cicids2017/)
"""

import json
import sys
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from stress_test import (
    preprocessing, attacks, constraints, metrics, victims,
)
from stress_test.spec import FeatureSpec

EPSILON = 0.3
STEPS = 20
RESTARTS = 2
SEEDS = [42, 7, 1337]
SAMPLE = 200_000       # CICIDS2017 is ~2.8M rows; subsample to keep runs tractable
REPORT_PATH = "cicids_audit_report.json"


def main():
    try:
        df = preprocessing.load_cicids2017()
    except FileNotFoundError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    print(f"loaded {len(df):,} flows, {df.shape[1]} columns")
    df = preprocessing.clean_data(df)
    X, y = preprocessing.encode_and_split_cicids(df)
    print(f"after cleaning: {X.shape[1]} features, "
          f"{y.mean():.1%} attack / {1 - y.mean():.1%} benign")

    spec = FeatureSpec.load("cicids2017")
    undeclared, unused = spec.validate(X.columns)
    print(f"\nspec: {spec}")
    if undeclared:
        # These silently default to immutable, which hands the attacker less
        # than they have and flatters the constrained numbers. Loud on purpose.
        print(f"  WARNING: {len(undeclared)} column(s) not in the spec, treated "
              f"as immutable: {undeclared[:6]}{' ...' if len(undeclared) > 6 else ''}")
    if unused:
        print(f"  note: {len(unused)} spec entry(ies) absent from the data "
              f"(likely dropped as constant): {unused[:6]}{' ...' if len(unused) > 6 else ''}")

    if SAMPLE and len(X) > SAMPLE:
        idx = np.random.RandomState(0).choice(len(X), SAMPLE, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
        print(f"\nsubsampled to {len(X):,} flows for tractability")

    names = list(X.columns)
    per_seed = {}

    for seed in SEEDS:
        print(f"\nseed {seed} ...", flush=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        X_trs, X_tes, scaler = preprocessing.scale_features(X_tr, X_te)
        projector = constraints.ConstraintProjector(
            names, scaler, spec=spec).fit(X_tr)

        victims.RANDOM_STATE = seed
        model = victims.train_linear_svc(X_trs, y_tr)
        y_clean = model.predict(X_tes)
        mask = (y_clean == 1) & (np.asarray(y_te) == 1)
        grad = attacks.linear_gradient_fn(model)

        row = {"clean_DR": metrics.detection_rate(
            int(((y_clean == 1) & (y_te == 1)).sum()),
            int(((y_clean == 0) & (y_te == 1)).sum()))}

        for regime, proj in (("unconstrained", None), ("constrained", projector)):
            X_adv = attacks.pgd(
                model, X_tes, y_te, EPSILON, steps=STEPS, restarts=RESTARTS,
                projector=proj, gradient_fn=grad, mask=mask, seed=seed,
            )[0]
            preds = model.predict(X_adv)
            row[f"{regime}_evasion"] = metrics.evasion_rate(
                int(((preds == 0) & mask).sum()), int(mask.sum()))
            row[f"{regime}_DR"] = metrics.detection_rate(
                int(((preds == 1) & (y_te == 1)).sum()),
                int(((preds == 0) & (y_te == 1)).sum()))
            row[f"{regime}_violations"] = constraints.realizability_violations(
                scaler.inverse_transform(X_adv[mask]), names, projector)

        c = row["constrained_evasion"]
        row["overstatement"] = (row["unconstrained_evasion"] / c) if c > 0 else float("nan")
        per_seed[seed] = row
        print(f"  evasion  uncon={row['unconstrained_evasion']:.4f}  "
              f"con={row['constrained_evasion']:.4f}  "
              f"overstated={row['overstatement']:.1f}x")
        print(f"  violations uncon: {row['unconstrained_violations']}")
        print(f"  violations con  : {row['constrained_violations']}")

    # ---------------- summary ----------------
    def agg(field):
        v = np.array([per_seed[s][field] for s in SEEDS], dtype=float)
        return float(np.nanmean(v)), float(np.nanstd(v, ddof=1) if len(v) > 1 else 0.0)

    bar = "=" * 74
    print("\n" + bar)
    print(f"CICIDS2017 over {len(SEEDS)} seeds (mean +/- std)")
    for f in ("clean_DR", "unconstrained_evasion", "constrained_evasion",
              "constrained_DR", "overstatement"):
        m, sd = agg(f)
        print(f"  {f:<24} {m:>8.4f} +/- {sd:.4f}")
    print("\n  NSL-KDD reference: 21.9 +/- 2.8x overstatement (LinearSVC, 5 seeds)")
    print(bar)

    with open(REPORT_PATH, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": "cicids2017", "spec": spec.name,
            "epsilon": EPSILON, "steps": STEPS, "restarts": RESTARTS,
            "seeds": SEEDS, "n_features": len(names), "n_flows": int(len(X)),
            "undeclared_columns": undeclared,
            "per_seed": {str(k): v for k, v in per_seed.items()},
            "aggregate": {f: dict(zip(("mean", "std"), agg(f)))
                          for f in ("clean_DR", "unconstrained_evasion",
                                    "constrained_evasion", "constrained_DR",
                                    "overstatement")},
        }, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
