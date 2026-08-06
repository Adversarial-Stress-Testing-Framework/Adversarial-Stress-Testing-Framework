"""
Repeat the headline measurement across several random seeds.

Every other runner uses random_state=42 once, so every figure the project
reports is a single sample with no indication of spread. A reviewer cannot tell
whether an 18.7x overstatement is a stable property or one lucky split.

This varies the seed everywhere it matters - the train/test split, victim
initialisation, substitute initialisation, and PGD's random start - and reports
mean +/- standard deviation. Attacks run black-box through a per-victim
substitute so all models stay comparable.

Run with: python -m stress_test.run_multiseed [n_seeds]
(from the IDS_project directory, so KDDTrain+.txt is found)
"""

import json
import sys
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from stress_test import preprocessing, attacks, constraints, metrics, victims


EPSILON = 0.3
STEPS = 20
RESTARTS = 2
SEEDS = [42, 7, 1337, 2024, 99]
REPORT_PATH = "multiseed_report.json"


def one_seed(X, y, feature_names, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train_s, X_test_s, scaler = preprocessing.scale_features(X_train, X_test)
    projector = constraints.ConstraintProjector(feature_names, scaler).fit(X_train)

    victims.RANDOM_STATE = seed  # victim + substitute initialisation
    trained = victims.train_all(X_train_s, y_train, verbose=False)

    out = {}
    for name, (model, _) in trained.items():
        y_clean = model.predict(X_test_s)
        mask = (y_clean == 1) & (np.asarray(y_test) == 1)
        tp, fn, fp, tn = (
            int(((y_clean == 1) & (y_test == 1)).sum()),
            int(((y_clean == 0) & (y_test == 1)).sum()),
            int(((y_clean == 1) & (y_test == 0)).sum()),
            int(((y_clean == 0) & (y_test == 0)).sum()),
        )

        sub = victims.train_substitute(X_train_s, model, seed=seed)
        grad = attacks.mlp_gradient_fn(sub)

        row = {"clean_DR": metrics.detection_rate(tp, fn),
               "clean_FPR": metrics.false_positive_rate(fp, tn)}

        for regime, proj in (("unconstrained", None), ("constrained", projector)):
            X_adv = attacks.pgd(
                sub, X_test_s, y_test, EPSILON, steps=STEPS, restarts=RESTARTS,
                projector=proj, gradient_fn=grad, mask=mask, eval_model=model,
                seed=seed,
            )[0]
            preds = model.predict(X_adv)
            a_tp = int(((preds == 1) & (y_test == 1)).sum())
            a_fn = int(((preds == 0) & (y_test == 1)).sum())
            row[f"{regime}_evasion"] = metrics.evasion_rate(
                int(((preds == 0) & mask).sum()), int(mask.sum()))
            row[f"{regime}_DR"] = metrics.detection_rate(a_tp, a_fn)

        c = row["constrained_evasion"]
        row["overstatement"] = (row["unconstrained_evasion"] / c) if c > 0 else float("nan")
        out[name] = row
    return out


def main():
    seeds = SEEDS
    if len(sys.argv) > 1:
        seeds = SEEDS[: max(2, int(sys.argv[1]))]

    df = preprocessing.clean_data(preprocessing.load_dataset("KDDTrain+.txt"))
    X, y = preprocessing.encode_and_split(df)
    feature_names = list(X.columns)

    per_seed = {}
    for i, seed in enumerate(seeds, 1):
        print(f"seed {seed}  ({i}/{len(seeds)}) ...", flush=True)
        per_seed[seed] = one_seed(X, y, feature_names, seed)

    models = list(per_seed[seeds[0]])
    fields = ["clean_DR", "clean_FPR", "unconstrained_evasion",
              "constrained_evasion", "unconstrained_DR", "constrained_DR",
              "overstatement"]

    agg = {}
    for m in models:
        agg[m] = {}
        for f in fields:
            vals = np.array([per_seed[s][m][f] for s in seeds], dtype=float)
            agg[m][f] = {"mean": float(np.nanmean(vals)),
                         "std": float(np.nanstd(vals, ddof=1)),
                         "values": vals.tolist()}

    def cell(m, f, pct=True):
        a = agg[m][f]
        return (f"{a['mean']:.2%} ± {a['std']:.2%}" if pct
                else f"{a['mean']:.1f} ± {a['std']:.1f}")

    print("\n" + "=" * 78)
    print(f"HEADLINE METRICS OVER {len(seeds)} SEEDS  (mean ± std)")
    print(f"\n  {'model':<14} {'clean DR':>18} {'clean FPR':>18}")
    for m in models:
        print(f"  {m:<14} {cell(m, 'clean_DR'):>18} {cell(m, 'clean_FPR'):>18}")

    print(f"\n  {'model':<14} {'evasion uncon.':>18} {'evasion constr.':>18} {'overstated':>16}")
    for m in models:
        print(f"  {m:<14} {cell(m, 'unconstrained_evasion'):>18} "
              f"{cell(m, 'constrained_evasion'):>18} "
              f"{cell(m, 'overstatement', pct=False) + 'x':>16}")

    print(f"\n  {'model':<14} {'DR uncon. attack':>18} {'DR constr. attack':>18}")
    for m in models:
        print(f"  {m:<14} {cell(m, 'unconstrained_DR'):>18} {cell(m, 'constrained_DR'):>18}")
    print("=" * 78)

    with open(REPORT_PATH, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seeds": seeds, "epsilon": EPSILON, "steps": STEPS,
            "restarts": RESTARTS, "attack_route": "black-box substitute",
            "aggregate": agg, "per_seed": {str(k): v for k, v in per_seed.items()},
        }, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
