"""
Full audit: 4 victim models x 3 attacks x 3 defenses, constrained and not.

Every victim is attacked through TWO routes:

  black-box  the adversary queries the victim, fits a differentiable substitute
             to its outputs, and attacks through that. Available for every
             model, including the trees, so all four are directly comparable.

  white-box  the adversary differentiates the victim itself. Only possible for
             LinearSVC and the MLP; trees are piecewise-constant and expose no
             usable input gradient.

The two routes are reported separately on purpose. An earlier version attacked
differentiable victims white-box and trees by transfer, then compared the
results - which confounds model robustness with attack strength. Any claim
about one model being more robust than another must come from the black-box
column, where the attack is identical.

Run with: python -m stress_test.run_full_matrix
(from the IDS_project directory, so KDDTrain+.txt is found)
"""

import json
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from stress_test import (
    preprocessing, attacks, constraints, defense, metrics, victims,
)


EPSILON = 0.3
STEPS = 20
RESTARTS = 2
SQUEEZE_BITS = 4
REPORT_PATH = "full_matrix_report.json"
RANDOM_STATE = 42


def confusion(preds, y):
    y = np.asarray(y)
    return (
        int(((preds == 1) & (y == 1)).sum()), int(((preds == 0) & (y == 1)).sum()),
        int(((preds == 1) & (y == 0)).sum()), int(((preds == 0) & (y == 0)).sum()),
    )


def detection_of(model, X, y, filt=None):
    preds = filt.predict(model, X) if filt else model.predict(X)
    tp, fn, fp, tn = confusion(preds, y)
    return {
        "DR": metrics.detection_rate(tp, fn),
        "FPR": metrics.false_positive_rate(fp, tn),
        "Balanced Accuracy": metrics.balanced_accuracy(tp, tn, fp, fn),
    }


def evasion_of(model, X_adv, mask, filt=None):
    preds = filt.predict(model, X_adv) if filt else model.predict(X_adv)
    return metrics.evasion_rate(int(((preds == 0) & mask).sum()), int(mask.sum()))


def main():
    df = preprocessing.clean_data(preprocessing.load_dataset("KDDTrain+.txt"))
    X, y = preprocessing.encode_and_split(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_s, X_test_s, scaler = preprocessing.scale_features(X_train, X_test)

    projector = constraints.ConstraintProjector(feature_names, scaler).fit(X_train)
    filt = defense.RealizabilityFilter(projector)

    false_alarm = float(filt.flag(X_test_s).mean())
    print(f"Realizability filter false-alarm rate on clean traffic: {false_alarm:.4%}")
    if false_alarm > 0.001:
        print("  WARNING: filter is rejecting genuine flows - bounds are too tight")

    print("\nTraining victims:")
    trained = victims.train_all(X_train_s, y_train)

    def run_attack(name, atk_model, grad_fn, proj, mask, victim):
        """One attack against one victim through one route."""
        kw = dict(projector=proj, gradient_fn=grad_fn, mask=mask)
        if name == "FGSM":
            return attacks.fgsm(atk_model, X_test_s, y_test, EPSILON, **kw)[0]
        if name == "BIM":
            return attacks.bim(atk_model, X_test_s, y_test, EPSILON, steps=STEPS,
                               eval_model=victim, **kw)[0]
        return attacks.pgd(atk_model, X_test_s, y_test, EPSILON, steps=STEPS,
                           restarts=RESTARTS, eval_model=victim, **kw)[0]

    results = {}

    for name, (model, grad_fn) in trained.items():
        y_clean = model.predict(X_test_s)
        mask = (y_clean == 1) & (np.asarray(y_test) == 1)

        print(f"\n  {name}: fitting substitute ...", flush=True)
        sub = victims.train_substitute(X_train_s, model)
        sub_grad = attacks.mlp_gradient_fn(sub)
        agreement = float((sub.predict(X_test_s) == y_clean).mean())
        print(f"    substitute agrees with victim on {agreement:.2%} of test flows")

        entry = {
            "clean": detection_of(model, X_test_s, y_test),
            "n_attackable": int(mask.sum()),
            "substitute_agreement": agreement,
            "white_box_available": grad_fn is not None,
            "attacks": {},
        }

        for atk_name in ("FGSM", "BIM", "PGD"):
            row = {}
            for route, (am, gf) in (
                ("black_box", (sub, sub_grad)),
                ("white_box", (model, grad_fn) if grad_fn else (None, None)),
            ):
                if am is None:
                    row[route] = None
                    continue
                per_regime = {}
                for regime, proj in (("unconstrained", None), ("constrained", projector)):
                    X_adv = run_attack(atk_name, am, gf, proj, mask, model)
                    post = detection_of(model, X_adv, y_test)
                    per_regime[regime] = {
                        "no_defense": evasion_of(model, X_adv, mask),
                        "feature_squeezing": evasion_of(
                            model, defense.feature_squeeze(X_adv, SQUEEZE_BITS), mask),
                        "realizability_filter": evasion_of(model, X_adv, mask, filt=filt),
                        "DR_after_attack": post["DR"],
                        "BalAcc_after_attack": post["Balanced Accuracy"],
                    }
                row[route] = per_regime
            entry["attacks"][atk_name] = row

        results[name] = entry
        print(f"    attacked (black-box{' + white-box' if grad_fn else ''})", flush=True)

    # ---------------- adversarial training ----------------
    print("\nAdversarial training on constrained PGD examples:")
    adv_train = {}
    for name, (model, grad_fn) in trained.items():
        sub = victims.train_substitute(X_train_s, model)
        X_adv_tr, y_adv_tr = defense.make_adversarial_augmentation(
            sub, X_train_s, y_train, EPSILON, attacks.pgd,
            projector=projector, gradient_fn=attacks.mlp_gradient_fn(sub),
            steps=STEPS, restarts=1, eval_model=model,
        )
        hardened = defense.adversarially_train(
            victims.VICTIMS[name]["train"], X_train_s, y_train, X_adv_tr, y_adv_tr
        )

        y_h = hardened.predict(X_test_s)
        h_mask = (y_h == 1) & (np.asarray(y_test) == 1)

        # Stale: the adversary reuses the substitute built against the ORIGINAL
        # model. Flattering, and what most papers report.
        X_stale = attacks.pgd(
            sub, X_test_s, y_test, EPSILON, steps=STEPS, restarts=RESTARTS,
            projector=projector, gradient_fn=attacks.mlp_gradient_fn(sub),
            mask=h_mask, eval_model=hardened)[0]

        # Adaptive: the adversary re-queries the HARDENED model and rebuilds the
        # substitute against it. This is what someone facing a deployed
        # defended IDS would actually do.
        sub_h = victims.train_substitute(X_train_s, hardened)
        X_adapt = attacks.pgd(
            sub_h, X_test_s, y_test, EPSILON, steps=STEPS, restarts=RESTARTS,
            projector=projector, gradient_fn=attacks.mlp_gradient_fn(sub_h),
            mask=h_mask, eval_model=hardened)[0]

        adv_train[name] = {
            "clean": detection_of(hardened, X_test_s, y_test),
            "evasion_stale_surrogate": evasion_of(hardened, X_stale, h_mask),
            "evasion_adaptive": evasion_of(hardened, X_adapt, h_mask),
            "baseline_evasion":
                results[name]["attacks"]["PGD"]["black_box"]["constrained"]["no_defense"],
            "n_adversarial_training_samples": int(len(y_adv_tr)),
        }
        print(f"  {name} hardened and re-attacked adaptively", flush=True)

    # ---------------- report ----------------
    bar = "=" * 78

    print("\n" + bar)
    print("CLEAN PERFORMANCE")
    print(f"  {'model':<14} {'DR':>9} {'FPR':>9} {'BalAcc':>9} {'substitute fit':>15}")
    for n, e in results.items():
        c = e["clean"]
        print(f"  {n:<14} {c['DR']:>9.4f} {c['FPR']:>9.4f} "
              f"{c['Balanced Accuracy']:>9.4f} {e['substitute_agreement']:>15.4f}")

    print("\nEVASION RATE - BLACK BOX (identical attack for every model)")
    print(f"  {'model':<14} {'attack':<6} {'unconstrained':>14} {'constrained':>13} {'overstated':>12}")
    for n, e in results.items():
        for a, r in e["attacks"].items():
            u = r["black_box"]["unconstrained"]["no_defense"]
            c = r["black_box"]["constrained"]["no_defense"]
            f = f"{u / c:>11.1f}x" if c > 0 else f"{'inf':>12}"
            print(f"  {n:<14} {a:<6} {u:>14.4f} {c:>13.4f} {f}")

    print("\nEVASION RATE - WHITE BOX (differentiable victims only)")
    for n, e in results.items():
        for a, r in e["attacks"].items():
            if r["white_box"] is None:
                continue
            u = r["white_box"]["unconstrained"]["no_defense"]
            c = r["white_box"]["constrained"]["no_defense"]
            f = f"{u / c:>11.1f}x" if c > 0 else f"{'inf':>12}"
            print(f"  {n:<14} {a:<6} {u:>14.4f} {c:>13.4f} {f}")

    print("\nROBUSTNESS RANKING - black-box constrained PGD, like for like")
    print(f"  {'model':<14} {'clean DR':>10} {'DR under attack':>17} {'points lost':>13}")
    ranking = sorted(
        (
            (n,
             e["clean"]["DR"],
             e["attacks"]["PGD"]["black_box"]["constrained"]["DR_after_attack"])
            for n, e in results.items()
        ),
        key=lambda r: r[1] - r[2], reverse=True,
    )
    for n, clean_dr, atk_dr in ranking:
        print(f"  {n:<14} {clean_dr:>10.4f} {atk_dr:>17.4f} {clean_dr - atk_dr:>13.4f}")

    print("\nDEFENSES vs black-box constrained PGD")
    print(f"  {'model':<14} {'none':>9} {'squeeze':>9} {'filter':>9} {'adv-train':>11}")
    for n, e in results.items():
        c = e["attacks"]["PGD"]["black_box"]["constrained"]
        print(f"  {n:<14} {c['no_defense']:>9.4f} {c['feature_squeezing']:>9.4f} "
              f"{c['realizability_filter']:>9.4f} "
              f"{adv_train[n]['evasion_adaptive']:>11.4f}")

    print("\nADVERSARIAL TRAINING vs ADAPTIVE ATTACKER")
    print(f"  {'model':<14} {'undefended':>11} {'stale sub.':>11} {'adaptive':>10}")
    for n, a in adv_train.items():
        print(f"  {n:<14} {a['baseline_evasion']:>11.4f} "
              f"{a['evasion_stale_surrogate']:>11.4f} {a['evasion_adaptive']:>10.4f}")

    print("\nADVERSARIAL TRAINING - CLEAN PERFORMANCE COST")
    print(f"  {'model':<14} {'DR before':>11} {'DR after':>10} {'FPR before':>12} {'FPR after':>11}")
    for n, e in results.items():
        b, a = e["clean"], adv_train[n]["clean"]
        print(f"  {n:<14} {b['DR']:>11.4f} {a['DR']:>10.4f} "
              f"{b['FPR']:>12.4f} {a['FPR']:>11.4f}")
    print(bar)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "epsilon": EPSILON,
        "steps": STEPS,
        "restarts": RESTARTS,
        "squeeze_bits": SQUEEZE_BITS,
        "attack_route": "per-victim substitute model (uniform black-box) + white-box where differentiable",
        "results": results,
        "adversarial_training": adv_train,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
