"""
Full audit: 4 victim models x 3 attacks x 3 defenses.

Models   LinearSVC, RandomForest, XGBoost, MLP
Attacks  FGSM (single-step), BIM (iterative), PGD (iterative + random restarts)
Defenses none, feature squeezing, realizability filter, adversarial training

Tree models expose no input gradient, so they are attacked by transfer from a
surrogate. Everything is run both unconstrained and constrained so the two
evaluation regimes stay directly comparable.

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
SURROGATE = "MLP"          # non-linear surrogate transfers to trees better than a linear one
REPORT_PATH = "full_matrix_report.json"
RANDOM_STATE = 42


def evasion_of(model, X_adv, mask, filt=None):
    preds = filt.predict(model, X_adv) if filt else model.predict(X_adv)
    evaded = (preds == 0) & mask
    return metrics.evasion_rate(int(evaded.sum()), int(mask.sum()))


def detection_of(model, X, y, filt=None):
    preds = filt.predict(model, X) if filt else model.predict(X)
    y = np.asarray(y)
    tp = int(((preds == 1) & (y == 1)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    return {
        "DR": metrics.detection_rate(tp, fn),
        "FPR": metrics.false_positive_rate(fp, tn),
        "Balanced Accuracy": metrics.balanced_accuracy(tp, tn, fp, fn),
    }


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

    # A defense that flags real traffic as impossible is worse than no defense,
    # so verify the filter is silent on the untouched test set before trusting
    # any evasion number it produces.
    false_alarm = float(filt.flag(X_test_s).mean())
    print(f"Realizability filter false-alarm rate on clean traffic: {false_alarm:.4%}")
    if false_alarm > 0.001:
        print("  WARNING: filter is rejecting genuine flows - bounds are too tight")

    print("Training victims:")
    trained = victims.train_all(X_train_s, y_train)
    surrogate, surrogate_grad = trained[SURROGATE]
    print(f"  surrogate for gradient-free models: {SURROGATE}\n")

    ATTACKS = {
        "FGSM": lambda m, g, proj, msk, ev: attacks.fgsm(
            m, X_test_s, y_test, EPSILON, projector=proj, gradient_fn=g, mask=msk
        ),
        "BIM": lambda m, g, proj, msk, ev: attacks.bim(
            m, X_test_s, y_test, EPSILON, steps=STEPS, projector=proj,
            gradient_fn=g, mask=msk, eval_model=ev
        ),
        "PGD": lambda m, g, proj, msk, ev: attacks.pgd(
            m, X_test_s, y_test, EPSILON, steps=STEPS, restarts=RESTARTS,
            projector=proj, gradient_fn=g, mask=msk, eval_model=ev
        ),
    }

    results = {}

    for name, (model, grad_fn) in trained.items():
        direct = grad_fn is not None
        atk_model, atk_grad = (model, grad_fn) if direct else (surrogate, surrogate_grad)

        y_clean = model.predict(X_test_s)
        mask = (y_clean == 1) & (np.asarray(y_test) == 1)

        entry = {
            "clean": detection_of(model, X_test_s, y_test),
            "attack_route": "direct" if direct else f"transfer from {SURROGATE}",
            "n_attackable": int(mask.sum()),
            "attacks": {},
        }

        for atk_name, atk in ATTACKS.items():
            row = {}
            for regime, proj in (("unconstrained", None), ("constrained", projector)):
                X_adv, _ = atk(atk_model, atk_grad, proj, mask, model)

                undef = evasion_of(model, X_adv, mask)
                squeezed = evasion_of(model, defense.feature_squeeze(X_adv, SQUEEZE_BITS), mask)
                filtered = evasion_of(model, X_adv, mask, filt=filt)

                # Detection rate on the attacked set is the same story told from
                # the defender's side, and it is the figure the milestone deck
                # reports ("94.1% -> 1.7%"). Keep both: evasion answers "how
                # often did the attacker get through", DR answers "how much of
                # the IDS is left".
                post = detection_of(model, X_adv, y_test)

                row[regime] = {
                    "no_defense": undef,
                    "feature_squeezing": squeezed,
                    "realizability_filter": filtered,
                    "DR_after_attack": post["DR"],
                    "BalAcc_after_attack": post["Balanced Accuracy"],
                }
            entry["attacks"][atk_name] = row

        results[name] = entry
        print(f"  {name}: attacked ({entry['attack_route']})", flush=True)

    # ---------------- adversarial training ----------------
    # Retrain each model on constrained PGD examples drawn from the training
    # split, then re-run the strongest attack against the hardened model.
    print("\nAdversarial training (constrained PGD examples):")
    adv_train = {}
    for name, (model, grad_fn) in trained.items():
        direct = grad_fn is not None
        atk_model, atk_grad = (model, grad_fn) if direct else (surrogate, surrogate_grad)

        X_adv_tr, y_adv_tr = defense.make_adversarial_augmentation(
            atk_model, X_train_s, y_train, EPSILON, attacks.pgd,
            projector=projector, gradient_fn=atk_grad,
            steps=STEPS, restarts=1, eval_model=model,
        )
        hardened = defense.adversarially_train(
            victims.VICTIMS[name]["train"], X_train_s, y_train, X_adv_tr, y_adv_tr
        )

        y_h = hardened.predict(X_test_s)
        h_mask = (y_h == 1) & (np.asarray(y_test) == 1)

        factory = victims.VICTIMS[name]["gradient_fn"]
        entry = {
            "clean": detection_of(hardened, X_test_s, y_test),
            "baseline_evasion_under_constrained_pgd":
                results[name]["attacks"]["PGD"]["constrained"]["no_defense"],
            "n_adversarial_training_samples": int(len(y_adv_tr)),
        }

        if factory:
            # Differentiable victim: the attacker simply re-derives gradients
            # from the hardened model. Fully adaptive by construction.
            X_adv_h, _ = attacks.pgd(
                hardened, X_test_s, y_test, EPSILON, steps=STEPS, restarts=RESTARTS,
                projector=projector, gradient_fn=factory(hardened), mask=h_mask,
                eval_model=hardened,
            )
            entry["evasion_adaptive"] = evasion_of(hardened, X_adv_h, h_mask)
            entry["evasion_stale_surrogate"] = None
            entry["attacker"] = "white-box, gradients recomputed on hardened model"
        else:
            # Tree victim. Two attackers, and the difference is the whole point:
            #
            #   stale    - reuses the surrogate built against the ORIGINAL model.
            #              Flattering, and what most papers report.
            #   adaptive - trains a fresh surrogate to mimic the HARDENED model's
            #              own decisions (substitute-model attack, Papernot et al.
            #              2017), then attacks through that. This is what a real
            #              adversary facing a deployed hardened IDS would do.
            X_stale, _ = attacks.pgd(
                surrogate, X_test_s, y_test, EPSILON, steps=STEPS, restarts=RESTARTS,
                projector=projector, gradient_fn=surrogate_grad, mask=h_mask,
                eval_model=hardened,
            )
            entry["evasion_stale_surrogate"] = evasion_of(hardened, X_stale, h_mask)

            adaptive = victims.train_mlp(X_train_s, hardened.predict(X_train_s))
            agreement = float(
                (adaptive.predict(X_test_s) == hardened.predict(X_test_s)).mean()
            )
            X_adapt, _ = attacks.pgd(
                adaptive, X_test_s, y_test, EPSILON, steps=STEPS, restarts=RESTARTS,
                projector=projector, gradient_fn=attacks.mlp_gradient_fn(adaptive),
                mask=h_mask, eval_model=hardened,
            )
            entry["evasion_adaptive"] = evasion_of(hardened, X_adapt, h_mask)
            entry["surrogate_agreement_with_victim"] = agreement
            entry["attacker"] = "black-box, substitute retrained against hardened model"

        adv_train[name] = entry
        print(f"  {name} hardened + re-attacked adaptively", flush=True)

    # ---------------- report ----------------
    print("\n" + "=" * 78)
    print("CLEAN PERFORMANCE")
    print(f"  {'model':<14} {'DR':>9} {'FPR':>9} {'BalAcc':>9}  route")
    for n, e in results.items():
        c = e["clean"]
        print(f"  {n:<14} {c['DR']:>9.4f} {c['FPR']:>9.4f} "
              f"{c['Balanced Accuracy']:>9.4f}  {e['attack_route']}")

    print("\nEVASION RATE, NO DEFENSE")
    print(f"  {'model':<14} {'attack':<6} {'unconstrained':>14} {'constrained':>13} {'overstated':>12}")
    for n, e in results.items():
        for a, r in e["attacks"].items():
            u = r["unconstrained"]["no_defense"]
            c = r["constrained"]["no_defense"]
            f = (u / c) if c > 0 else float("inf")
            print(f"  {n:<14} {a:<6} {u:>14.4f} {c:>13.4f} {f:>11.1f}x")

    print("\nDETECTION RATE: CLEAN -> AFTER ATTACK (no defense)")
    print(f"  {'model':<14} {'attack':<6} {'clean DR':>9} {'uncon. DR':>10} {'constr. DR':>11}")
    for n, e in results.items():
        for a, r in e["attacks"].items():
            print(f"  {n:<14} {a:<6} {e['clean']['DR']:>9.4f} "
                  f"{r['unconstrained']['DR_after_attack']:>10.4f} "
                  f"{r['constrained']['DR_after_attack']:>11.4f}")

    print("\nDEFENSE COMPARISON (vs constrained PGD, the strongest realistic attack)")
    print(f"  {'model':<14} {'none':>9} {'squeeze':>9} {'filter':>9} {'adv-train':>11}")
    for n, e in results.items():
        c = e["attacks"]["PGD"]["constrained"]
        print(f"  {n:<14} {c['no_defense']:>9.4f} {c['feature_squeezing']:>9.4f} "
              f"{c['realizability_filter']:>9.4f} "
              f"{adv_train[n]['evasion_adaptive']:>11.4f}")

    print("\nADVERSARIAL TRAINING vs ADAPTIVE ATTACKER")
    print("  (stale = attacker reuses the old surrogate; adaptive = attacker rebuilds it)")
    print(f"  {'model':<14} {'stale':>9} {'adaptive':>10}  attacker")
    for n, a in adv_train.items():
        stale = f"{a['evasion_stale_surrogate']:.4f}" if a["evasion_stale_surrogate"] is not None else "n/a"
        print(f"  {n:<14} {stale:>9} {a['evasion_adaptive']:>10.4f}  {a['attacker']}")

    print("\nREALIZABILITY FILTER vs UNCONSTRAINED ATTACKS (should neutralise them)")
    print(f"  {'model':<14} {'attack':<6} {'no defense':>12} {'+ filter':>10}")
    for n, e in results.items():
        for a, r in e["attacks"].items():
            u = r["unconstrained"]
            print(f"  {n:<14} {a:<6} {u['no_defense']:>12.4f} {u['realizability_filter']:>10.4f}")

    print("\nADVERSARIAL TRAINING - CLEAN PERFORMANCE COST")
    print(f"  {'model':<14} {'DR before':>11} {'DR after':>10} {'FPR before':>12} {'FPR after':>11}")
    for n, e in results.items():
        b, a = e["clean"], adv_train[n]["clean"]
        print(f"  {n:<14} {b['DR']:>11.4f} {a['DR']:>10.4f} "
              f"{b['FPR']:>12.4f} {a['FPR']:>11.4f}")
    print("=" * 78)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "epsilon": EPSILON,
        "steps": STEPS,
        "restarts": RESTARTS,
        "squeeze_bits": SQUEEZE_BITS,
        "surrogate": SURROGATE,
        "results": results,
        "adversarial_training": adv_train,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
