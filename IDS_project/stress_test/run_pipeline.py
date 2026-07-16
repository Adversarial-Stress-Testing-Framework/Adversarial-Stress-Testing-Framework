"""
Run with: python -m stress_test.run_pipeline
(from the IDS_project/IDS_project directory, so KDDTrain+.txt is found)
"""

import json
from datetime import datetime

from sklearn.model_selection import train_test_split

from stress_test import preprocessing, baseline_model, fgsm_attack, defense, metrics


EPSILON = 0.3
SQUEEZE_BITS = 4
REPORT_PATH = "stress_test_report.json"


def compute_report(tp, fn, fp, tn):
    return {
        "DR (Attack Recall)": metrics.detection_rate(tp, fn),
        "Normal Recall": metrics.normal_recall(tn, fp),
        "FPR": metrics.false_positive_rate(fp, tn),
        "Accuracy": metrics.accuracy(tp, tn, fp, fn),
        "Balanced Accuracy": metrics.balanced_accuracy(tp, tn, fp, fn),
    }


def print_report(title, report):
    print(f"\n=== {title} ===")
    print("  ".join(f"{k}: {v:.4f}" for k, v in report.items()))


def print_prediction_distribution(title, y_pred):
    attack_frac = (y_pred == 1).mean()
    print(f"  [{title}] predicted Attack: {attack_frac:.4f}  predicted Normal: {1 - attack_frac:.4f}")


def prediction_distribution(y_pred):
    attack_frac = float((y_pred == 1).mean())
    return {"predicted_attack": attack_frac, "predicted_normal": 1 - attack_frac}


def save_report(path, sections):
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "epsilon": EPSILON,
        "squeeze_bits": SQUEEZE_BITS,
        "sections": sections,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nReport saved to {path}")


def main():
    df = preprocessing.load_dataset("KDDTrain+.txt")
    df = preprocessing.clean_data(df)
    X, y = preprocessing.encode_and_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(X_train, X_test)

    model = baseline_model.train_victim_model(X_train_scaled, y_train)

    baseline = baseline_model.evaluate_baseline(model, X_test_scaled, y_test)
    baseline_report = compute_report(baseline["tp"], baseline["fn"], baseline["fp"], baseline["tn"])
    print_report("BASELINE (clean)", baseline_report)
    print_prediction_distribution("clean baseline", baseline["y_pred"])
    baseline_report["prediction_distribution"] = prediction_distribution(baseline["y_pred"])

    X_adv, attacked_mask = fgsm_attack.fgsm_perturb(model, X_test_scaled, y_test.to_numpy(), EPSILON)

    with metrics.LatencyTimer() as timer:
        y_pred_adv = model.predict(X_adv)
    latency_no_defense = timer.elapsed_ms / len(X_adv)

    tp = int(((y_pred_adv == 1) & (y_test == 1)).sum())
    fn = int(((y_pred_adv == 0) & (y_test == 1)).sum())
    fp = int(((y_pred_adv == 1) & (y_test == 0)).sum())
    tn = int(((y_pred_adv == 0) & (y_test == 0)).sum())

    adv_report = compute_report(tp, fn, fp, tn)
    fn_adv_evaded = int(((y_pred_adv == 0) & attacked_mask).sum())
    total_adv_samples = int(attacked_mask.sum())
    evaded_mask = (y_pred_adv == 0) & attacked_mask

    adv_report["Evasion Rate"] = metrics.evasion_rate(fn_adv_evaded, total_adv_samples)
    adv_report["Robustness Score"] = metrics.robustness_score(baseline_report["Accuracy"], adv_report["Accuracy"])
    adv_report["MPD"] = metrics.min_perturbation_distance(X_test_scaled, X_adv, evaded_mask)
    adv_report["Latency (ms/sample)"] = latency_no_defense
    print_report(f"ADVERSARIAL (epsilon={EPSILON}, no defense)", adv_report)
    print_prediction_distribution("no defense", y_pred_adv)
    adv_report["prediction_distribution"] = prediction_distribution(y_pred_adv)

    X_adv_squeezed = defense.feature_squeeze(X_adv, bits=SQUEEZE_BITS)

    with metrics.LatencyTimer() as timer:
        y_pred_defended = model.predict(X_adv_squeezed)
    latency_defense = timer.elapsed_ms / len(X_adv_squeezed)

    tp_d = int(((y_pred_defended == 1) & (y_test == 1)).sum())
    fn_d = int(((y_pred_defended == 0) & (y_test == 1)).sum())
    fp_d = int(((y_pred_defended == 1) & (y_test == 0)).sum())
    tn_d = int(((y_pred_defended == 0) & (y_test == 0)).sum())

    defended_report = compute_report(tp_d, fn_d, fp_d, tn_d)
    fn_defended_evaded = int(((y_pred_defended == 0) & attacked_mask).sum())
    evaded_mask_defended = (y_pred_defended == 0) & attacked_mask

    defended_report["Evasion Rate"] = metrics.evasion_rate(fn_defended_evaded, total_adv_samples)
    defended_report["Robustness Score"] = metrics.robustness_score(baseline_report["Accuracy"], defended_report["Accuracy"])
    defended_report["MPD"] = metrics.min_perturbation_distance(X_test_scaled, X_adv_squeezed, evaded_mask_defended)
    defended_report["Latency (ms/sample)"] = latency_defense

    print(f"\n=== FEATURE SQUEEZING (epsilon={EPSILON}, bits={SQUEEZE_BITS}) - DIAGNOSED FAILURE MODE, NOT A FIX ===")
    print_prediction_distribution("no defense", y_pred_adv)
    print_prediction_distribution("+ feature squeezing", y_pred_defended)
    print(
        "  Squeezing pushes predictions further toward the majority ('normal') class rather than "
        "recovering attack detection: DR barely moves while FPR drops only because the model predicts "
        "'attack' even less often. This is class-collapse, not robustness. See Balanced Accuracy vs. "
        "Accuracy below to see the raw accuracy figure is not trustworthy on its own on this imbalanced data."
    )
    print_report("metrics", defended_report)
    defended_report["prediction_distribution"] = prediction_distribution(y_pred_defended)
    defended_report["diagnosis"] = (
        "Feature squeezing does not defend against this static (non-adaptive) FGSM attack. "
        "Predictions collapse toward the majority 'normal' class rather than recovering true detection "
        "capability; apparent Accuracy/FPR improvement is an artifact of that collapse, not robustness."
    )
    defended_report["next_step"] = (
        "Evaluate against an adaptive attacker that accounts for the squeeze function when generating "
        "perturbations (e.g. straight-through gradient approximation through the rounding operation). "
        "This is out of scope for this 30-40% milestone and targeted for the next milestone."
    )

    save_report(REPORT_PATH, {
        "baseline_clean": baseline_report,
        "adversarial_no_defense": adv_report,
        "adversarial_with_feature_squeezing": defended_report,
    })


if __name__ == "__main__":
    main()
