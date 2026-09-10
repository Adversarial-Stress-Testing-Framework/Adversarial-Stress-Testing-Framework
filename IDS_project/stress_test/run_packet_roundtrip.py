"""
Phase 3: does a constrained adversarial perturbation survive being realised as
actual packets?

Every result so far lives in feature space. A constrained attack produces a
vector that could correspond to a real flow, but nothing has checked that a real
flow can be made to produce that vector. This closes the loop:

    real packets -> perturb the packets -> re-extract features -> re-test

Running it forward rather than backwards is deliberate. Synthesising a capture
that yields a chosen feature vector is an inverse problem with no general
solution; applying operations an attacker can actually perform and measuring
what comes out is tractable and is what an attacker would really do.

The operations are the ones the constraint taxonomy already calls
increase-only, now applied to packets instead of numbers:

    padding   append bytes to forward payloads
    delay     insert inter-packet gaps on the forward direction

Three rungs get measured on the same flows:

    unconstrained   feature-space attack, free to emit impossible values
    constrained     feature-space attack, restricted to realizable vectors
    realized        packet-level, features re-derived from modified traffic

The gap between the second and third is the number that does not exist in the
literature.

Run with: python -m stress_test.run_packet_roundtrip
"""

import copy
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from stress_test import (
    preprocessing, attacks, constraints, flow_features, metrics, victims,
)
from stress_test.spec import FeatureSpec

FLOWS = "../data/cicids2017/attack_flows.pkl"
TUESDAY = "../data/cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv"
EPSILON = 0.3
STEPS = 20
RESTARTS = 2
SEED = 42
SAMPLE = 150_000
REPORT = "packet_roundtrip_report.json"

# Attacker budgets, in the units an operator would actually think in.
BUDGETS = [
    ("none", 0, 0.0),
    ("pad 64B", 64, 0.0),
    ("pad 256B", 256, 0.0),
    ("delay 100ms", 0, 0.100),
    ("pad 256B + delay 100ms", 256, 0.100),
    ("pad 1024B + delay 500ms", 1024, 0.500),
]


def perturb(packets, pad_bytes, delay_s):
    """Apply attacker-performable packet operations to one flow.

    Only forward packets are touched: the attacker controls their own side of
    the conversation, not the server's responses. Padding grows the payload,
    which is what appending junk to a request does. Delay accumulates, because
    stalling packet three also postpones everything after it.
    """
    out = copy.deepcopy(packets)
    shift = 0.0
    for p in sorted(out, key=lambda q: q["t"]):
        if p["fwd"]:
            if pad_bytes and p["payload"] > 0:
                p["payload"] += pad_bytes
                p["len"] += pad_bytes
            shift += delay_s
        p["t"] += shift
    return out


def main():
    # ---------------- victim model, trained on published features ----------
    print("training victim on CICIDS2017 ...", flush=True)
    df = preprocessing.clean_data(preprocessing.load_cicids2017())
    X, y = preprocessing.encode_and_split_cicids(df)
    idx = np.random.RandomState(0).choice(len(X), min(SAMPLE, len(X)), replace=False)
    X, y = X.iloc[idx], y.iloc[idx]
    names = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_trs, X_tes, scaler = preprocessing.scale_features(X_tr, X_te)
    spec = FeatureSpec.load("cicids2017")
    projector = constraints.ConstraintProjector(names, scaler, spec=spec).fit(X_tr)

    victims.RANDOM_STATE = SEED
    model = victims.train_linear_svc(X_trs, y_tr)
    grad = attacks.linear_gradient_fn(model)
    print(f"  clean detection rate: "
          f"{metrics.detection_rate(int(((model.predict(X_tes) == 1) & (y_te == 1)).sum()), int(((model.predict(X_tes) == 0) & (y_te == 1)).sum())):.4f}")

    # ---------------- our extractor, checked against the published CSV -----
    with open(FLOWS, "rb") as fh:
        raw = pickle.load(fh)
    extracted = [flow_features.extract_features(v) for v in raw.values()]
    print(f"\nextracted {len(extracted)} attack flows from the capture")

    pub = pd.read_csv(TUESDAY, low_memory=False)
    pub.columns = [c.strip() for c in pub.columns]
    ftp = pub[pub["Label"].str.strip() == "FTP-Patator"]
    print(f"published FTP-Patator rows for comparison: {len(ftp):,}")

    check = flow_features.validate_against_csv(
        extracted, ftp,
        features=["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                  "Total Length of Fwd Packets", "Fwd Packet Length Max",
                  "Fwd Packet Length Mean", "Bwd Packet Length Max",
                  "Destination Port", "Init_Win_bytes_forward"])
    agree = sum(1 for v in check.values() if v["agrees"])
    print(f"\nextractor agreement with CICFlowMeter: {agree}/{len(check)} features")
    for f, v in check.items():
        flag = "ok " if v["agrees"] else "NO "
        print(f"  {flag} {f:<30} mine {v['mine_median']:>12,.1f}   "
              f"published {v['published_median']:>12,.1f}")

    # ---------------- align extracted flows to the model's schema ----------
    rows = []
    for e in extracted:
        rows.append([e.get(n, 0.0) for n in names])
    F = pd.DataFrame(rows, columns=names).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y_flows = np.ones(len(F), dtype=int)
    F_s = scaler.transform(F)
    detected = model.predict(F_s) == 1
    print(f"\nvictim detects {detected.sum()}/{len(F)} of our extracted flows as attacks")
    if detected.sum() < 5:
        print("  too few detected to attack; stopping.")
        return

    keep = np.where(detected)[0]
    F_s, y_flows = F_s[keep], y_flows[keep]
    flows_kept = [list(raw.values())[i] for i in keep]
    mask = np.ones(len(F_s), dtype=bool)

    # ---------------- rung 1 and 2: feature space --------------------------
    res = {}
    for label, proj in (("unconstrained", None), ("constrained", projector)):
        X_adv = attacks.pgd(model, F_s, y_flows, EPSILON, steps=STEPS,
                            restarts=RESTARTS, projector=proj,
                            gradient_fn=grad, mask=mask, seed=SEED)[0]
        ev = metrics.evasion_rate(int((model.predict(X_adv) == 0).sum()), len(X_adv))
        res[label] = ev
        print(f"\n{label:<14} feature-space evasion: {ev:.4f}")

    # ---------------- rung 3: realized in packets --------------------------
    print("\nrealizing perturbations as packet operations:")
    realized = []
    for name, pad, delay in BUDGETS:
        mod = [perturb(f, pad, delay) for f in flows_kept]
        feats = [flow_features.extract_features(m) for m in mod]
        M = pd.DataFrame([[e.get(n, 0.0) for n in names] for e in feats],
                         columns=names).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        preds = model.predict(scaler.transform(M))
        ev = float((preds == 0).mean())
        realized.append({"budget": name, "pad_bytes": pad, "delay_s": delay,
                         "evasion": ev})
        print(f"  {name:<24} evasion {ev:.4f}")

    best = max(r["evasion"] for r in realized)
    print("\n" + "=" * 70)
    print("THE THREE RUNGS")
    print(f"  unconstrained feature space : {res['unconstrained']:.4f}")
    print(f"  constrained feature space   : {res['constrained']:.4f}")
    print(f"  realized in packets         : {best:.4f}")
    print("=" * 70)

    with open(REPORT, "w") as fh:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_flows_extracted": len(extracted),
            "n_flows_detected": int(detected.sum()),
            "extractor_agreement": check,
            "feature_space": res,
            "realized": realized,
        }, fh, indent=2)
    print(f"\nReport saved to {REPORT}")


if __name__ == "__main__":
    main()
