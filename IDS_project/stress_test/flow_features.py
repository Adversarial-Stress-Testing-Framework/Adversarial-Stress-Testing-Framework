"""
Compute CICFlowMeter-compatible features from packet records.

Phase 3 needs to go from packets to a feature vector and back, so it needs an
extractor. The official CICFlowMeter is Java and depends on jnetpcap, whose
native libraries are difficult to build on Windows; rather than fight that, this
reimplements the subset of features the audit uses.

A reimplementation is only worth anything if it agrees with the original, so
validate_against_csv() compares the output against the published CSVs for the
same traffic. An extractor that does not reproduce CICFlowMeter's own numbers
cannot be used to argue anything about CICFlowMeter's own models, and the
comparison is what licenses the rest of the phase.

Packet records carry what the features need and nothing else - timestamp,
direction, sizes, flags, window - because holding scapy objects for a capture
this size exhausts memory.
"""

import numpy as np

ACTIVE_TIMEOUT = 5.0     # seconds of quiet that ends an active period
FLAG_LETTERS = {"FIN": "F", "SYN": "S", "RST": "R", "PSH": "P",
                "ACK": "A", "URG": "U", "ECE": "E", "CWE": "C"}


def _stats(v, prefix, out):
    """Max/Min/Mean/Std under CICFlowMeter's naming, zero-filled when empty."""
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        out[f"{prefix} Max"] = out[f"{prefix} Min"] = 0.0
        out[f"{prefix} Mean"] = out[f"{prefix} Std"] = 0.0
        return
    out[f"{prefix} Max"] = float(a.max())
    out[f"{prefix} Min"] = float(a.min())
    out[f"{prefix} Mean"] = float(a.mean())
    # CICFlowMeter reports the population standard deviation.
    out[f"{prefix} Std"] = float(a.std())


def _iat(times):
    return np.diff(np.sort(np.asarray(times, dtype=float))) if len(times) > 1 else np.array([])


def extract_features(packets):
    """One packet list -> one CICFlowMeter-shaped feature dict."""
    pk = sorted(packets, key=lambda p: p["t"])
    t = np.array([p["t"] for p in pk], dtype=float)
    fwd = np.array([p["fwd"] for p in pk], dtype=bool)
    payload = np.array([p["payload"] for p in pk], dtype=float)

    duration_s = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    duration_us = duration_s * 1e6
    # Rates divide by duration; a single-packet flow has none. CICFlowMeter
    # emits infinity here and the published CSVs carry it, which is why the
    # loader drops those rows. Zero is the honest stand-in inside the extractor.
    per_s = (lambda x: x / duration_s) if duration_s > 0 else (lambda x: 0.0)

    f, b = payload[fwd], payload[~fwd]
    out = {
        "Destination Port": float(pk[0]["dport"]),
        "Flow Duration": duration_us,
        "Total Fwd Packets": float(fwd.sum()),
        "Total Backward Packets": float((~fwd).sum()),
        "Total Length of Fwd Packets": float(f.sum()),
        "Total Length of Bwd Packets": float(b.sum()),
    }
    _stats(f, "Fwd Packet Length", out)
    _stats(b, "Bwd Packet Length", out)

    out["Flow Bytes/s"] = per_s(payload.sum())
    out["Flow Packets/s"] = per_s(len(pk))
    out["Fwd Packets/s"] = per_s(fwd.sum())
    out["Bwd Packets/s"] = per_s((~fwd).sum())

    _stats(_iat(t) * 1e6, "Flow IAT", out)
    for name, sel in (("Fwd", fwd), ("Bwd", ~fwd)):
        iat = _iat(t[sel]) * 1e6
        out[f"{name} IAT Total"] = float(iat.sum())
        _stats(iat, f"{name} IAT", out)

    for name, sel in (("Fwd", fwd), ("Bwd", ~fwd)):
        flags = [pk[i]["flags"] for i in np.where(sel)[0]]
        out[f"{name} PSH Flags"] = float(sum("P" in x for x in flags))
        out[f"{name} URG Flags"] = float(sum("U" in x for x in flags))
        out[f"{name} Header Length"] = float(sum(pk[i]["hdr"] for i in np.where(sel)[0]))

    allflags = [p["flags"] for p in pk]
    for label, letter in FLAG_LETTERS.items():
        out[f"{label} Flag Count"] = float(sum(letter in x for x in allflags))

    out["Min Packet Length"] = float(payload.min())
    out["Max Packet Length"] = float(payload.max())
    out["Packet Length Mean"] = float(payload.mean())
    out["Packet Length Std"] = float(payload.std())
    out["Packet Length Variance"] = float(payload.var())

    n_f, n_b = fwd.sum(), (~fwd).sum()
    out["Down/Up Ratio"] = float(n_b / n_f) if n_f else 0.0
    out["Average Packet Size"] = float(payload.mean())
    out["Avg Fwd Segment Size"] = out["Fwd Packet Length Mean"]
    out["Avg Bwd Segment Size"] = out["Bwd Packet Length Mean"]
    out["Fwd Header Length.1"] = out["Fwd Header Length"]

    for k in ("Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
              "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"):
        out[k] = 0.0   # constant zero in the released data too

    out["Subflow Fwd Packets"] = float(n_f)
    out["Subflow Fwd Bytes"] = float(f.sum())
    out["Subflow Bwd Packets"] = float(n_b)
    out["Subflow Bwd Bytes"] = float(b.sum())

    fwd_i, bwd_i = np.where(fwd)[0], np.where(~fwd)[0]
    out["Init_Win_bytes_forward"] = float(pk[fwd_i[0]]["win"]) if len(fwd_i) else -1.0
    out["Init_Win_bytes_backward"] = float(pk[bwd_i[0]]["win"]) if len(bwd_i) else -1.0
    out["act_data_pkt_fwd"] = float(sum(1 for i in fwd_i if pk[i]["payload"] > 0))
    out["min_seg_size_forward"] = float(min((pk[i]["hdr"] for i in fwd_i), default=0))

    # Active/idle: runs of packets separated by more than ACTIVE_TIMEOUT.
    active, idle, start, last = [], [], t[0], t[0]
    for cur in t[1:]:
        if cur - last > ACTIVE_TIMEOUT:
            active.append(last - start)
            idle.append(cur - last)
            start = cur
        last = cur
    active.append(last - start)
    _stats(np.array(active) * 1e6, "Active", out)
    _stats(np.array(idle) * 1e6, "Idle", out)
    return out


def validate_against_csv(extracted, csv_rows, features=None, rtol=0.05):
    """Compare extracted flows against published rows of the same kind.

    Exact row-level matching is impossible: the released CSVs strip source and
    destination addresses, so an extracted flow cannot be tied to its published
    counterpart. What can be checked is whether the distributions agree - if the
    extractor systematically disagrees with CICFlowMeter, the medians will not
    line up.
    """
    feats = features or sorted(set(extracted[0]) & set(csv_rows.columns))
    report = {}
    for f in feats:
        mine = np.array([e[f] for e in extracted], dtype=float)
        theirs = csv_rows[f].to_numpy(dtype=float)
        mine, theirs = mine[np.isfinite(mine)], theirs[np.isfinite(theirs)]
        if mine.size == 0 or theirs.size == 0:
            continue
        m1, m2 = float(np.median(mine)), float(np.median(theirs))
        denom = max(abs(m1), abs(m2), 1e-9)
        report[f] = {
            "mine_median": m1, "published_median": m2,
            "rel_diff": abs(m1 - m2) / denom,
            "agrees": abs(m1 - m2) / denom <= rtol,
        }
    return report
