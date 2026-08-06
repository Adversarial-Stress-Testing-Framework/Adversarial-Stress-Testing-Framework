"""
Domain constraints for NSL-KDD network flows.

An unconstrained gradient attack treats all 41 NSL-KDD features as free real
numbers. They are not. A flow vector describes a real TCP/UDP/ICMP connection,
so most features are either fixed by the attack's own semantics, controlled by
the victim rather than the attacker, or arithmetically derived from other
features. Perturbing them freely produces vectors that no packet capture could
ever yield - negative byte counts, fractional protocol identifiers, service
indices halfway between 'http' and 'smtp'.

This module classifies every feature by what an attacker can actually do to it,
then supplies a projection operator that snaps an arbitrary adversarial
candidate back onto the set of realizable flows.

Roles
-----
IMMUTABLE      The attacker cannot alter this without either destroying the
               attack's function or controlling the victim host. Includes the
               categorical identifiers (perturbing them is meaningless once
               label-encoded), the connection outcome flags, and the
               server-side byte count.

INCREASE_ONLY  The attacker can raise this but not lower it. You can pad a
               payload, stall a connection, or open extra sessions; you cannot
               un-send bytes or retroactively reduce a connection count.

DERIVED_RATE   A ratio over a traffic window, bounded to [0, 1] and coupled to
               its siblings. Not directly writable, but an attacker steers it
               indirectly by choosing which connections to open, so it is
               allowed to move within its bounds and pairwise sum limits.
"""

import numpy as np

IMMUTABLE = "immutable"
INCREASE_ONLY = "increase_only"
DERIVED_RATE = "derived_rate"


FEATURE_ROLES = {
    # --- attacker-controllable volume and timing -------------------------
    "duration":            INCREASE_ONLY,  # can stall, cannot retroactively shorten
    "src_bytes":           INCREASE_ONLY,  # can pad the payload
    "wrong_fragment":      INCREASE_ONLY,  # can emit additional malformed fragments
    "urgent":              INCREASE_ONLY,  # can set more URG pointers
    "num_failed_logins":   INCREASE_ONLY,  # can add throwaway failed attempts
    "num_file_creations":  INCREASE_ONLY,  # can touch extra files
    "count":               INCREASE_ONLY,  # can open more same-host connections
    "srv_count":           INCREASE_ONLY,  # can open more same-service connections
    "dst_host_count":      INCREASE_ONLY,
    "dst_host_srv_count":  INCREASE_ONLY,

    # --- fixed by the connection's identity ------------------------------
    "protocol_type":       IMMUTABLE,  # label-encoded; a fractional protocol is meaningless
    "service":             IMMUTABLE,  # ditto, and changing it changes the attack's target
    "flag":                IMMUTABLE,  # determined by how the connection actually terminated
    "land":                IMMUTABLE,  # structural: src addr/port == dst addr/port

    # --- victim-controlled or attack-outcome features --------------------
    "dst_bytes":           IMMUTABLE,  # the server decides how much it sends back
    "hot":                 IMMUTABLE,  # payload content indicators; changing them changes the attack
    "logged_in":           IMMUTABLE,  # outcome, not an input
    "num_compromised":     IMMUTABLE,  # outcome
    "root_shell":          IMMUTABLE,  # outcome - this IS the attack succeeding
    "su_attempted":        IMMUTABLE,  # outcome
    "num_root":            IMMUTABLE,  # outcome
    "num_shells":          IMMUTABLE,  # outcome
    "num_access_files":    IMMUTABLE,  # outcome
    "num_outbound_cmds":   IMMUTABLE,  # all-zero in KDDTrain+; dropped by clean_data
    "is_host_login":       IMMUTABLE,  # property of the account, not the traffic
    "is_guest_login":      IMMUTABLE,

    # --- window-derived ratios -------------------------------------------
    "serror_rate":                  DERIVED_RATE,
    "srv_serror_rate":              DERIVED_RATE,
    "rerror_rate":                  DERIVED_RATE,
    "srv_rerror_rate":              DERIVED_RATE,
    "same_srv_rate":                DERIVED_RATE,
    "diff_srv_rate":                DERIVED_RATE,
    "srv_diff_host_rate":           DERIVED_RATE,
    "dst_host_same_srv_rate":       DERIVED_RATE,
    "dst_host_diff_srv_rate":       DERIVED_RATE,
    "dst_host_same_src_port_rate":  DERIVED_RATE,
    "dst_host_srv_diff_host_rate":  DERIVED_RATE,
    "dst_host_serror_rate":         DERIVED_RATE,
    "dst_host_srv_serror_rate":     DERIVED_RATE,
    "dst_host_rerror_rate":         DERIVED_RATE,
    "dst_host_srv_rerror_rate":     DERIVED_RATE,
}


# Rate pairs that describe complementary slices of the same traffic window, and
# so are jointly bounded.
#
# The obvious bound is 1.0 - a connection is either to the same service or a
# different one - but that is NOT what NSL-KDD actually contains. Real,
# unmodified traffic reaches 1.5 on the same_srv/diff_srv pairs, because the
# KDD feature extractor computes the two rates over different reference windows
# and rounds each to two decimals. Asserting the textbook bound flagged 5.2% of
# genuine flows as impossible and tripled the false-positive rate.
#
# So the cap is learned from training data in ConstraintProjector.fit() rather
# than assumed here. This list only declares which pairs are coupled; the data
# decides how tightly.
RATE_SUM_PAIRS = [
    ("same_srv_rate", "diff_srv_rate"),
    ("dst_host_same_srv_rate", "dst_host_diff_srv_rate"),
    ("serror_rate", "rerror_rate"),
    ("srv_serror_rate", "srv_rerror_rate"),
    ("dst_host_serror_rate", "dst_host_rerror_rate"),
    ("dst_host_srv_serror_rate", "dst_host_srv_rerror_rate"),
]


# Features that count discrete events and must land on integers.
INTEGER_FEATURES = {
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "dst_host_count", "dst_host_srv_count",
    "protocol_type", "service", "flag",
}


class ConstraintProjector:
    """Projects adversarial candidates onto the realizable-flow set.

    The attack operates in StandardScaler space, but realizability is a property
    of the original units, so every projection round-trips through the scaler.
    Per-feature bounds are learned from the training split rather than hardcoded,
    which keeps the class usable on datasets other than NSL-KDD.
    """

    def __init__(self, feature_names, scaler, roles=None):
        self.feature_names = list(feature_names)
        self.scaler = scaler
        self.roles = roles or FEATURE_ROLES

        self._idx = {name: i for i, name in enumerate(self.feature_names)}
        self._immutable = self._mask_for(IMMUTABLE)
        self._increase_only = self._mask_for(INCREASE_ONLY)
        self._rate = self._mask_for(DERIVED_RATE)
        self._integer = np.array(
            [n in INTEGER_FEATURES for n in self.feature_names], dtype=bool
        )
        self._rate_pairs = [
            (self._idx[a], self._idx[b])
            for a, b in RATE_SUM_PAIRS
            if a in self._idx and b in self._idx
        ]
        self.lower_ = None
        self.upper_ = None
        self.pair_caps_ = None

    def _mask_for(self, role):
        # A feature with no declared role is treated as immutable: the
        # conservative choice, since granting the attacker capabilities they
        # may not have would inflate the measured vulnerability.
        return np.array(
            [self.roles.get(n, IMMUTABLE) == role for n in self.feature_names],
            dtype=bool,
        )

    def fit(self, X_train_original):
        """Learn the realizable envelope from observed (unscaled) training data."""
        X = np.asarray(X_train_original, dtype=float)
        self.lower_ = X.min(axis=0)
        self.upper_ = X.max(axis=0)
        # Rates are bounded by definition, not by what happened to be observed.
        self.lower_[self._rate] = 0.0
        self.upper_[self._rate] = 1.0

        # Learn each coupled pair's joint cap from real traffic. Anything the
        # data actually produces is realizable by definition, so the observed
        # maximum is the honest bound.
        self.pair_caps_ = {
            (i, j): float((X[:, i] + X[:, j]).max())
            for i, j in self._rate_pairs
        }
        return self

    def project(self, X_adv_scaled, X_clean_scaled):
        """Snap adversarial candidates back onto the realizable set.

        Both arguments are in scaled space; the return value is too, so this
        drops directly into an attack loop.
        """
        if self.lower_ is None:
            raise RuntimeError("call fit() with the unscaled training features first")

        adv = self.scaler.inverse_transform(np.asarray(X_adv_scaled, dtype=float))
        clean = self.scaler.inverse_transform(np.asarray(X_clean_scaled, dtype=float))

        # 1. Immutable features revert to their original values outright.
        adv[:, self._immutable] = clean[:, self._immutable]

        # 2. Increase-only features may not fall below where they started.
        if self._increase_only.any():
            cols = self._increase_only
            adv[:, cols] = np.maximum(adv[:, cols], clean[:, cols])

        # 3. Everything stays inside the observed/definitional envelope.
        adv = np.clip(adv, self.lower_, self.upper_)

        # 4. Discrete counters land on integers.
        if self._integer.any():
            adv[:, self._integer] = np.rint(adv[:, self._integer])

        # 5. Coupled rates cannot jointly exceed the cap real traffic exhibits;
        #    scale the pair down together so their ratio - which is what the
        #    rate actually encodes - survives.
        for (i, j), cap in self.pair_caps_.items():
            total = adv[:, i] + adv[:, j]
            over = total > cap
            if over.any():
                scale = cap / total[over]
                adv[over, i] *= scale
                adv[over, j] *= scale

        # Re-clip: rounding in step 4 can push a value one unit past a bound.
        adv = np.clip(adv, self.lower_, self.upper_)

        return self.scaler.transform(adv)

    def summary(self):
        """Feature counts per role, for reporting."""
        counts = {IMMUTABLE: 0, INCREASE_ONLY: 0, DERIVED_RATE: 0}
        for name in self.feature_names:
            counts[self.roles.get(name, IMMUTABLE)] += 1
        return counts


def realizability_violations(X_original, feature_names, projector):
    """Count how many domain rules a batch of unscaled flows breaks.

    Used to quantify the gap between what an unconstrained attack produces and
    what a packet capture could actually contain.
    """
    X = np.asarray(X_original, dtype=float)
    idx = {n: i for i, n in enumerate(feature_names)}
    report = {}

    negatives = 0
    for name, i in idx.items():
        if name in INTEGER_FEATURES and (X[:, i] < 0).any():
            negatives += 1
    report["features_with_negative_values"] = negatives

    fractional = 0
    for name, i in idx.items():
        if name in INTEGER_FEATURES and not np.allclose(X[:, i], np.rint(X[:, i])):
            fractional += 1
    report["integer_features_with_fractional_values"] = fractional

    out_of_range_rates = 0
    for name, i in idx.items():
        if projector.roles.get(name) == DERIVED_RATE:
            if (X[:, i] < 0).any() or (X[:, i] > 1).any():
                out_of_range_rates += 1
    report["rate_features_outside_0_1"] = out_of_range_rates

    broken_pairs = 0
    for (i, j), cap in (projector.pair_caps_ or {}).items():
        if (X[:, i] + X[:, j] > cap + 1e-9).any():
            broken_pairs += 1
    report["violated_rate_sum_constraints"] = broken_pairs

    return report
