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

from stress_test.spec import (  # noqa: F401  (re-exported for callers)
    FeatureSpec, SpecError, IMMUTABLE, INCREASE_ONLY, DERIVED_RATE, DERIVED,
    COMPUTED,
)

_DEFAULT_SPEC = None


def _default_spec():
    """The NSL-KDD specification, loaded once.

    The taxonomy below is retained as a literal fallback so the module still
    imports if specs/ is missing, but specs/nsl_kdd.yaml is the source of truth
    and is what other datasets are modelled on.
    """
    global _DEFAULT_SPEC
    if _DEFAULT_SPEC is None:
        try:
            _DEFAULT_SPEC = FeatureSpec.load("nsl_kdd")
        except SpecError:
            _DEFAULT_SPEC = FeatureSpec(
                name="nsl-kdd (builtin fallback)",
                features={n: {"role": r, "integer": n in INTEGER_FEATURES}
                          for n, r in FEATURE_ROLES.items()},
                coupled_rates=RATE_SUM_PAIRS,
            )
    return _DEFAULT_SPEC


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

    def __init__(self, feature_names, scaler, roles=None, spec=None):
        """`spec` is a FeatureSpec describing this dataset's schema.

        Omitting it falls back to the built-in NSL-KDD specification, so the
        existing runners keep working unchanged. `roles`, a plain dict, is the
        older interface and still overrides the spec when supplied.
        """
        self.feature_names = list(feature_names)
        self.scaler = scaler

        if spec is None and roles is None:
            spec = _default_spec()
        self.spec = spec
        self.roles = roles or (spec.roles_map() if spec else FEATURE_ROLES)

        integer_names = spec.integer_features() if spec else INTEGER_FEATURES
        pairs = spec.coupled_rates if spec else RATE_SUM_PAIRS

        self._idx = {name: i for i, name in enumerate(self.feature_names)}
        self._immutable = self._mask_for(IMMUTABLE)
        self._increase_only = self._mask_for(INCREASE_ONLY)
        self._rate = self._mask_for(DERIVED_RATE)
        self._integer = np.array(
            [n in integer_names for n in self.feature_names], dtype=bool
        )
        self._rate_pairs = [
            (self._idx[a], self._idx[b])
            for a, b in pairs
            if a in self._idx and b in self._idx
        ]
        # Ordering chains: each element must stay <= the next. NSL-KDD needs
        # none; CICIDS2017 has many (min <= mean <= max on packet lengths and
        # inter-arrival times). Chains are kept only where every member is
        # present in this schema, so a partially-matching spec degrades to
        # enforcing nothing rather than indexing off the end.
        chains = spec.ordering if spec else []
        self._ordering = [
            [self._idx[f] for f in chain]
            for chain in chains
            if all(f in self._idx for f in chain)
        ]
        self._computed = self._compile_computed(spec)
        self.lower_ = None
        self.upper_ = None
        self.pair_caps_ = None

    def _compile_computed(self, spec):
        """Compile computed features into evaluation order.

        Order matters: Avg Fwd Segment Size is defined as Fwd Packet Length
        Mean, which is itself computed, so it must be evaluated after it. A
        topological sort over the reference graph gives that. A cycle means the
        specification is contradictory and is worth failing on rather than
        resolving arbitrarily.
        """
        if spec is None:
            return []

        from stress_test.spec import compile_formula

        pending = {}
        for name in self.feature_names:
            if spec.role_of(name) == DERIVED and spec.formula_of(name):
                pass  # a formula on a non-computed role is advisory only
            if spec.role_of(name) != COMPUTED:
                continue
            formula = spec.formula_of(name)
            fn, reads = compile_formula(formula, self._idx)
            pending[self._idx[name]] = (fn, reads, name)

        ordered, emitted = [], set()
        while pending:
            ready = [i for i, (_, reads, _) in pending.items()
                     if not (set(reads) & set(pending) - {i} - emitted)]
            if not ready:
                cycle = [n for _, _, n in pending.values()]
                raise SpecError(f"computed features form a dependency cycle: {cycle}")
            for i in sorted(ready):
                fn, reads, name = pending.pop(i)
                ordered.append((i, fn, name))
                emitted.add(i)
        return ordered

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

        # 6. Ordered siblings must stay ordered. A running maximum along the
        #    chain is the cheapest projection that guarantees it: each element
        #    is pushed up to at least its predecessor, so min <= mean <= max
        #    holds afterwards regardless of where the attack left them.
        #    Raising rather than lowering keeps the perturbation in the
        #    direction the attack chose wherever the constraint allows.
        for chain in self._ordering:
            for a, b in zip(chain, chain[1:]):
                adv[:, b] = np.maximum(adv[:, b], adv[:, a])

        # 7. Computed features are recalculated from whatever the projection
        #    left their constituents at. This is the step that makes derived
        #    quantities follow the attack rather than be chosen by it: pad a
        #    payload and the resulting mean packet length is determined, not
        #    free. Bounding such a feature is not a constraint at all when its
        #    observed range spans millions.
        for i, fn, _name in self._computed:
            with np.errstate(divide="ignore", invalid="ignore"):
                val = fn(adv)
            # A zero denominator means the flow has no packets or no duration.
            # CICFlowMeter emits infinity there and the loader drops those rows;
            # mid-attack the honest fallback is the value the sample started
            # with, which is by construction realizable.
            bad = ~np.isfinite(val)
            if bad.any():
                val = np.where(bad, clean[:, i], val)
            adv[:, i] = val

        # 8. Re-establish ordering around the recomputed values. Step 6 ordered
        #    the chains, but step 7 then moved the computed members, which can
        #    place a recomputed mean outside the min and max that had just been
        #    made consistent with it.
        #
        #    A computed feature is determined, so it is the anchor: the free
        #    members move to accommodate it, not the other way round. Pushing
        #    the mean back between them instead would silently break the very
        #    relationship step 7 exists to enforce.
        if self._computed and self._ordering:
            anchored = {i for i, _, _ in self._computed}
            for chain in self._ordering:
                for pos, idx_b in enumerate(chain):
                    if idx_b not in anchored:
                        continue
                    for idx_a in chain[:pos]:          # predecessors must not exceed it
                        if idx_a not in anchored:
                            adv[:, idx_a] = np.minimum(adv[:, idx_a], adv[:, idx_b])
                    for idx_c in chain[pos + 1:]:      # successors must not fall below it
                        if idx_c not in anchored:
                            adv[:, idx_c] = np.maximum(adv[:, idx_c], adv[:, idx_b])

        # Re-clip: rounding in step 4, the running maximum in step 6, and
        # recomputation in step 7 can all push a value past a bound.
        adv = np.clip(adv, self.lower_, self.upper_)

        return self.scaler.transform(adv)

    def summary(self):
        """Feature counts per role, for reporting."""
        counts = {IMMUTABLE: 0, INCREASE_ONLY: 0, DERIVED_RATE: 0, DERIVED: 0, COMPUTED: 0}
        for name in self.feature_names:
            counts[self.roles.get(name, IMMUTABLE)] += 1
        return counts


def realizability_violations(X_original, feature_names, projector, tol=1e-9):
    """Count how many domain rules a batch of unscaled flows breaks.

    Used to quantify the gap between what an unconstrained attack produces and
    what a packet capture could actually contain.

    `tol` absorbs floating-point noise rather than real violations. Projecting a
    rate to exactly 1.0 and round-tripping it through StandardScaler returns
    1.0000000000000002 - one unit in the last place of a float64. A strict
    comparison counts that as an out-of-range rate and reports a violation the
    constrained attack did not actually commit.
    """
    X = np.asarray(X_original, dtype=float)
    idx = {n: i for i, n in enumerate(feature_names)}
    report = {}

    # Counters are scale-aware: an absolute tolerance is meaningless against
    # src_bytes, which reaches 1.4e9 and carries proportionally larger
    # round-trip error than a rate bounded to [0, 1].
    def slack(col):
        return tol * np.maximum(1.0, np.abs(col))

    # Non-negativity is a property of the feature, not of whether it happens to
    # be integral. This previously only checked integer features, which was
    # adequate for NSL-KDD - its counters are integers - but silently ignored 20
    # CICIDS2017 columns that are equally incapable of going negative: flow
    # durations, packet lengths, inter-arrival times. Whether a feature can be
    # negative is learned from the training split like every other bound.
    non_negative = (
        projector.lower_ >= 0 if projector.lower_ is not None
        else np.array([n in INTEGER_FEATURES for n in feature_names])
    )
    negatives = 0
    for name, i in idx.items():
        if non_negative[i] and (X[:, i] < -slack(X[:, i])).any():
            negatives += 1
    report["features_with_negative_values"] = negatives

    fractional = 0
    for name, i in idx.items():
        if name in INTEGER_FEATURES:
            col = X[:, i]
            if (np.abs(col - np.rint(col)) > slack(col)).any():
                fractional += 1
    report["integer_features_with_fractional_values"] = fractional

    out_of_range_rates = 0
    for name, i in idx.items():
        if projector.roles.get(name) == DERIVED_RATE:
            col = X[:, i]
            if (col < -tol).any() or (col > 1 + tol).any():
                out_of_range_rates += 1
    report["rate_features_outside_0_1"] = out_of_range_rates

    broken_pairs = 0
    for (i, j), cap in (projector.pair_caps_ or {}).items():
        if (X[:, i] + X[:, j] > cap + tol).any():
            broken_pairs += 1
    report["violated_rate_sum_constraints"] = broken_pairs

    # Flows whose smallest packet exceeds their largest, and similar. NSL-KDD
    # declares no ordering chains so this is always zero there; it is the
    # violation class that dominates on richer schemas like CICIDS2017.
    broken_order = 0
    for chain in getattr(projector, "_ordering", []):
        for a, b in zip(chain, chain[1:]):
            if (X[:, a] > X[:, b] + tol * np.maximum(1.0, np.abs(X[:, b]))).any():
                broken_order += 1
    report["violated_ordering_constraints"] = broken_order

    return report
