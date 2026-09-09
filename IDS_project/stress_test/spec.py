"""
Dataset-independent feature specifications.

The constraint taxonomy used to be hardcoded for NSL-KDD, which made the
framework a study of one dataset rather than a tool. A specification declares,
for one dataset's schema, what an attacker can do to each feature:

    name: nsl-kdd
    features:
      protocol_type: {role: immutable}
      src_bytes:     {role: increase_only, integer: true}
      same_srv_rate: {role: derived_rate}
    coupled_rates:
      - [same_srv_rate, diff_srv_rate]

Roles
  immutable      the attacker cannot alter it without destroying the attack's
                 function, or it is controlled by the victim
  increase_only  the attacker can raise it but not lower it - padding, stalling,
                 opening extra connections
  derived_rate   a ratio over a traffic window, bounded to [0, 1] and coupled to
                 its siblings
  derived        a statistic of other features - a mean, a standard deviation, a
                 per-second rate. Movable in either direction but bounded by
                 what real traffic exhibits, and often ordered against its
                 siblings

`derived` exists because NSL-KDD is unusually simple. Every one of its computed
features happens to be a percentage, so [0, 1] sufficed. CICIDS2017 has packet
length means reaching 1500, flow rates in the millions and inter-arrival
standard deviations in microseconds - forcing those into [0, 1] would destroy
them.

Ordering constraints capture the other thing NSL-KDD did not need:

    ordering:
      - [Min Packet Length, Packet Length Mean, Max Packet Length]

meaning each element must be <= the next. An unconstrained attack moving those
three independently will happily emit a flow whose smallest packet is larger
than its largest. Without this the projector has no way to notice.

What is deliberately NOT in a spec: numeric bounds. Those are learned from the
training split. An earlier version asserted that same_srv_rate + diff_srv_rate
could not exceed 1, on the grounds that it must logically hold; 3,574 genuine
NSL-KDD flows reach 1.5. Anything a spec asserts about magnitudes is a guess,
so specs assert only structure and the data supplies the numbers.
"""

import json
from pathlib import Path

IMMUTABLE = "immutable"
INCREASE_ONLY = "increase_only"
DERIVED_RATE = "derived_rate"
DERIVED = "derived"
ROLES = (IMMUTABLE, INCREASE_ONLY, DERIVED_RATE, DERIVED)

SPEC_DIR = Path(__file__).parent / "specs"


class SpecError(ValueError):
    """Raised when a specification is malformed or does not fit the data."""


class FeatureSpec:
    """What an attacker can do to each feature of one dataset."""

    def __init__(self, name, features, coupled_rates=None, ordering=None,
                 description=""):
        self.name = name
        self.description = description
        self.features = dict(features)
        self.coupled_rates = [tuple(p) for p in (coupled_rates or [])]
        self.ordering = [list(c) for c in (ordering or [])]

        for feat, entry in self.features.items():
            role = entry.get("role")
            if role not in ROLES:
                raise SpecError(
                    f"feature {feat!r} has role {role!r}; expected one of {ROLES}"
                )
        for a, b in self.coupled_rates:
            for f in (a, b):
                if f not in self.features:
                    raise SpecError(f"coupled_rates references unknown feature {f!r}")
        for chain in self.ordering:
            if len(chain) < 2:
                raise SpecError(f"ordering chain {chain!r} needs at least two features")
            for f in chain:
                if f not in self.features:
                    raise SpecError(f"ordering references unknown feature {f!r}")

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path):
        """Load a spec from YAML or JSON. Bare names resolve against specs/."""
        p = Path(path)
        if not p.exists() and not p.suffix:
            for ext in (".yaml", ".yml", ".json"):
                if (SPEC_DIR / f"{p.name}{ext}").exists():
                    p = SPEC_DIR / f"{p.name}{ext}"
                    break
        if not p.exists():
            raise SpecError(f"no specification at {path!r} (looked in {SPEC_DIR})")

        text = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            raw = json.loads(text)
        else:
            import yaml  # only needed for YAML specs
            raw = yaml.safe_load(text)

        if "features" not in raw:
            raise SpecError(f"{p.name} has no 'features' section")
        return cls(
            name=raw.get("name", p.stem),
            features=raw["features"],
            coupled_rates=raw.get("coupled_rates"),
            ordering=raw.get("ordering"),
            description=raw.get("description", ""),
        )

    # ---------------------------------------------------------------- queries
    def role_of(self, feature):
        # Unknown features are treated as immutable. Granting the attacker
        # capabilities they may not have would inflate measured vulnerability,
        # so the conservative default is the safe one.
        return self.features.get(feature, {}).get("role", IMMUTABLE)

    def is_integer(self, feature):
        return bool(self.features.get(feature, {}).get("integer", False))

    def integer_features(self):
        return {f for f in self.features if self.is_integer(f)}

    def roles_map(self):
        return {f: self.role_of(f) for f in self.features}

    def counts(self, feature_names=None):
        names = list(feature_names) if feature_names is not None else list(self.features)
        out = {r: 0 for r in ROLES}
        for n in names:
            out[self.role_of(n)] += 1
        return out

    # ------------------------------------------------------------- validation
    def validate(self, feature_names, strict=False):
        """Check this spec against an actual schema.

        Returns (undeclared, unused). `undeclared` are columns present in the
        data with no entry here - they silently default to immutable, which
        understates the attacker, so they are worth surfacing.
        """
        names = list(feature_names)
        undeclared = [n for n in names if n not in self.features]
        unused = [f for f in self.features if f not in names]
        if strict and undeclared:
            raise SpecError(
                f"{len(undeclared)} column(s) absent from spec {self.name!r}: "
                f"{undeclared[:8]}{' ...' if len(undeclared) > 8 else ''}"
            )
        return undeclared, unused

    def __repr__(self):
        c = self.counts()
        parts = [f"{c[IMMUTABLE]} immutable", f"{c[INCREASE_ONLY]} increase-only"]
        # Only mention the derived kinds a schema actually uses: NSL-KDD has no
        # plain derived features and CICIDS2017 has no rates, so printing both
        # unconditionally reads as a zero where there is simply no such column.
        if c[DERIVED_RATE]:
            parts.append(f"{c[DERIVED_RATE]} rate")
        if c[DERIVED]:
            parts.append(f"{c[DERIVED]} derived")
        chains = f", {len(self.ordering)} ordering chains" if self.ordering else ""
        return (f"<FeatureSpec {self.name!r}: {len(self.features)} features, "
                f"{' / '.join(parts)}{chains}>")


def available():
    """Names of the specs shipped with the framework."""
    if not SPEC_DIR.exists():
        return []
    return sorted({p.stem for p in SPEC_DIR.iterdir()
                   if p.suffix in (".yaml", ".yml", ".json")})


def load_default():
    return FeatureSpec.load("nsl_kdd")
