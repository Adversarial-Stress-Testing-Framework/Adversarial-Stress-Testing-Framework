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
  computed       determined by other features through a stated formula, and
                 recalculated from them after every projection rather than
                 merely bounded

`computed` exists because bounding a derived feature is not enough. Packet
Length Mean is total bytes over total packets. An attacker who pads a payload
does not then get to choose the resulting mean - it follows. Declaring the
relationship makes that automatic:

    Fwd Packet Length Mean:
      role: computed
      formula: "{Total Length of Fwd Packets} / {Total Fwd Packets}"

Only formulas verified against real data belong here. Of eighteen candidates
checked against CICIDS2017, fourteen reproduce the published column on every
flow and four do not - CICFlowMeter computes Packet Length Mean, Average Packet
Size, Down/Up Ratio and Fwd IAT Total differently from what their names imply.
Those four stay `derived` rather than carry a formula that is wrong.

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

import ast
import json
import re
from pathlib import Path

IMMUTABLE = "immutable"
INCREASE_ONLY = "increase_only"
DERIVED_RATE = "derived_rate"
DERIVED = "derived"
COMPUTED = "computed"
ROLES = (IMMUTABLE, INCREASE_ONLY, DERIVED_RATE, DERIVED, COMPUTED)

# Expression nodes a formula may contain. Deliberately minimal: arithmetic over
# feature references and literals, nothing else. Specifications are authored in
# this project rather than supplied by users, but a formula that can only add,
# subtract, multiply, divide and raise to a power cannot be made to do anything
# surprising by a typo either.
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)
_REF = re.compile(r"\{([^{}]+)\}")


def compile_formula(formula, feature_index):
    """Turn '{a} / {b}' into a callable over a column matrix.

    Returns (fn, [indices it reads]). fn takes the full (n_samples, n_features)
    array and returns one column.
    """
    refs = _REF.findall(formula)
    missing = [r for r in refs if r not in feature_index]
    if missing:
        raise SpecError(f"formula {formula!r} references unknown feature(s) {missing}")

    slots = {}
    expr = formula
    for r in refs:
        var = slots.setdefault(r, f"_v{len(slots)}")
        expr = expr.replace("{" + r + "}", var)

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SpecError(
                f"formula {formula!r} contains a disallowed expression "
                f"({type(node).__name__}); only arithmetic is permitted"
            )
    code = compile(tree, "<formula>", "eval")
    order = [(var, feature_index[name]) for name, var in slots.items()]

    def fn(X):
        return eval(code, {"__builtins__": {}}, {v: X[:, i] for v, i in order})

    return fn, [i for _, i in order]

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
        for feat, entry in self.features.items():
            if entry.get("role") == COMPUTED and not entry.get("formula"):
                raise SpecError(f"computed feature {feat!r} has no formula")
            if entry.get("formula"):
                for ref in _REF.findall(entry["formula"]):
                    if ref not in self.features:
                        raise SpecError(
                            f"formula for {feat!r} references unknown feature {ref!r}")
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

    def formula_of(self, feature):
        return self.features.get(feature, {}).get("formula")

    def computed_features(self):
        return {f for f in self.features if self.role_of(f) == COMPUTED}

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
        if c[COMPUTED]:
            parts.append(f"{c[COMPUTED]} computed")
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
