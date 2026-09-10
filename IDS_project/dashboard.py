"""
Audit dashboard - visualises every test the stress-testing framework runs.

Reads the JSON reports produced by the experiment runners; it does not retrain
or re-attack anything, so it opens instantly. Regenerate the underlying data
with:

    python -m stress_test.run_full_matrix            # 4 models x 3 attacks x 3 defenses
    python -m stress_test.run_constrained_comparison # budget sweep
    python -m stress_test.run_multiseed              # error bars over five splits
    python -m stress_test.run_tree_variance          # why the trees swing
    python -m stress_test.run_cicids_audit           # second dataset
    python -m stress_test.run_packet_roundtrip       # packets -> features -> packets

Every tab states which report it came from and how many splits are behind it, so
a single-split number is never mistaken for a repeated one.

Launch with:

    streamlit run dashboard.py
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IDS Adversarial Audit", layout="wide")

FULL = "full_matrix_report.json"
SWEEP = "constrained_comparison_report.json"
LEGACY = "stress_test_report.json"
MULTISEED = "multiseed_report.json"
CICIDS = "cicids_audit_report.json"
TREEVAR = "tree_variance_report.json"
PACKETS = "packet_roundtrip_report.json"

ROLE_LABELS = {
    "immutable": "Locked",
    "increase_only": "Increase-only",
    "derived_rate": "Rate / percentage",
}

# Weakest to strongest. Altair sorts alphabetically by default, which would put
# BIM first and contradict every caption about strength increasing left to right.
ATTACK_ORDER = ["FGSM", "BIM", "PGD"]


# ----------------------------------------------------------------- loading
@st.cache_data
def _read(path_str, _mtime):
    # _mtime is part of the cache key, not the body: re-running an experiment
    # rewrites the report and must invalidate the cached copy. Without it the
    # dashboard silently keeps serving stale numbers, which is how two
    # contradictory figures end up on the same screen.
    return json.loads(Path(path_str).read_text())


def load(name):
    """Find a report next to this file or in the working directory."""
    for base in (Path(__file__).parent, Path.cwd()):
        p = base / name
        if p.exists():
            return _read(str(p), p.stat().st_mtime)
    return None


def evasion_after_hardening(entry):
    """Adversarial-training result, tolerating both report schema versions."""
    for key in ("evasion_adaptive", "evasion_under_constrained_pgd"):
        if entry.get(key) is not None:
            return entry[key]
    return None


def pct(x):
    return "n/a" if x is None else f"{x:.2%}"


def bar(df, x, y, color, title, y_title, facet=None, fmt=".0%", sort=None):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", title=None, sort=sort,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=fmt)),
            color=alt.Color(f"{color}:N", title=None,
                            scale=alt.Scale(scheme="tableau10")),
            xOffset=alt.XOffset(f"{color}:N"),
            tooltip=list(df.columns),
        )
        .properties(width=230, height=240)
    )
    if facet:
        # Wrap to two columns; four models side by side overflow the pane and
        # the last one is silently clipped.
        return chart.facet(
            facet=alt.Facet(f"{facet}:N", title=None), columns=2
        ).properties(title=title)
    return chart.properties(title=title)


def band(stat, lo=0.0, hi=None):
    """mean/std pair -> the mean and a one-SD interval, clipped to sane bounds."""
    m, s = stat["mean"], stat["std"]
    a, b = m - s, m + s
    if lo is not None:
        a = max(lo, a)
    if hi is not None:
        b = min(hi, b)
    return m, a, b


full = load(FULL)
sweep = load(SWEEP)
legacy = load(LEGACY)
multiseed = load(MULTISEED)
cicids = load(CICIDS)
treevar = load(TREEVAR)
packets = load(PACKETS)

st.title("🛡️ Adversarial Stress-Test Audit")

if full is None and sweep is None:
    st.error(
        "No report files found. Generate them first:\n\n"
        "```\npython -m stress_test.run_full_matrix\n"
        "python -m stress_test.run_constrained_comparison\n```"
    )
    st.stop()

with st.sidebar:
    st.header("Run configuration")
    if full:
        st.caption(f"Generated {full['timestamp']}")
        st.metric("Attack budget (ε)", full["epsilon"])
        st.metric("Iterative steps", full["steps"])
        st.metric("PGD restarts", full["restarts"])
        st.caption(full.get("attack_route", ""))
    if sweep:
        st.write(f"**Features audited:** {sweep['n_features']}")
    st.divider()
    st.caption(
        "Constrained = the attacker may only make changes a real network flow "
        "could actually contain. Unconstrained = the attacker may set any "
        "feature to any value, including impossible ones."
    )

TAB_NAMES = [
    "Overview", "Attacks", "Detection impact", "Attack budget",
    "Defenses", "Constraints", "Error bars", "Second dataset",
    "Packet round-trip",
]
T = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

# ----------------------------------------------------------------- overview
with T["Overview"]:
    st.subheader("Headline findings")

    # The repeated figure leads. A single split of this experiment lands
    # anywhere between 18x and 25x, so showing one split as "the" number is how
    # the dashboard and the write-up end up quoting different values.
    if multiseed:
        agg = multiseed["aggregate"]["LinearSVC"]
        n = len(multiseed["seeds"])
        st.caption(
            f"LinearSVC on NSL-KDD, black-box, **mean ± SD over {n} random "
            "train/test splits**. Every model is attacked through its own "
            "substitute, so all four face an identical attack."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attack wins — unconstrained",
                  f"{agg['unconstrained_evasion']['mean']:.1%}",
                  delta=f"± {agg['unconstrained_evasion']['std']:.1%}",
                  delta_color="off")
        c2.metric("Attack wins — constrained",
                  f"{agg['constrained_evasion']['mean']:.1%}",
                  delta=f"± {agg['constrained_evasion']['std']:.1%}",
                  delta_color="off")
        c3.metric("Vulnerability overstated by",
                  f"{agg['overstatement']['mean']:.1f}×",
                  delta=f"± {agg['overstatement']['std']:.1f}",
                  delta_color="off")
        if cicids:
            c4.metric("Same test, CICIDS2017",
                      f"{cicids['aggregate']['overstatement']['mean']:.2f}×",
                      delta=f"± {cicids['aggregate']['overstatement']['std']:.3f}",
                      delta_color="off",
                      help="The headline does not generalise. See the Second "
                           "dataset tab for why that is the finding, not a "
                           "failure.")
        elif sweep:
            c4.metric("Never evadable",
                      pct(sweep["headline"]["certifiably_unevadable_fraction"]),
                      help="Attacks whose locked features expose them at any budget.")

        st.info(
            "**What holds and what does not.** Unconstrained evaluation produces "
            "physically impossible traffic on both datasets — that replicates. "
            "That it *inflates the measured vulnerability* by 20× is specific to "
            "NSL-KDD; on CICIDS2017 the same test gives 1.05×.",
            icon="🔎",
        )

    elif sweep:
        h = sweep["headline"]
        st.caption(
            "LinearSVC, **white-box** (the attacker differentiates the victim "
            "directly), **single split** — run `run_multiseed` for error bars."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attack wins — unconstrained", pct(h["unconstrained_evasion_rate"]))
        c2.metric("Attack wins — constrained", pct(h["constrained_evasion_rate"]))
        c3.metric("Vulnerability overstated by", f"{h['overstatement_factor']:.1f}×")
        c4.metric("Never evadable", pct(h["certifiably_unevadable_fraction"]),
                  help="Attacks whose locked features expose them at any budget.")

    if multiseed and packets:
        st.divider()
        st.subheader("Three rungs of realism")
        st.caption(
            "Each rung holds the attacker to a stricter definition of what it can "
            "actually do. CICIDS2017 FTP-Patator flows pulled from the raw capture."
        )
        best = max(r["evasion"] for r in packets["realized"])
        r1, r2, r3 = st.columns(3)
        r1.metric("1 · Unconstrained feature space",
                  f"{packets['feature_space']['unconstrained']:.0%}")
        r2.metric("2 · Constrained feature space",
                  f"{packets['feature_space']['constrained']:.0%}")
        r3.metric("3 · Realized in real packets", f"{best:.0%}")

    if full:
        st.divider()
        st.subheader("Clean performance, before any attack")
        clean = pd.DataFrame([
            {
                "Model": n,
                "Detection rate": e["clean"]["DR"],
                "False positive rate": e["clean"]["FPR"],
                "Balanced accuracy": e["clean"]["Balanced Accuracy"],
                "Substitute fit": e.get("substitute_agreement", float("nan")),
            }
            for n, e in full["results"].items()
        ])
        st.dataframe(
            clean.style.format({
                "Detection rate": "{:.2%}",
                "False positive rate": "{:.2%}",
                "Balanced accuracy": "{:.2%}",
                "Substitute fit": "{:.2%}",
            }),
            width="stretch", hide_index=True,
        )

        st.caption(
            "Every victim is attacked through its own substitute - a model fitted "
            "to its outputs - so all four face an identical attack. Substitute fit "
            "is how often that stand-in agrees with the real model."
        )

        st.divider()
        st.subheader("Alert volume at operational scale")
        st.caption(
            "False positive rate translated to daily alerts, assuming 1M flows/day "
            "at 90% benign. A low percentage is still a large number."
        )
        vol = clean.assign(**{
            "Alerts per day": (clean["False positive rate"] * 900_000).round().astype(int)
        })[["Model", "False positive rate", "Alerts per day"]]
        st.dataframe(
            vol.style.format({"False positive rate": "{:.2%}", "Alerts per day": "{:,}"}),
            width="stretch", hide_index=True,
        )

# ----------------------------------------------------------------- attacks
with T["Attacks"]:
    if not full:
        st.info("Run `run_full_matrix` to populate this tab.")
    else:
        st.subheader("How often the attack succeeds")
        rows = []
        for model, e in full["results"].items():
            for atk, r in e["attacks"].items():
                for regime in ("unconstrained", "constrained"):
                    rows.append({
                        "Model": model, "Attack": atk,
                        "Evaluation": regime.capitalize(),
                        "Evasion rate": r["black_box"][regime]["no_defense"],
                    })
        df = pd.DataFrame(rows)
        st.altair_chart(
            bar(df, "Attack", "Evasion rate", "Evaluation",
                "Evasion rate by attack and evaluation regime",
                "Attack success", facet="Model", sort=ATTACK_ORDER),
            use_container_width=True,
        )
        st.caption(
            "FGSM takes one step, BIM many small steps, PGD adds random restarts. "
            "Strength should increase left to right — most visible on the tree models."
        )

        st.divider()
        st.subheader("How much the unconstrained test overstates the danger")
        over = []
        for model, e in full["results"].items():
            for atk, r in e["attacks"].items():
                c = r["black_box"]["constrained"]["no_defense"]
                u = r["black_box"]["unconstrained"]["no_defense"]
                over.append({
                    "Model": model, "Attack": atk,
                    "Overstatement": (u / c) if c > 0 else None,
                })
        od = pd.DataFrame(over).dropna()
        st.altair_chart(
            alt.Chart(od).mark_bar().encode(
                x=alt.X("Attack:N", title=None, sort=ATTACK_ORDER,
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Overstatement:Q", title="× overstated"),
                color=alt.Color("Model:N", scale=alt.Scale(scheme="tableau10")),
                xOffset="Model:N",
                tooltip=["Model", "Attack", alt.Tooltip("Overstatement", format=".1f")],
            ).properties(height=320),
            use_container_width=True,
        )

# ----------------------------------------------------- detection impact
with T["Detection impact"]:
    if not full:
        st.info("Run `run_full_matrix` to populate this tab.")
    else:
        st.subheader("Detection rate: clean vs. under attack")
        rows = []
        for model, e in full["results"].items():
            for atk, r in e["attacks"].items():
                rows.append({"Model": model, "Attack": atk,
                             "Condition": "1 · Clean", "Detection rate": e["clean"]["DR"]})
                rows.append({"Model": model, "Attack": atk,
                             "Condition": "2 · Unconstrained attack",
                             "Detection rate": r["black_box"]["unconstrained"]["DR_after_attack"]})
                rows.append({"Model": model, "Attack": atk,
                             "Condition": "3 · Constrained attack",
                             "Detection rate": r["black_box"]["constrained"]["DR_after_attack"]})
        df = pd.DataFrame(rows)
        st.altair_chart(
            bar(df, "Attack", "Detection rate", "Condition",
                "What survives the attack", "Detection rate", facet="Model",
                sort=ATTACK_ORDER),
            use_container_width=True,
        )
        st.caption(
            "The gap between bars 2 and 3 is the finding: the collapse reported in "
            "the literature largely disappears once the attacker is held to what a "
            "real network flow can contain."
        )

        st.divider()
        st.subheader("Ranking inversion")
        worst = []
        for model, e in full["results"].items():
            drs_u = [r["black_box"]["unconstrained"]["DR_after_attack"] for r in e["attacks"].values()]
            drs_c = [r["black_box"]["constrained"]["DR_after_attack"] for r in e["attacks"].values()]
            worst.append({
                "Model": model,
                "Clean": e["clean"]["DR"],
                "Worst case — unconstrained": min(drs_u),
                "Worst case — constrained": min(drs_c),
                "Points lost (constrained)": e["clean"]["DR"] - min(drs_c),
            })
        wd = pd.DataFrame(worst).sort_values("Points lost (constrained)", ascending=False)
        st.dataframe(
            wd.style.format({c: "{:.2%}" for c in wd.columns if c != "Model"}),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Sorted by how much each model actually loses under a realistic attack. "
            "The ordering here differs from the unconstrained column — models that "
            "look safest under the standard protocol are not the most robust."
        )

# ----------------------------------------------------------- budget sweep
with T["Attack budget"]:
    if not sweep:
        st.info("Run `run_constrained_comparison` to populate this tab.")
    else:
        st.subheader("Attack success vs. attacker budget")
        sw = pd.DataFrame(sweep["budget_sweep"])
        long = sw.melt(
            id_vars="epsilon",
            value_vars=["constrained_evasion_rate", "unconstrained_evasion_rate"],
            var_name="Evaluation", value_name="Evasion rate",
        ).replace({
            "constrained_evasion_rate": "Constrained",
            "unconstrained_evasion_rate": "Unconstrained",
        })
        st.altair_chart(
            alt.Chart(long).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("epsilon:Q", title="Attack budget (ε, standard deviations)"),
                y=alt.Y("Evasion rate:Q", title="Attack success",
                        axis=alt.Axis(format=".0%")),
                color=alt.Color("Evaluation:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=["epsilon", "Evaluation",
                         alt.Tooltip("Evasion rate", format=".2%")],
            ).properties(height=360),
            use_container_width=True,
        )
        st.caption(
            "The gap is widest at small budgets — the regime a realistic attacker "
            "operates in — and closes at large ones, where an attacker free to "
            "rewrite any feature can evade anything. That is why unconstrained "
            "evaluation is most misleading exactly where it matters."
        )
        st.altair_chart(
            alt.Chart(sw).mark_bar().encode(
                x=alt.X("epsilon:O", title="Attack budget (ε)",
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("overstatement_factor:Q", title="× overstated"),
                tooltip=["epsilon",
                         alt.Tooltip("overstatement_factor", format=".1f")],
            ).properties(height=260, title="Overstatement shrinks as the attacker gets stronger"),
            use_container_width=True,
        )

# ---------------------------------------------------------------- defenses
with T["Defenses"]:
    if not full:
        st.info("Run `run_full_matrix` to populate this tab.")
    else:
        st.subheader("Defenses against the strongest realistic attack")
        rows = []
        for model, e in full["results"].items():
            c = e["attacks"]["PGD"]["black_box"]["constrained"]
            hardened = evasion_after_hardening(full["adversarial_training"][model])
            for label, val in (
                ("No defense", c["no_defense"]),
                ("Feature squeezing", c["feature_squeezing"]),
                ("Realizability filter", c["realizability_filter"]),
                ("Adversarial training", hardened),
            ):
                if val is not None:
                    rows.append({"Model": model, "Defense": label, "Evasion rate": val})
        st.altair_chart(
            alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                x=alt.X("Defense:N", title=None, axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("Evasion rate:Q", axis=alt.Axis(format=".0%")),
                color=alt.Color("Defense:N", scale=alt.Scale(scheme="tableau10"),
                                legend=None),
                tooltip=["Model", "Defense",
                         alt.Tooltip("Evasion rate", format=".2%")],
            ).properties(width=230, height=240).facet(
                facet=alt.Facet("Model:N", title=None), columns=2),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Realizability filter vs. unconstrained attacks")
        rows = []
        for model, e in full["results"].items():
            for atk, r in e["attacks"].items():
                rows.append({"Model": model, "Attack": atk, "Defense": "None",
                             "Evasion rate": r["black_box"]["unconstrained"]["no_defense"]})
                rows.append({"Model": model, "Attack": atk, "Defense": "Filter",
                             "Evasion rate": r["black_box"]["unconstrained"]["realizability_filter"]})
        st.altair_chart(
            bar(pd.DataFrame(rows), "Attack", "Evasion rate", "Defense",
                "Impossible traffic is rejected outright", "Attack success",
                facet="Model", sort=ATTACK_ORDER),
            use_container_width=True,
        )
        st.warning(
            "The filter eliminates unconstrained attacks completely and does "
            "**nothing** against constrained ones. It is not robustness — it is a "
            "forcing function that removes the attacker's impossible options.",
            icon="ℹ️",
        )

        st.divider()
        st.subheader("Cost of adversarial training")
        rows = []
        for model, e in full["results"].items():
            a = full["adversarial_training"][model]["clean"]
            rows.append({
                "Model": model,
                "DR before": e["clean"]["DR"], "DR after": a["DR"],
                "FPR before": e["clean"]["FPR"], "FPR after": a["FPR"],
            })
        cd = pd.DataFrame(rows)
        st.dataframe(
            cd.style.format({c: "{:.2%}" for c in cd.columns if c != "Model"}),
            width="stretch", hide_index=True,
        )
        st.caption("Hardening is close to free on clean traffic — detection and "
                   "false-positive rates barely move.")

# ------------------------------------------------------------- constraints
with T["Constraints"]:
    if not sweep:
        st.info("Run `run_constrained_comparison` to populate this tab.")
    else:
        st.subheader("What the attacker is allowed to touch")
        roles = pd.DataFrame([
            {"Role": ROLE_LABELS.get(k, k), "Features": v}
            for k, v in sweep["feature_roles"].items()
        ])
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(roles, width="stretch", hide_index=True)
        with col2:
            st.altair_chart(
                alt.Chart(roles).mark_arc(innerRadius=60).encode(
                    theta="Features:Q",
                    color=alt.Color("Role:N", scale=alt.Scale(scheme="tableau10")),
                    tooltip=["Role", "Features"],
                ).properties(height=260),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Impossible values produced by each evaluation regime")
        rows = []
        for key, label in (("fgsm_unconstrained", "Unconstrained"),
                           ("fgsm_constrained", "Constrained")):
            v = sweep["sections"][key].get("realizability_violations", {})
            for rule, count in v.items():
                rows.append({
                    "Rule broken": rule.replace("_", " ").capitalize(),
                    "Evaluation": label, "Features affected": count,
                })
        st.altair_chart(
            alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                x=alt.X("Features affected:Q"),
                y=alt.Y("Rule broken:N", title=None, sort="-x"),
                color=alt.Color("Evaluation:N", scale=alt.Scale(scheme="tableau10")),
                yOffset="Evaluation:N",
                tooltip=["Rule broken", "Evaluation", "Features affected"],
            ).properties(height=300),
            use_container_width=True,
        )
        st.caption(
            "The unconstrained attack produces negative byte counts, fractional "
            "protocol identifiers and out-of-range percentages. The constrained "
            "attack produces none — every sample it emits could be captured on a "
            "real network."
        )

# -------------------------------------------------------------- error bars
with T["Error bars"]:
    if not multiseed:
        st.info("Run `run_multiseed` to populate this tab.")
    else:
        seeds = multiseed["seeds"]
        st.subheader(f"Every attack figure, repeated over {len(seeds)} splits")
        st.caption(
            "Bars are means, the black rule spans one standard deviation. Seeds "
            + ", ".join(str(s) for s in seeds)
            + ". A result that only exists on one split is not a result."
        )

        rows = []
        for model, agg in multiseed["aggregate"].items():
            for key, label in (("unconstrained_evasion", "Unconstrained"),
                               ("constrained_evasion", "Constrained")):
                m, lo, hi = band(agg[key], lo=0.0, hi=1.0)
                rows.append({"Model": model, "Evaluation": label,
                             "Evasion rate": m, "lo": lo, "hi": hi})
        ev = pd.DataFrame(rows)
        base = alt.Chart(ev).encode(
            x=alt.X("Evaluation:N", title=None, axis=alt.Axis(labelAngle=0)))
        st.altair_chart(
            (base.mark_bar().encode(
                y=alt.Y("Evasion rate:Q", title="Attack success",
                        axis=alt.Axis(format=".0%"),
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Evaluation:N",
                                scale=alt.Scale(scheme="tableau10"), legend=None),
                tooltip=["Model", "Evaluation",
                         alt.Tooltip("Evasion rate", format=".2%"),
                         alt.Tooltip("lo", format=".2%"),
                         alt.Tooltip("hi", format=".2%")])
             + base.mark_rule(strokeWidth=2, color="#222").encode(
                 y=alt.Y("lo:Q", title=""), y2="hi:Q")
             ).properties(width=190, height=250).facet(
                facet=alt.Facet("Model:N", title=None), columns=4),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Which conclusions survive repetition")
        rows = []
        for model, agg in multiseed["aggregate"].items():
            o = agg["overstatement"]
            spread = agg["constrained_evasion"]["values"]
            rows.append({
                "Model": model,
                "Overstated by": f"{o['mean']:.1f} ± {o['std']:.1f}×",
                "Constrained evasion, worst split": max(spread),
                "Constrained evasion, best split": min(spread),
                "Stable?": "yes" if o["std"] / max(o["mean"], 1e-9) < 0.25 else "NO",
            })
        st.dataframe(
            pd.DataFrame(rows).style.format({
                "Constrained evasion, worst split": "{:.1%}",
                "Constrained evasion, best split": "{:.1%}",
            }),
            width="stretch", hide_index=True,
        )
        st.warning(
            "**The overstatement claim holds for the differentiable models and "
            "not for the trees.** RandomForest and XGBoost swing so far between "
            "splits that their factors have no stable centre — reporting either "
            "as a single number would be an artefact of the split that produced "
            "it.",
            icon="⚠️",
        )

        if treevar:
            st.divider()
            st.subheader("Two explanations for the tree swing, both rejected")
            tv = pd.DataFrame(treevar["per_run"])
            tv = tv[tv["model"].isin(["RandomForest", "XGBoost"])]
            left, right = st.columns(2)
            for col, xcol, label in (
                (left, "substitute_agreement_attacked",
                 "Substitute fidelity — how well the stand-in matches the victim"),
                (right, "movable_feature_importance",
                 "Importance concentrated in features the attacker may move"),
            ):
                with col:
                    st.altair_chart(
                        alt.Chart(tv).mark_point(size=140, filled=True).encode(
                            x=alt.X(f"{xcol}:Q", title=label,
                                    scale=alt.Scale(zero=False),
                                    axis=alt.Axis(format=".1%")),
                            y=alt.Y("constrained_evasion:Q",
                                    title="Constrained evasion",
                                    axis=alt.Axis(format=".0%")),
                            color=alt.Color("model:N", title=None,
                                            scale=alt.Scale(scheme="tableau10")),
                            tooltip=["seed", "model",
                                     alt.Tooltip(xcol, format=".3%"),
                                     alt.Tooltip("constrained_evasion", format=".1%")],
                        ).properties(height=300),
                        use_container_width=True,
                    )
            st.caption(
                "If either explained the swing, the points would trend. Neither "
                "does: substitute agreement varies by half a percentage point "
                "across splits whose evasion rates differ by forty-five. The "
                "cause is recorded as unexplained rather than guessed at — "
                "resolving it needs 20+ seeds."
            )

# ---------------------------------------------------------- second dataset
with T["Second dataset"]:
    if not cicids:
        st.info("Run `run_cicids_audit` to populate this tab.")
    else:
        ca = cicids["aggregate"]
        st.subheader("The same engine, a different schema")
        st.caption(
            f"{cicids['n_flows']:,} CICIDS2017 flows, {cicids['n_features']} "
            f"features, {len(cicids['seeds'])} splits. Adding this dataset was a "
            "YAML specification file — `specs/cicids2017.yaml` — not a code change."
        )

        comparison = [{
            "Measure": "Unconstrained evasion",
            "NSL-KDD": None, "CICIDS2017": ca["unconstrained_evasion"]["mean"],
        }, {
            "Measure": "Constrained evasion",
            "NSL-KDD": None, "CICIDS2017": ca["constrained_evasion"]["mean"],
        }]
        if multiseed:
            nk = multiseed["aggregate"]["LinearSVC"]
            comparison[0]["NSL-KDD"] = nk["unconstrained_evasion"]["mean"]
            comparison[1]["NSL-KDD"] = nk["constrained_evasion"]["mean"]
        cmp_df = pd.DataFrame(comparison)
        long = cmp_df.melt(id_vars="Measure", var_name="Dataset",
                           value_name="Evasion rate").dropna()
        c1, c2 = st.columns([3, 2])
        with c1:
            st.altair_chart(
                alt.Chart(long).mark_bar().encode(
                    x=alt.X("Dataset:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Evasion rate:Q", axis=alt.Axis(format=".0%"),
                            scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Dataset:N",
                                    scale=alt.Scale(scheme="tableau10"), legend=None),
                    # Unconstrained first: the narrative is what constraining
                    # takes away, and alphabetical order reverses it.
                    column=alt.Column("Measure:N", title=None,
                                      sort=["Unconstrained evasion",
                                            "Constrained evasion"]),
                    tooltip=["Dataset", "Measure",
                             alt.Tooltip("Evasion rate", format=".2%")],
                ).properties(width=170, height=280),
                use_container_width=False,
            )
        with c2:
            st.metric("Overstatement — NSL-KDD",
                      f"{multiseed['aggregate']['LinearSVC']['overstatement']['mean']:.1f}×"
                      if multiseed else "n/a")
            st.metric("Overstatement — CICIDS2017",
                      f"{ca['overstatement']['mean']:.2f}×",
                      delta=f"± {ca['overstatement']['std']:.3f}",
                      delta_color="off")

        st.error(
            "**Constraining the attacker barely helps here, and the evasions that "
            "survive are fully realizable — zero violations of any kind.** That "
            "detector genuinely is that vulnerable to traffic a real adversary "
            "could send.",
            icon="🚨",
        )

        st.markdown(
            "**This separates two claims NSL-KDD had fused together:**\n\n"
            "- *Unconstrained evaluation produces physically impossible traffic.* "
            "Replicates on both datasets. Appears universal.\n"
            "- *Constraining substantially reduces measured evasion.* Does **not** "
            "replicate.\n\n"
            "NSL-KDD's 15 rate features are bounded to [0, 1], which boxes an "
            "attacker in. CICIDS2017's 78 columns leave far more room, so the "
            "attacker evades regardless of what is locked. **How much domain "
            "constraints protect a detector is a property of the schema, not a "
            "constant.**"
        )

        st.divider()
        st.subheader("Impossible values, CICIDS2017")
        first = cicids["per_seed"][str(cicids["seeds"][0])]
        rows = []
        for key, label in (("unconstrained_violations", "Unconstrained"),
                           ("constrained_violations", "Constrained")):
            for rule, count in first[key].items():
                rows.append({"Rule broken": rule.replace("_", " ").capitalize(),
                             "Evaluation": label, "Features affected": count})
        st.altair_chart(
            alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                x=alt.X("Features affected:Q"),
                y=alt.Y("Rule broken:N", title=None, sort="-x"),
                color=alt.Color("Evaluation:N", scale=alt.Scale(scheme="tableau10")),
                yOffset="Evaluation:N",
                tooltip=["Rule broken", "Evaluation", "Features affected"],
            ).properties(height=260),
            use_container_width=True,
        )
        st.caption(
            "59 of 70 features driven negative, and every one of the 16 packet-length "
            "and inter-arrival ordering constraints violated — flows whose smallest "
            "packet exceeds their largest. The constrained attack breaks none."
        )

# ------------------------------------------------------- packet round-trip
with T["Packet round-trip"]:
    if not packets:
        st.info("Run `run_packet_roundtrip` to populate this tab.")
    else:
        st.subheader("From real packets to features and back")
        st.caption(
            f"{packets['n_flows_extracted']} FTP-Patator flows lifted out of the "
            "10.5 GB Tuesday capture, perturbed with operations an attacker can "
            "actually perform — padding forward payloads and delaying forward "
            "packets — then re-extracted and re-tested."
        )

        best = max(r["evasion"] for r in packets["realized"])
        order = ["1 · Unconstrained feature space",
                 "2 · Constrained feature space",
                 "3 · Realized in packets"]
        rungs = pd.DataFrame([
            {"Rung": order[0], "Evasion": packets["feature_space"]["unconstrained"]},
            {"Rung": order[1], "Evasion": packets["feature_space"]["constrained"]},
            {"Rung": order[2], "Evasion": best},
        ])
        rung_base = alt.Chart(rungs).encode(
            x=alt.X("Rung:N", title=None, sort=order,
                    axis=alt.Axis(labelAngle=0, labelLimit=220)),
        )
        st.altair_chart(
            (rung_base.mark_bar(size=90).encode(
                y=alt.Y("Evasion:Q", title="Attack success",
                        axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Rung:N", sort=order,
                                scale=alt.Scale(scheme="tableau10"), legend=None),
                tooltip=["Rung", alt.Tooltip("Evasion", format=".1%")])
             # The third rung is 0%, which draws no bar at all. Without the
             # printed value it reads as missing data rather than as the result.
             + rung_base.mark_text(dy=-10, fontSize=15, fontWeight="bold",
                                   color="#888").encode(
                 y=alt.Y("Evasion:Q"), text=alt.Text("Evasion:Q", format=".0%"))
             ).properties(height=330),
            use_container_width=True,
        )
        st.success(
            "**Even constrained feature-space evaluation overstates what an "
            "attacker limited to real packet operations achieves.** The gap "
            "between rungs 2 and 3 is the measurement that does not exist in this "
            "literature.",
            icon="✅",
        )

        st.divider()
        st.subheader("Attacker budgets tried")
        bud = pd.DataFrame(packets["realized"]).rename(columns={
            "budget": "Budget", "pad_bytes": "Padding (bytes)",
            "delay_s": "Delay (s)", "evasion": "Evasion"})
        st.dataframe(bud.style.format({"Evasion": "{:.1%}", "Delay (s)": "{:.3f}"}),
                     width="stretch", hide_index=True)
        st.caption(
            "Zero at every budget. The perturbations are not inert — the largest "
            "budget moves 30 of 78 features, taking Average Packet Size from 7.4 "
            "to 169.1 — the detector holds anyway."
        )

        st.divider()
        st.subheader("How far to trust this")
        ag = packets["extractor_agreement"]
        n_ok = sum(1 for v in ag.values() if v["agrees"])
        st.metric("Extractor agreement with the published CICFlowMeter rows",
                  f"{n_ok} of {len(ag)} features")
        st.dataframe(
            pd.DataFrame([
                {"Feature": f, "Ours (median)": v["mine_median"],
                 "Published (median)": v["published_median"],
                 "Relative difference": v["rel_diff"],
                 "Agrees": "yes" if v["agrees"] else "no"}
                for f, v in ag.items()
            ]).style.format({
                "Ours (median)": "{:,.1f}", "Published (median)": "{:,.1f}",
                "Relative difference": "{:.1%}",
            }),
            width="stretch", hide_index=True,
        )
        st.warning(
            "**Indicative, not definitive.** This extractor is a reimplementation, "
            "not CICFlowMeter — the official tool is Java over `jnetpcap` and was "
            "impractical to build here. It reads high on durations and packet "
            "counts. The released CSVs strip IP addresses, so no extracted flow "
            "can be matched to its published counterpart and the disagreement "
            "cannot be resolved directly. The same extractor produces both sides "
            "of the comparison, so the *relative* measurement is sounder than the "
            "absolute fidelity suggests.",
            icon="⚠️",
        )
