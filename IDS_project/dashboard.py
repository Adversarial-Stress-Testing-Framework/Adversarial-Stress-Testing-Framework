"""
Audit dashboard - visualises every test the stress-testing framework runs.

Reads the JSON reports produced by the experiment runners; it does not retrain
or re-attack anything, so it opens instantly. Regenerate the underlying data
with:

    python -m stress_test.run_full_matrix
    python -m stress_test.run_constrained_comparison

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


full = load(FULL)
sweep = load(SWEEP)
legacy = load(LEGACY)

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

tabs = st.tabs([
    "Overview", "Attacks", "Detection impact", "Attack budget",
    "Defenses", "Constraints",
])

# ----------------------------------------------------------------- overview
with tabs[0]:
    st.subheader("Headline findings")

    if sweep:
        h = sweep["headline"]
        st.caption(
            "LinearSVC, **white-box** (the attacker differentiates the victim "
            "directly). The Attacks tab reports the black-box route, where every "
            "model is attacked through its own substitute — those figures differ "
            "slightly by design, not by error."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attack wins — unconstrained", pct(h["unconstrained_evasion_rate"]))
        c2.metric("Attack wins — constrained", pct(h["constrained_evasion_rate"]))
        c3.metric("Vulnerability overstated by", f"{h['overstatement_factor']:.1f}×")
        c4.metric("Never evadable", pct(h["certifiably_unevadable_fraction"]),
                  help="Attacks whose locked features expose them at any budget.")

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
with tabs[1]:
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
with tabs[2]:
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
with tabs[3]:
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
with tabs[4]:
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
with tabs[5]:
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
