import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("database/alberta_wmu_survey_database_2025.json")
APP_TITLE = "Alberta WMU Hunting Effort vs Reward Dashboard"
APP_SUBTITLE = "Interactive WMU explorer with species tabs, scoring, and parallel-coordinates analysis"
LOGO_PATH = "docs/assets/logo1-r.png"
# Basic weighting formula for a global success score.
# Positive metrics push success up; effort pushes it down.
WEIGHTS = {
    "density": 0.35,
    "abundance": 0.20,
    "observed": 0.15,
    "trend": 0.10,
    "sex_ratio": 0.10,
    "juvenile_ratio": 0.10,
}

EFFORT_PENALTY_WEIGHT = 0.45


# -----------------------------
# Helpers
# -----------------------------
def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prettify_species_name(name: str) -> str:
    return name.replace("_", " ").title()


def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "non-public-data available":
            return None
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_series(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series([np.nan] * len(s), index=s.index)

    min_v, max_v = valid.min(), valid.max()
    if np.isclose(min_v, max_v):
        norm = pd.Series(
            [0.5 if pd.notna(v) else np.nan for v in s], index=s.index)
    else:
        norm = (s - min_v) / (max_v - min_v)

    if not higher_is_better:
        norm = 1 - norm
    return norm.clip(0, 1)


def compute_trend_score(trend_direction: Optional[str], trend_percent: Optional[float]) -> float:
    direction = (trend_direction or "").lower().strip()
    pct = safe_float(trend_percent)

    if direction == "increase":
        if pct is None:
            return 0.75
        return min(1.0, 0.50 + pct / 200.0)
    if direction == "stable":
        return 0.60
    if direction == "decrease":
        if pct is None:
            return 0.35
        return max(0.0, 0.50 - pct / 100.0)
    return np.nan


def build_species_records(raw: Dict) -> pd.DataFrame:
    rows: List[Dict] = []
    wmus = raw.get("wmus", {})

    for wmu_id, wmu_data in wmus.items():
        survey = wmu_data.get("survey", {})
        terrain = wmu_data.get("terrain", {})
        species_map = wmu_data.get("species", {})

        for species_name, sp in species_map.items():
            rows.append(
                {
                    "wmu_id": str(wmu_id),
                    "species": species_name,
                    "species_label": prettify_species_name(species_name),
                    "year": wmu_data.get("year"),
                    "source_pdf": wmu_data.get("source_pdf"),
                    "area_km2": safe_float(wmu_data.get("area_km2")),
                    "survey_method": survey.get("method"),
                    "survey_effort_km": safe_float(survey.get("effort_km")),
                    "coverage_percent": safe_float(survey.get("coverage_percent")),
                    "area_surveyed_km2": safe_float(survey.get("area_surveyed_km2")),
                    "terrain_description": terrain.get("description"),
                    "mean_average_elevation_m": safe_float(terrain.get("mean_average_elevation_m")),
                    "hydro_water_percent": safe_float(terrain.get("hydro_water_percent")),
                    "observed_count": safe_float(sp.get("observed_count")),
                    "minimum_total_count": safe_float(sp.get("minimum_total_count")),
                    "average_group_size": safe_float(sp.get("average_group_size")),
                    "group_size_min": safe_float((sp.get("group_size_range") or [None, None])[0]),
                    "group_size_max": safe_float((sp.get("group_size_range") or [None, None])[1]),
                    "density_per_km2": safe_float(sp.get("density_per_km2")),
                    "abundance_estimate": safe_float(sp.get("abundance_estimate")),
                    "ci90_low": safe_float((sp.get("confidence_interval_90") or [None, None])[0]),
                    "ci90_high": safe_float((sp.get("confidence_interval_90") or [None, None])[1]),
                    "male_per_100_female": safe_float(sp.get("male_per_100_female")),
                    "juvenile_per_100_female": safe_float(sp.get("juvenile_per_100_female")),
                    "trend_summary": sp.get("trend_summary"),
                    "trend_direction": sp.get("trend_direction"),
                    "trend_percent": safe_float(sp.get("trend_percent")),
                    "trend_reason": sp.get("trend_reason"),
                    "derived_effort_km_per_expected_animal": safe_float(
                        sp.get("derived_effort_km_per_expected_animal")
                    ),
                }
            )

    df = pd.DataFrame(rows)

    # Fallback density for species like elk when abundance and area are available but density is missing.
    density_fallback = df["abundance_estimate"] / df["area_km2"]
    df["density_proxy_per_km2"] = df["density_per_km2"].fillna(
        density_fallback)

    # Fallback effort metric from survey effort / abundance.
    effort_fallback = df["survey_effort_km"] / df["abundance_estimate"]
    df["effort_proxy_km"] = df["derived_effort_km_per_expected_animal"].fillna(
        effort_fallback)

    # Additional rate metrics useful for charts.
    df["observed_per_100km_survey"] = np.where(
        df["survey_effort_km"].notna() & (df["survey_effort_km"] > 0),
        (df["observed_count"] / df["survey_effort_km"]) * 100,
        np.nan,
    )

    df["abundance_per_km2_wmu"] = np.where(
        df["area_km2"].notna() & (df["area_km2"] > 0),
        df["abundance_estimate"] / df["area_km2"],
        df["density_proxy_per_km2"],
    )

    df["trend_score_raw"] = df.apply(
        lambda r: compute_trend_score(r["trend_direction"], r["trend_percent"]), axis=1
    )

    return df


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["density_norm"] = normalize_series(
        out["density_proxy_per_km2"], higher_is_better=True)
    out["abundance_norm"] = normalize_series(
        out["abundance_estimate"], higher_is_better=True)
    out["observed_norm"] = normalize_series(
        out["observed_count"], higher_is_better=True)
    out["trend_norm"] = normalize_series(
        out["trend_score_raw"], higher_is_better=True)
    out["sex_ratio_norm"] = normalize_series(
        out["male_per_100_female"], higher_is_better=True)
    out["juvenile_ratio_norm"] = normalize_series(
        out["juvenile_per_100_female"], higher_is_better=True)
    out["effort_norm_red"] = normalize_series(
        out["effort_proxy_km"], higher_is_better=False)
    out["effort_raw_norm"] = normalize_series(
        out["effort_proxy_km"], higher_is_better=True)

    positive = (
        WEIGHTS["density"] * out["density_norm"].fillna(0)
        + WEIGHTS["abundance"] * out["abundance_norm"].fillna(0)
        + WEIGHTS["observed"] * out["observed_norm"].fillna(0)
        + WEIGHTS["trend"] * out["trend_norm"].fillna(0)
        + WEIGHTS["sex_ratio"] * out["sex_ratio_norm"].fillna(0)
        + WEIGHTS["juvenile_ratio"] * out["juvenile_ratio_norm"].fillna(0)
    )

    # Success score: positive ecological / survey indicators minus effort burden.
    out["global_success_score"] = (
        100
        * (
            positive
            + EFFORT_PENALTY_WEIGHT * out["effort_norm_red"].fillna(0)
        )
        / (sum(WEIGHTS.values()) + EFFORT_PENALTY_WEIGHT)
    ).round(1)

    out["global_effort_score"] = (
        100 * out["effort_raw_norm"].fillna(0)).round(1)
    out["success_minus_effort"] = (
        out["global_success_score"] - out["global_effort_score"]).round(1)

    return out


def get_wmu_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("wmu_id", as_index=False)
        .agg(
            overall_success=("global_success_score", "mean"),
            overall_effort=("global_effort_score", "mean"),
            avg_density=("density_proxy_per_km2", "mean"),
            avg_effort_km=("effort_proxy_km", "mean"),
            total_observed=("observed_count", "sum"),
            avg_abundance=("abundance_estimate", "mean"),
            n_species=("species", "nunique"),
        )
        .sort_values(["overall_success", "overall_effort"], ascending=[False, True])
    )
    return agg


def build_parallel_species_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="wmu_id", columns="species_label", values=metric, aggfunc="mean")
    pivot = pivot.reset_index()
    return pivot


def metric_card(label: str, value, help_text: Optional[str] = None, fmt: str = "{:.2f}"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        display = "N/A"
    elif isinstance(value, (float, np.floating, int, np.integer)):
        display = fmt.format(value)
    else:
        display = str(value)
    st.metric(label, display, help=help_text)


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide",
                   page_icon=LOGO_PATH)
# Display logo at the top
if Path(LOGO_PATH).exists():
    st.image(LOGO_PATH, width=200)
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.markdown(
    """
This dashboard uses the uploaded Alberta WMU 2025 survey JSON to visualize species-level density,
proxy effort, abundance, and a simple combined success-vs-effort score.

**Global score formula used in this first version**
- Positive factors: density, abundance estimate, observed count, trend, male ratio, juvenile ratio
- Negative factor: effort proxy in km per expected animal
- `global_success_score` is scaled to 0-100 and rewards stronger wildlife indicators while penalizing effort.
- `global_effort_score` is scaled to 0-100, where higher means harder / more effort.

This is a first-pass analytical score, not an official harvest success model.
"""
)

if not DATA_PATH.exists():
    st.error(
        f"Could not find `{DATA_PATH}`. Place the JSON file in the same folder as this Streamlit script."
    )
    st.stop()

raw_data = load_json(DATA_PATH)
df = compute_scores(build_species_records(raw_data))
wmu_summary = get_wmu_summary(df)
all_wmus = sorted(df["wmu_id"].unique(), key=lambda x: int(x))

with st.sidebar:
    st.header("Controls")
    selected_wmu = st.selectbox("Select WMU", all_wmus, index=all_wmus.index(
        "306") if "306" in all_wmus else 0)
    sort_metric = st.selectbox(
        "Sort species ranking by",
        [
            "global_success_score",
            "global_effort_score",
            "density_proxy_per_km2",
            "abundance_estimate",
            "observed_count",
            "effort_proxy_km",
        ],
        index=0,
    )
    show_all_parallel = st.checkbox(
        "Use all WMUs in parallel-coordinates charts", value=True)

selected_df = df[df["wmu_id"] == selected_wmu].copy(
).sort_values(sort_metric, ascending=False)
selected_meta = raw_data["wmus"][selected_wmu]

# -----------------------------
# WMU Header
# -----------------------------
st.subheader(f"WMU {selected_wmu} overview")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    metric_card("Species in WMU",
                selected_df["species"].nunique(), fmt="{:.0f}")
with col2:
    metric_card("Avg success score",
                selected_df["global_success_score"].mean())
with col3:
    metric_card("Avg effort score", selected_df["global_effort_score"].mean())
with col4:
    metric_card("Survey effort (km)", safe_float(
        selected_meta.get("survey", {}).get("effort_km")))
with col5:
    metric_card("WMU area (km²)", safe_float(selected_meta.get("area_km2")))

st.markdown("**Terrain summary**")
st.write(selected_meta.get("terrain", {}).get("description", "N/A"))

if selected_meta.get("background_summary"):
    with st.expander("Background summary"):
        st.write(selected_meta.get("background_summary"))

# -----------------------------
# Cross-species overview for selected WMU
# -----------------------------
st.markdown("## Species comparison inside selected WMU")
overview_cols = [
    "species_label",
    "observed_count",
    "density_proxy_per_km2",
    "abundance_estimate",
    "effort_proxy_km",
    "global_success_score",
    "global_effort_score",
    "trend_direction",
    "trend_percent",
]

st.dataframe(
    selected_df[overview_cols].rename(
        columns={
            "species_label": "Species",
            "observed_count": "Observed",
            "density_proxy_per_km2": "Animals / km²",
            "abundance_estimate": "Abundance",
            "effort_proxy_km": "Avg hunting walk (km)",
            "global_success_score": "Success score",
            "global_effort_score": "Effort score",
            "trend_direction": "Trend",
            "trend_percent": "Trend %",
        }
    ),
    use_container_width=True,
)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig = px.bar(
        selected_df,
        x="species_label",
        y="global_success_score",
        color="global_success_score",
        title=f"WMU {selected_wmu} - Global success score by species",
        labels={"species_label": "Species",
                "global_success_score": "Success score"},
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    fig = px.bar(
        selected_df,
        x="species_label",
        y="global_effort_score",
        color="global_effort_score",
        title=f"WMU {selected_wmu} - Effort score by species",
        labels={"species_label": "Species",
                "global_effort_score": "Effort score"},
    )
    st.plotly_chart(fig, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)
with chart_col3:
    fig = px.scatter(
        selected_df,
        x="effort_proxy_km",
        y="global_success_score",
        size="observed_count",
        color="species_label",
        hover_data=["density_proxy_per_km2", "abundance_estimate"],
        title=f"WMU {selected_wmu} - Success vs average hunting walk",
        labels={
            "effort_proxy_km": "Average hunting walk (km)",
            "global_success_score": "Global success score",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col4:
    fig = px.bar(
        selected_df,
        x="species_label",
        y="density_proxy_per_km2",
        color="species_label",
        title=f"WMU {selected_wmu} - Animals per km²",
        labels={"species_label": "Species",
                "density_proxy_per_km2": "Animals / km²"},
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Per-species tabs
# -----------------------------
st.markdown("## Species tabs")
if selected_df.empty:
    st.warning("No species data available for this WMU.")
else:
    tabs = st.tabs(selected_df["species_label"].tolist())

    for tab, (_, row) in zip(tabs, selected_df.iterrows()):
        with tab:
            st.subheader(row["species_label"])

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                metric_card("Observed count",
                            row["observed_count"], fmt="{:.0f}")
            with m2:
                metric_card("Animals / km²", row["density_proxy_per_km2"])
            with m3:
                metric_card(
                    "Abundance", row["abundance_estimate"], fmt="{:.0f}")
            with m4:
                metric_card("Avg hunting walk (km)", row["effort_proxy_km"])
            with m5:
                metric_card("Success score", row["global_success_score"])
            with m6:
                metric_card("Effort score", row["global_effort_score"])

            d1, d2 = st.columns([1, 1])
            with d1:
                details = pd.DataFrame(
                    {
                        "Metric": [
                            "Average group size",
                            "Group size min",
                            "Group size max",
                            "Male / 100 female",
                            "Juvenile / 100 female",
                            "Trend direction",
                            "Trend percent",
                            "Survey effort (km)",
                            "Area surveyed (km²)",
                        ],
                        "Value": [
                            row["average_group_size"],
                            row["group_size_min"],
                            row["group_size_max"],
                            row["male_per_100_female"],
                            row["juvenile_per_100_female"],
                            row["trend_direction"],
                            row["trend_percent"],
                            row["survey_effort_km"],
                            row["area_surveyed_km2"],
                        ],
                    }
                )
                st.dataframe(details, use_container_width=True)

            with d2:
                comp = pd.DataFrame(
                    {
                        "component": [
                            "Density",
                            "Abundance",
                            "Observed",
                            "Trend",
                            "Male ratio",
                            "Juvenile ratio",
                            "Effort advantage",
                        ],
                        "normalized_value": [
                            row.get("density_norm"),
                            row.get("abundance_norm"),
                            row.get("observed_norm"),
                            row.get("trend_norm"),
                            row.get("sex_ratio_norm"),
                            row.get("juvenile_ratio_norm"),
                            row.get("effort_norm_red"),
                        ],
                    }
                )
                fig = px.bar(
                    comp,
                    x="component",
                    y="normalized_value",
                    color="component",
                    title=f"{row['species_label']} score components",
                    labels={
                        "normalized_value": "Normalized contribution (0-1)"},
                )
                st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                single_df = pd.DataFrame(
                    {
                        "type": ["Success", "Effort"],
                        "score": [row["global_success_score"], row["global_effort_score"]],
                    }
                )
                fig = px.bar(
                    single_df,
                    x="type",
                    y="score",
                    color="type",
                    title=f"{row['species_label']} - success vs effort",
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                radar_values = [
                    row.get("density_norm", 0) or 0,
                    row.get("abundance_norm", 0) or 0,
                    row.get("observed_norm", 0) or 0,
                    row.get("trend_norm", 0) or 0,
                    row.get("effort_norm_red", 0) or 0,
                ]
                radar_labels = ["Density", "Abundance",
                                "Observed", "Trend", "Effort Advantage"]
                radar_values += radar_values[:1]
                radar_labels += radar_labels[:1]
                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(r=radar_values, theta=radar_labels,
                                    fill="toself", name=row["species_label"])
                )
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f"{row['species_label']} profile",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            if pd.notna(row.get("trend_summary")) or pd.notna(row.get("trend_reason")):
                with st.expander("Trend notes"):
                    if pd.notna(row.get("trend_summary")):
                        st.write(f"**Summary:** {row['trend_summary']}")
                    if pd.notna(row.get("trend_reason")):
                        st.write(f"**Reason / note:** {row['trend_reason']}")

# -----------------------------
# All-WMU ranking view
# -----------------------------
st.markdown("## WMU ranking overview")
rank_fig = px.scatter(
    wmu_summary,
    x="overall_effort",
    y="overall_success",
    size="total_observed",
    color="n_species",
    hover_name="wmu_id",
    title="WMU success vs effort overview",
    labels={
        "overall_effort": "Overall effort score",
        "overall_success": "Overall success score",
        "n_species": "Species count",
    },
)
st.plotly_chart(rank_fig, use_container_width=True)

st.dataframe(
    wmu_summary.rename(
        columns={
            "wmu_id": "WMU",
            "overall_success": "Overall success",
            "overall_effort": "Overall effort",
            "avg_density": "Avg animals / km²",
            "avg_effort_km": "Avg hunting walk (km)",
            "total_observed": "Total observed",
            "avg_abundance": "Avg abundance",
            "n_species": "Species count",
        }
    ),
    use_container_width=True,
)

# -----------------------------
# Parallel coordinates section
# -----------------------------
st.markdown("## Parallel coordinates")
st.write(
    "Each chart uses species as axes. Because parallel coordinates work best with one metric family at a time, the app shows one chart per measurable variable."
)

parallel_df = df.copy() if show_all_parallel else selected_df.copy()

parallel_configs = [
    ("global_success_score", "Global success score by species axes"),
    ("global_effort_score", "Global effort score by species axes"),
    ("density_proxy_per_km2", "Animals per km² by species axes"),
    ("effort_proxy_km", "Average hunting walk (km) by species axes"),
    ("abundance_estimate", "Abundance estimate by species axes"),
]

for metric, title in parallel_configs:
    pvt = build_parallel_species_table(parallel_df, metric)
    metric_cols = [c for c in pvt.columns if c != "wmu_id"]
    if len(metric_cols) < 2:
        continue

    color_series = pvt[metric_cols].mean(axis=1, skipna=True)

    fig = px.parallel_coordinates(
        pvt,
        dimensions=metric_cols,
        color=color_series,
        title=title,
        labels={c: c for c in metric_cols},
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Composite green vs red stacked view
# -----------------------------
st.markdown("## Global success vs effort stacked view")
stacked = wmu_summary.copy()
stacked["success_green"] = stacked["overall_success"]
stacked["effort_red"] = stacked["overall_effort"]
stacked = stacked.sort_values("overall_success", ascending=False)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=stacked["wmu_id"],
        y=stacked["success_green"],
        name="Success",
        marker_color="green",
    )
)
fig.add_trace(
    go.Bar(
        x=stacked["wmu_id"],
        y=stacked["effort_red"],
        name="Effort",
        marker_color="red",
    )
)
fig.update_layout(
    barmode="stack",
    title="Stacked success (green) and effort (red) by WMU",
    xaxis_title="WMU",
    yaxis_title="Score",
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Footnotes
# -----------------------------
with st.expander("Model assumptions and caveats"):
    st.markdown(
        """
- `Avg hunting walk (km)` uses `derived_effort_km_per_expected_animal` when present.
- If that field is missing, the app falls back to `survey_effort_km / abundance_estimate` when possible.
- Some species such as elk have missing density in several WMUs. The app uses abundance/area as a fallback only when both values exist.
- Missing values are handled conservatively in the scoring formula.
- This dashboard is a visualization layer over survey data and proxy metrics; it is **not** an official hunting success predictor.
"""
    )

st.caption("Built for the uploaded Alberta WMU 2025 survey JSON.")
