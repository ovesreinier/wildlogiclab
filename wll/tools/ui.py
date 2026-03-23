import base64
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
DATA_PATH = Path("database/alberta_wmu_survey_database_2025_2.json")
APP_TITLE = "Alberta WMUs Animal Surveys"
APP_SUBTITLE = "Alberta Public Surveys - 2025"
LOGO_PATH = "docs/assets/logo2-r.png"
CHANNEL_NAME = "Wild Logic Lab"
# Set to your channel pages (used by sidebar follow buttons).
YOUTUBE_URL = "https://www.youtube.com/@WildLogicLab"
FACEBOOK_URL = "https://www.facebook.com/wildlogic.ca"
# Success score weights (independent from hike effort).
SUCCESS_WEIGHTS = {
    "density": 0.45,
    "abundance": 0.25,
    "observed": 0.20,
    "sex_ratio": 0.05,
    "juvenile_ratio": 0.05,
}


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
                    "density_per_km2_inferred": safe_float(
                        sp.get("density_per_km2_inferred_from_count")
                    ),
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

    # Density priority:
    # 1) explicit density_per_km2 from parser
    # 2) inferred density_per_km2_inferred_from_count (new parser field)
    # 3) abundance / area fallback when possible
    density_fallback = df["abundance_estimate"] / df["area_km2"]
    df["density_proxy_per_km2"] = (
        df["density_per_km2"]
        .fillna(df["density_per_km2_inferred"])
        .fillna(density_fallback)
    )

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
    out["sex_ratio_norm"] = normalize_series(
        out["male_per_100_female"], higher_is_better=True)
    out["juvenile_ratio_norm"] = normalize_series(
        out["juvenile_per_100_female"], higher_is_better=True)
    positive = (
        SUCCESS_WEIGHTS["density"] * out["density_norm"].fillna(0)
        + SUCCESS_WEIGHTS["abundance"] * out["abundance_norm"].fillna(0)
        + SUCCESS_WEIGHTS["observed"] * out["observed_norm"].fillna(0)
        + SUCCESS_WEIGHTS["sex_ratio"] * out["sex_ratio_norm"].fillna(0)
        + SUCCESS_WEIGHTS["juvenile_ratio"] *
        out["juvenile_ratio_norm"].fillna(0)
    )

    # Success score intentionally ignores hike effort.
    out["global_success_score"] = (
        100 * positive / sum(SUCCESS_WEIGHTS.values())).round(1)

    # Realistic hike burden: combine direct hike proxy with spatial search pressure.
    density_pressure = np.where(
        out["density_proxy_per_km2"].notna() & (
            out["density_proxy_per_km2"] > 0),
        1.0 / out["density_proxy_per_km2"],
        np.nan,
    )
    area_pressure = np.where(
        out["area_km2"].notna()
        & (out["area_km2"] > 0)
        & out["abundance_estimate"].notna()
        & (out["abundance_estimate"] > 0),
        out["area_km2"] / out["abundance_estimate"],
        np.nan,
    )

    effort_components = pd.DataFrame(
        {
            "hike_component": np.log1p(out["effort_proxy_km"]),
            "density_component": np.log1p(density_pressure),
            "area_component": np.log1p(area_pressure),
        }
    )
    component_weights = pd.Series(
        {"hike_component": 0.55, "density_component": 0.30, "area_component": 0.15}
    )
    weighted_sum = effort_components.mul(
        component_weights, axis=1).sum(axis=1, skipna=True)
    weight_present = effort_components.notna().mul(
        component_weights, axis=1).sum(axis=1)
    out["effort_raw_blend"] = np.where(
        weight_present > 0, weighted_sum / weight_present, np.nan)

    valid_effort = pd.Series(out["effort_raw_blend"]).dropna()
    if valid_effort.empty:
        effort_scaled = pd.Series([0.5] * len(out), index=out.index)
    else:
        lo = np.percentile(valid_effort, 10)
        hi = np.percentile(valid_effort, 90)
        if np.isclose(lo, hi):
            lo = valid_effort.min()
            hi = valid_effort.max()
        if np.isclose(lo, hi):
            effort_scaled = pd.Series([0.5 if pd.notna(
                v) else np.nan for v in out["effort_raw_blend"]], index=out.index)
        else:
            effort_scaled = (
                (out["effort_raw_blend"] - lo) / (hi - lo)).clip(0, 1)

    out["effort_raw_norm"] = effort_scaled
    out["effort_norm_red"] = 1 - effort_scaled
    # Hunting is always hard: keep score in a practical 35-100 band.
    out["global_effort_score"] = (35 + 65 * effort_scaled.fillna(0.5)).round(1)
    out["effort_difficulty"] = np.select(
        [
            out["global_effort_score"] < 55,
            out["global_effort_score"] < 78,
        ],
        ["Hard", "Very hard"],
        default="Fucking hard",
    )
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


def _parcoords_dim_to_dict(dim) -> Dict:
    if hasattr(dim, "to_plotly_json"):
        return dim.to_plotly_json()
    return dict(dim)


def parallel_coordinates_with_wmu_axis(
    pvt: pd.DataFrame, metric_cols: List[str], title: str
) -> go.Figure:
    """Parallel coordinates with a left WMU axis (readable ticks) and room for labels."""
    plot_df = pvt.reset_index(drop=True).copy()
    n = len(plot_df)
    plot_df["_par_wmu_idx"] = np.arange(n, dtype=float)
    wmu_labels = plot_df["wmu_id"].astype(str).tolist()

    color_series = plot_df[metric_cols].mean(axis=1, skipna=True)
    dim_cols = ["_par_wmu_idx"] + metric_cols
    labels = {"_par_wmu_idx": "WMU", **{c: c for c in metric_cols}}

    fig = px.parallel_coordinates(
        plot_df,
        dimensions=dim_cols,
        color=color_series,
        title=title,
        labels=labels,
    )

    max_ticks = 32
    if n <= max_ticks:
        tickvals = list(range(n))
        ticktext = wmu_labels
    else:
        step = max(1, (n + max_ticks - 1) // max_ticks)
        tickvals = list(range(0, n, step))
        if tickvals[-1] != n - 1:
            tickvals.append(n - 1)
        ticktext = [wmu_labels[i] for i in tickvals]

    dims = fig.data[0].dimensions
    first = _parcoords_dim_to_dict(dims[0])
    first.update(
        {
            "label": "WMU",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "range": [-0.5, float(n) - 0.5] if n > 1 else [-0.5, 0.5],
        }
    )
    rest = [_parcoords_dim_to_dict(dims[i]) for i in range(1, len(dims))]
    fig.update_traces(dimensions=[first] + rest)

    fig.update_traces(
        # Keep lines dark, but avoid bright tones that create a halo-like effect.
        line=dict(
            colorscale=[
                [0.0, "#1f1f1f"],
                [0.5, "#2a2a2a"],
                [1.0, "#353535"],
            ],
            cmin=float(color_series.min()
                       ) if color_series.notna().any() else 0.0,
            cmax=float(color_series.max()
                       ) if color_series.notna().any() else 1.0,
            showscale=False,
        ),
        # Use black text for all axis labels/ticks (regular weight).
        labelfont=dict(size=14, color="#000000"),
        tickfont=dict(size=12, color="#000000"),
    )
    fig.update_layout(
        margin=dict(l=150, r=100, t=88, b=88, pad=6),
        font=dict(size=11),
        height=max(480, 340 + 14 * len(dim_cols)),
    )
    return fig


def metric_card(label: str, value, help_text: Optional[str] = None, fmt: str = "{:.2f}"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        display = "N/A"
    elif isinstance(value, (float, np.floating, int, np.integer)):
        display = fmt.format(value)
    else:
        display = str(value)
    st.metric(label, display, help=help_text)


def format_detail_value(value) -> str:
    """Normalize mixed metric types for display-only tables."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return str(value)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _logo_mime_and_b64(logo_path: str) -> Optional[Tuple[str, str]]:
    p = Path(logo_path)
    if not p.exists():
        return None
    ext = p.suffix.lower()
    mime = (
        "image/png"
        if ext == ".png"
        else "image/jpeg"
        if ext in (".jpg", ".jpeg")
        else "image/png"
    )
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return (mime, b64)


def render_app_header(logo_path: str, title: str, subtitle: str) -> None:
    """Logo and title on one row with vertical alignment and a simple hero treatment."""
    parts = _logo_mime_and_b64(logo_path)
    if not parts:
        st.title(title)
        st.caption(subtitle)
        return

    mime, b64 = parts
    title_esc = _html_escape(title)
    sub_esc = _html_escape(subtitle)

    st.markdown(
        f"""
<style>
.wll-hero {{
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 1.25rem;
  flex-wrap: wrap;
  padding: 0.35rem 0 1.15rem;
  margin-bottom: 0.35rem;
}}
.wll-hero__logo {{
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.wll-hero__logo img {{
  width: clamp(56px, 9vw, 88px);
  height: auto;
  display: block;
  border-radius: 10px;
}}
.wll-hero__text {{
  min-width: 0;
  flex: 1;
}}
.wll-hero__text h1 {{
  font-size: clamp(1.55rem, 3.2vw, 2.1rem);
  font-weight: 700;
  margin: 0;
  line-height: 1.18;
  letter-spacing: -0.02em;
}}
.wll-hero__text p {{
  margin: 0.38rem 0 0;
  color: rgba(49, 51, 63, 0.68);
  font-size: 1.02rem;
  line-height: 1.35;
}}
</style>
<div class="wll-hero">
  <div class="wll-hero__logo">
    <img src="data:{mime};base64,{b64}" alt="" />
  </div>
  <div class="wll-hero__text">
    <h1>{title_esc}</h1>
    <p>{sub_esc}</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar_brand(
    logo_path: str,
    caption: str,
    channel_name: str = CHANNEL_NAME,
    youtube_url: str = YOUTUBE_URL,
    facebook_url: str = FACEBOOK_URL,
) -> None:
    """Centered logo, channel name, caption, and social links at the top of the sidebar."""
    name_esc = _html_escape(channel_name)
    cap_esc = _html_escape(caption)
    yt_esc = _html_escape(youtube_url)
    fb_esc = _html_escape(facebook_url)
    parts = _logo_mime_and_b64(logo_path)
    if parts:
        mime, b64 = parts
        st.markdown(
            f"""
<style>
.wll-sidebar-brand {{
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 0.2rem 0 0.65rem;
}}
.wll-sidebar-brand img {{
  width: 200px;
  max-width: 100%;
  height: auto;
  display: block;
}}
.wll-sidebar-brand__name {{
  margin: 0.55rem 0 0;
  font-size: 1.05rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: rgba(49, 51, 63, 0.95);
}}
.wll-sidebar-brand__cap {{
  margin: 0.35rem 0 0;
  font-size: 0.78rem;
  line-height: 1.35;
  color: rgba(49, 51, 63, 0.68);
}}
.wll-sidebar-brand__social {{
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  width: 100%;
  max-width: 220px;
  margin-top: 0.75rem;
}}
.wll-sidebar-brand__social a {{
  display: block;
  padding: 0.45rem 0.65rem;
  border-radius: 8px;
  font-size: 0.82rem;
  text-align: center;
  text-decoration: none;
  font-weight: 500;
  transition: opacity 0.15s ease;
}}
.wll-sidebar-brand__social a:hover {{
  opacity: 0.9;
}}
.wll-btn-yt {{
  background: #c4302b;
  color: #fff !important;
}}
.wll-btn-fb {{
  background: #1877f2;
  color: #fff !important;
}}
</style>
<div class="wll-sidebar-brand">
  <img src="data:{mime};base64,{b64}" alt="" />
  <div class="wll-sidebar-brand__name">{name_esc}</div>
  <p class="wll-sidebar-brand__cap">{cap_esc}</p>
  <div class="wll-sidebar-brand__social">
    <a class="wll-btn-yt" href="{yt_esc}" target="_blank" rel="noopener noreferrer">Follow on YouTube</a>
    <a class="wll-btn-fb" href="{fb_esc}" target="_blank" rel="noopener noreferrer">Follow on Facebook</a>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**{_html_escape(channel_name)}**")
        st.caption(caption)
        st.markdown(
            f"""
<div class="wll-sidebar-brand__social" style="margin-top:0.5rem;">
  <a class="wll-btn-yt" href="{yt_esc}" target="_blank" rel="noopener noreferrer"
     style="display:block;padding:0.45rem 0.65rem;border-radius:8px;background:#c4302b;color:#fff!important;text-align:center;text-decoration:none;font-size:0.82rem;font-weight:500;margin-bottom:0.45rem;">Follow on YouTube</a>
  <a class="wll-btn-fb" href="{fb_esc}" target="_blank" rel="noopener noreferrer"
     style="display:block;padding:0.45rem 0.65rem;border-radius:8px;background:#1877f2;color:#fff!important;text-align:center;text-decoration:none;font-size:0.82rem;font-weight:500;">Follow on Facebook</a>
</div>
""",
            unsafe_allow_html=True,
        )

    st.divider()


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide",
                   page_icon=LOGO_PATH)
render_app_header(LOGO_PATH, APP_TITLE, APP_SUBTITLE)


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
    render_sidebar_brand(LOGO_PATH, APP_SUBTITLE)

    view_mode = st.radio(
        "View mode",
        ["Basic", "Expert"],
        index=0,
        help="Basic: quick first view. Expert: full analytics dashboard.",
    )

    if view_mode == "Expert":
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

if view_mode == "Basic":
    st.subheader("Quick view: Best 5 WMUs by species")
    st.caption(
        "Ranking is based on highest success chance and lower effort. "
        "Green = success chance, Red = effort."
    )
    st.markdown(
        """
<style>
.wll-basic-msg {
  border-radius: 10px;
  padding: 0.6rem 0.75rem;
  margin: 0.35rem 0 0.45rem;
}
.wll-basic-msg h4 {
  margin: 0 0 0.22rem;
  font-size: 0.92rem;
  font-weight: 700;
  color: #ffffff;
}
.wll-basic-msg p {
  margin: 0;
  font-size: 0.82rem;
  line-height: 1.35;
  color: #ffffff;
}
.wll-basic-msg-success {
  background: #1f9d55;
}
.wll-basic-msg-effort {
  background: #d64545;
}
</style>
<div class="wll-basic-msg wll-basic-msg-success">
  <h4>Success</h4>
  <p>
    Higher score means better odds of finding animals.
    It mostly comes from density (animals per km²), abundance, and observed count.
  </p>
</div>
<div class="wll-basic-msg wll-basic-msg-effort">
  <h4>Effort</h4>
  <p>
    Higher score means a harder hunt.
    It reflects expected walking distance plus search pressure from low density and large area per animal.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<style>
.wll-basic-card {
  border: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 10px;
  padding: 0.65rem 0.75rem;
  margin: 0.45rem 0 0.7rem;
}
.wll-basic-row {
  margin: 0.45rem 0 0.65rem;
}
.wll-basic-row-title {
  font-size: 0.92rem;
  font-weight: 600;
  margin-bottom: 0.3rem;
}
.wll-basic-bar-label {
  font-size: 0.75rem;
  color: rgba(49, 51, 63, 0.78);
  margin-bottom: 0.12rem;
}
.wll-basic-track {
  width: 100%;
  height: 24px;
  border-radius: 999px;
  background: rgba(49, 51, 63, 0.1);
  overflow: hidden;
}
.wll-basic-fill {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  color: #fff;
  font-size: 0.73rem;
  font-weight: 600;
  white-space: nowrap;
  padding-right: 0.4rem;
  min-width: 3.3rem;
  box-sizing: border-box;
}
.wll-basic-success {
  background: #1f9d55;
}
.wll-basic-effort {
  background: #d64545;
}
</style>
""",
        unsafe_allow_html=True,
    )

    species_labels = sorted(df["species_label"].dropna().unique().tolist())
    show_all_species = st.checkbox(
        "Show all species (can be long on mobile)",
        value=False,
    )
    selected_species_basic = st.selectbox(
        "Species",
        species_labels,
        index=0,
        disabled=show_all_species,
    )
    species_to_render = species_labels if show_all_species else [selected_species_basic]

    for species_label in species_to_render:
        sp_rank = (
            df[df["species_label"] == species_label]
            .sort_values(
                ["global_success_score", "global_effort_score"],
                ascending=[False, True],
            )
            .head(5)
            .copy()
        )

        if sp_rank.empty:
            continue

        sp_rank = sp_rank.reset_index(drop=True)
        sp_rank["rank_label"] = sp_rank.apply(
            lambda r: f"#{int(r.name) + 1} - WMU {r['wmu_id']}", axis=1
        )
        container = st.container() if not show_all_species else st.expander(
            f"{species_label} - Top 5 WMUs", expanded=False
        )
        with container:
            if not show_all_species:
                st.markdown(f"**{species_label} - Top 5 WMUs**")
            for _, row in sp_rank.iterrows():
                success_pct = float(np.clip(row["global_success_score"], 0, 100))
                effort_pct = float(np.clip(row["global_effort_score"], 0, 100))
                rank_label = _html_escape(str(row["rank_label"]))
                st.markdown(
                    f"""
<div class="wll-basic-card">
  <div class="wll-basic-row">
    <div class="wll-basic-row-title">{rank_label}</div>
    <div class="wll-basic-bar-label">Success chance</div>
    <div class="wll-basic-track">
      <div class="wll-basic-fill wll-basic-success" style="width:{success_pct:.1f}%;">{success_pct:.1f}%</div>
    </div>
    <div class="wll-basic-bar-label" style="margin-top:0.32rem;">Effort</div>
    <div class="wll-basic-track">
      <div class="wll-basic-fill wll-basic-effort" style="width:{effort_pct:.1f}%;">{effort_pct:.1f}%</div>
    </div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

    st.stop()

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
    "effort_difficulty",
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
            "effort_difficulty": "Effort level",
        }
    ),
    width="stretch",
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
    st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig, width="stretch")

# -----------------------------
# Per-species tabs
# -----------------------------
st.markdown("## Species Details")
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
                            "Effort level",
                            "Trend direction",
                            "Trend percent",
                            "Survey effort (km)",
                            "Area surveyed (km²)",
                        ],
                        "Value": [
                            format_detail_value(row["average_group_size"]),
                            format_detail_value(row["group_size_min"]),
                            format_detail_value(row["group_size_max"]),
                            format_detail_value(row["male_per_100_female"]),
                            format_detail_value(
                                row["juvenile_per_100_female"]),
                            format_detail_value(row.get("effort_difficulty")),
                            format_detail_value(row["trend_direction"]),
                            format_detail_value(row["trend_percent"]),
                            format_detail_value(row["survey_effort_km"]),
                            format_detail_value(row["area_surveyed_km2"]),
                        ],
                    }
                )
                st.dataframe(details, width="stretch")

            with d2:
                comp = pd.DataFrame(
                    {
                        "component": [
                            "Density",
                            "Abundance",
                            "Observed",
                            "Male ratio",
                            "Juvenile ratio",
                            "Effort advantage",
                        ],
                        "normalized_value": [
                            row.get("density_norm"),
                            row.get("abundance_norm"),
                            row.get("observed_norm"),
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
                st.plotly_chart(fig, width="stretch")

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
                st.plotly_chart(fig, width="stretch")

            with c2:
                radar_values = [
                    row.get("density_norm", 0) or 0,
                    row.get("abundance_norm", 0) or 0,
                    row.get("observed_norm", 0) or 0,
                    row.get("sex_ratio_norm", 0) or 0,
                    row.get("effort_norm_red", 0) or 0,
                ]
                radar_labels = ["Density", "Abundance",
                                "Observed", "Male ratio", "Effort Advantage"]
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
                st.plotly_chart(fig, width="stretch")

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
st.plotly_chart(rank_fig, width="stretch")

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
    width="stretch",
)

# -----------------------------
# Parallel coordinates section
# -----------------------------
st.markdown("## Global WMUs Overview")
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

    fig = parallel_coordinates_with_wmu_axis(pvt, metric_cols, title)
    st.plotly_chart(fig, width="stretch")

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
st.plotly_chart(fig, width="stretch")

# -----------------------------
# Plain-language score guide
# -----------------------------
st.markdown("## What these scores mean (plain English)")
st.markdown(
    """
**Success score (green):** "How likely you are to find animals in this WMU/species."

- Higher score = better odds.
- Built from animal data, not from how hard the walk is.
- Main ingredients:
  - **Density (45%)**: more animals per km² helps the most.
  - **Abundance (25%)**: total number of animals helps.
  - **Observed count (20%)**: how many were actually seen helps.
  - **Male ratio (5%)** and **juvenile ratio (5%)**: small adjustments.
- Simple rule: **higher success score means better chances**.

**Effort score (red):** "How hard it may be to get an opportunity."

- Higher score = more work/harder hunt.
- Built from 3 pressure signals:
  - **Hike component (55%)**: your expected walk distance per animal.
  - **Density pressure (30%)**: lower animal density means more searching.
  - **Area pressure (15%)**: bigger area per estimated animal means more ground to cover.
- This is scaled to a practical **35-100** range.
- Simple rule: **lower effort score means easier conditions**.

**Quick read:** best targets are usually **high success + low effort**.
"""
)

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

st.caption("Alberta WMUs Aerial Survey - 2025. Reference: https://github.com/wildlife-labs/alberta-wmu-survey-dashboard")
