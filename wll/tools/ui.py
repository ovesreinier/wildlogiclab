import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("database/alberta_wmu_survey_database_2025_with_draws.json")
APP_TITLE = "Wild Logic Lab"
APP_SUBTITLE = "Alberta Public Aereal Survey - 2025"
LOGO_PATH = "docs/assets/logo2-r.png"
CHANNEL_NAME = ""
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


def _first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _normalize_draw_required(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    raw = str(value).strip().lower()
    if raw in {"yes", "y", "true", "required", "draw", "draw required", "1"}:
        return "Yes"
    if raw in {"no", "n", "false", "not required", "general", "over the counter", "0"}:
        return "No"
    return str(value).strip()


def _derive_draw_fields_from_summary(sp: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    draw_summary = sp.get("draw_summary")
    if not isinstance(draw_summary, dict):
        return (None, None, None)

    available_draws = draw_summary.get("available_draws")
    if not isinstance(available_draws, list):
        available_draws = []
    available_draws = [d for d in available_draws if isinstance(d, dict)]

    if available_draws:
        draw_required = "Yes"
    else:
        draw_required = "No" if draw_summary.get("na_reason") else None

    draw_classes: List[str] = []
    difficulties: List[str] = []
    for draw in available_draws:
        section = str(draw.get("section") or "").lower()
        section_norm = section.replace("-", "_").replace(" ", "_")

        # Draw class is inferred directly from section labels in draw_summary:
        # e.g., antlered_*, antlerless_*, calf_* (including common typos).
        if "calf" in section_norm:
            draw_classes.append("calf")
        if (
            "antlerless" in section_norm
            or "antereless" in section_norm
            or "antler_les" in section_norm
        ):
            draw_classes.append("antlerless")
        if "antlered" in section_norm or "anterleed" in section_norm:
            draw_classes.append("antlered")

        difficulty = draw.get("difficulty")
        if difficulty is not None and str(difficulty).strip():
            difficulties.append(str(difficulty).strip().replace("_", " "))

    sex_value = None
    if draw_classes:
        unique_classes = list(dict.fromkeys(draw_classes))
        if len(unique_classes) == 1:
            sex_value = unique_classes[0]
        else:
            sex_value = " / ".join(unique_classes)

    complexity_value = None
    if difficulties:
        unique_difficulties = list(dict.fromkeys(difficulties))
        complexity_value = " / ".join(unique_difficulties)

    return (draw_required, sex_value, complexity_value)


def _collect_species_draw_entries(sp: Dict) -> List[Dict[str, str]]:
    draw_summary = sp.get("draw_summary")
    if not isinstance(draw_summary, dict):
        return []

    available_draws = draw_summary.get("available_draws")
    if not isinstance(available_draws, list):
        return []

    entries: List[Dict[str, str]] = []
    for draw in available_draws:
        if not isinstance(draw, dict):
            continue
        # Support both parser naming variants:
        # quote/quota and num_points/min_points.
        quote_value = _first_non_empty(draw.get("quote"), draw.get("quota"))
        num_points_value = _first_non_empty(draw.get("num_points"), draw.get("min_points"))
        entry = {
            "quote": format_detail_value(quote_value),
            "num_points": format_detail_value(num_points_value),
            "guaranteed_points": format_detail_value(draw.get("guaranteed_points")),
            "chance_at_min_points_percent": format_detail_value(
                draw.get("chance_at_min_points_percent")
            ),
            "difficulty": format_detail_value(draw.get("difficulty")),
        }
        entries.append(entry)
    return entries


def _derive_wmu_area_from_background(background_summary: Optional[str]) -> Optional[float]:
    if not background_summary:
        return None
    text = background_summary.replace("\n", " ")
    patterns = [
        r"covers an area of ([\d,]+(?:\.\d+)?)\s*km(?:\s*2|²)",
        r"covering an area of ([\d,]+(?:\.\d+)?)\s*km(?:\s*2|²)",
        r"area of (?:approximately )?([\d,]+(?:\.\d+)?)\s*km(?:\s*2|²)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return safe_float(m.group(1).replace(",", ""))
    return None


def _derive_surveyed_area_from_background(background_summary: Optional[str]) -> Optional[float]:
    if not background_summary:
        return None
    text = background_summary.replace("\n", " ")
    patterns = [
        r"approximately\s*([\d,]+(?:\.\d+)?)\s*km(?:\s*2|²)[^\.]{0,80}surveyed",
        r"([\d,]+(?:\.\d+)?)\s*km(?:\s*2|²)[^\.]{0,80}has been surveyed",
        r"surveyed[^\.]{0,80}([\d,]+(?:\.\d+)?)\s*km(?:\s*2|²)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return safe_float(m.group(1).replace(",", ""))
    return None


def _derive_surveyed_area_from_species(species_map: Dict) -> Optional[float]:
    estimates: List[float] = []
    for sp in (species_map or {}).values():
        if not isinstance(sp, dict):
            continue
        abundance = safe_float(sp.get("abundance_estimate"))
        density = safe_float(sp.get("density_per_km2"))
        if (
            pd.notna(abundance)
            and pd.notna(density)
            and abundance is not None
            and density is not None
            and abundance > 0
            and density > 0
        ):
            estimates.append(float(abundance) / float(density))
    if not estimates:
        return None
    return float(np.median(estimates))


def get_wmu_area_summary(raw: Dict) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for wmu_id, wmu_data in (raw.get("wmus", {}) or {}).items():
        survey = (wmu_data.get("survey") or {})
        area_km2 = safe_float(wmu_data.get("area_km2"))
        area_surveyed_km2 = safe_float(survey.get("area_surveyed_km2"))
        surveyed_pct = (
            (area_surveyed_km2 / area_km2) * 100
            if pd.notna(area_km2) and area_km2 and area_km2 > 0 and pd.notna(area_surveyed_km2)
            else np.nan
        )
        out[str(wmu_id)] = {
            "area_km2": area_km2,
            "area_surveyed_km2": area_surveyed_km2,
            "surveyed_pct": surveyed_pct,
        }
    return out


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
        background_summary = wmu_data.get("background_summary")
        area_km2_value = safe_float(wmu_data.get("area_km2"))
        if area_km2_value is None:
            area_km2_value = _derive_wmu_area_from_background(background_summary)
        area_surveyed_value = safe_float(survey.get("area_surveyed_km2"))
        if area_surveyed_value is None:
            area_surveyed_value = _derive_surveyed_area_from_background(background_summary)
        if area_surveyed_value is None:
            area_surveyed_value = _derive_surveyed_area_from_species(species_map)

        for species_name, sp in species_map.items():
            draw_required_summary, draw_sex_summary, draw_complexity_summary = _derive_draw_fields_from_summary(
                sp
            )
            draw_obj = sp.get("draw") if isinstance(sp.get("draw"), dict) else {}
            draw_required_raw = _first_non_empty(
                draw_required_summary,
                sp.get("draw_required"),
                sp.get("requires_draw"),
                sp.get("draw_needed"),
                sp.get("is_draw"),
                draw_obj.get("required"),
                draw_obj.get("needed"),
                draw_obj.get("is_required"),
            )
            draw_sex_raw = _first_non_empty(
                draw_sex_summary,
                sp.get("draw_sex"),
                sp.get("sex_for_draw"),
                sp.get("draw_gender"),
                draw_obj.get("sex"),
                draw_obj.get("gender"),
            )
            draw_complexity_raw = _first_non_empty(
                draw_complexity_summary,
                sp.get("draw_complexity"),
                sp.get("draw_difficulty"),
                sp.get("draw_level"),
                draw_obj.get("complexity"),
                draw_obj.get("difficulty"),
                draw_obj.get("level"),
            )
            rows.append(
                {
                    "wmu_id": str(wmu_id),
                    "species": species_name,
                    "species_label": prettify_species_name(species_name),
                    "year": wmu_data.get("year"),
                    "source_pdf": wmu_data.get("source_pdf"),
                    "area_km2": area_km2_value,
                    "survey_method": survey.get("method"),
                    "survey_effort_km": safe_float(survey.get("effort_km")),
                    "coverage_percent": safe_float(survey.get("coverage_percent")),
                    "area_surveyed_km2": area_surveyed_value,
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
                    "draw_required": _normalize_draw_required(draw_required_raw),
                    "draw_sex": None if draw_sex_raw is None else str(draw_sex_raw),
                    "draw_complexity": None
                    if draw_complexity_raw is None
                    else str(draw_complexity_raw),
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


def render_app_header(
    logo_path: str,
    title: str,
    subtitle: str,
    company_name: str = CHANNEL_NAME,
) -> None:
    """Top-centered branding with logo, app title, and company name."""
    parts = _logo_mime_and_b64(logo_path)
    if not parts:
        st.title(title)
        st.markdown(f"**{company_name}**")
        st.caption(subtitle)
        return

    mime, b64 = parts
    title_esc = _html_escape(title)
    sub_esc = _html_escape(subtitle)
    company_esc = _html_escape(company_name)

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
.wll-hero__text .wll-hero__company {{
  margin: 0.32rem 0 0;
  color: rgba(49, 51, 63, 0.92);
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}}
</style>
<div class="wll-hero">
  <div class="wll-hero__logo">
    <img src="data:{mime};base64,{b64}" alt="" />
  </div>
  <div class="wll-hero__text">
    <h1>{title_esc}</h1>
    <div class="wll-hero__company">{company_esc}</div>
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
render_app_header(LOGO_PATH, APP_TITLE, APP_SUBTITLE, CHANNEL_NAME)


if not DATA_PATH.exists():
    st.error(
        f"Could not find `{DATA_PATH}`. Place the JSON file in the same folder as this Streamlit script."
    )
    st.stop()

raw_data = load_json(DATA_PATH)
df = compute_scores(build_species_records(raw_data))
wmu_area_summary = get_wmu_area_summary(raw_data)
st.subheader("Best 5 WMUs by species")
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
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.55rem;
  font-size: 0.92rem;
  font-weight: 600;
  margin-bottom: 0.3rem;
}
.wll-basic-stars {
  display: inline-flex;
  align-items: center;
  gap: 0.1rem;
  letter-spacing: 0.02em;
}
.wll-basic-star {
  font-size: 0.92rem;
  line-height: 1;
}
.wll-basic-star--filled {
  color: #f6b000;
}
.wll-basic-star--empty {
  color: #c3c8d1;
}
.wll-basic-bar-label {
  font-size: 0.75rem;
  color: rgba(49, 51, 63, 0.78);
  margin-bottom: 0.12rem;
}
.wll-basic-draw-text {
  margin: 0.35rem 0 0.28rem;
  font-size: 0.76rem;
  color: rgba(49, 51, 63, 0.88);
}
.wll-basic-draw-item {
  margin-top: 0.12rem;
}
.wll-basic-card-footer {
  margin-top: 0.38rem;
  padding-top: 0.32rem;
  border-top: 1px dashed rgba(49, 51, 63, 0.22);
  font-size: 0.76rem;
  color: rgba(49, 51, 63, 0.88);
}
.wll-basic-card-footer span {
  display: inline-block;
  margin-right: 0.62rem;
  margin-top: 0.08rem;
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
    "Show All Species",
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
    container = st.container() if not show_all_species else st.expander(
        f"{species_label} - Top 5 WMUs", expanded=False
    )
    with container:
        if not show_all_species:
            st.markdown(f"**{species_label} - Top 5 WMUs**")
        for _, row in sp_rank.iterrows():
            success_pct = float(np.clip(row["global_success_score"], 0, 100))
            effort_pct = float(np.clip(row["global_effort_score"], 0, 100))
            rank_pos = int(row.name) + 1
            filled_stars = max(1, 6 - rank_pos)
            empty_stars = 5 - filled_stars
            wmu_label = _html_escape(f"WMU-{row['wmu_id']}")
            stars_html = (
                '<span class="wll-basic-star wll-basic-star--filled">★</span>' * filled_stars
                + '<span class="wll-basic-star wll-basic-star--empty">★</span>' * empty_stars
            )
            wmu_area = wmu_area_summary.get(str(row["wmu_id"]), {})
            total_area = safe_float(wmu_area.get("area_km2"))
            area_surveyed = safe_float(wmu_area.get("area_surveyed_km2"))
            surveyed_pct = safe_float(wmu_area.get("surveyed_pct"))
            total_animals = safe_float(row.get("abundance_estimate"))
            wmu_raw = (raw_data.get("wmus") or {}).get(str(row["wmu_id"]), {})
            species_raw = ((wmu_raw.get("species") or {}) if isinstance(wmu_raw, dict) else {}).get(
                row["species"], {}
            )
            draw_entries = _collect_species_draw_entries(species_raw if isinstance(species_raw, dict) else {})

            if draw_entries:
                draw_lines = []
                for draw_entry in draw_entries:
                    draw_lines.append(
                        '<div class="wll-basic-draw-item">'
                        f'<strong>Quote:</strong> {_html_escape(draw_entry["quote"])} | '
                        f'<strong>Num points:</strong> {_html_escape(draw_entry["num_points"])} | '
                        f'<strong>Guaranteed points:</strong> {_html_escape(draw_entry["guaranteed_points"])} | '
                        f'<strong>Chance at min points %:</strong> {_html_escape(draw_entry["chance_at_min_points_percent"])} | '
                        f'<strong>Difficulty:</strong> {_html_escape(draw_entry["difficulty"])}'
                        "</div>"
                    )
                draw_text_html = (
                    '<div class="wll-basic-draw-text">'
                    + "".join(draw_lines)
                    + "</div>"
                )
            else:
                draw_text_html = (
                    '<div class="wll-basic-draw-text">'
                    "<strong>Draw information:</strong> N/A"
                    "</div>"
                )

            footer_html = (
                '<div class="wll-basic-card-footer">'
                f'<span><strong>Area covered:</strong> {_html_escape(format_detail_value(total_area))} km²</span>'
                f'<span><strong>Total animals:</strong> {_html_escape(format_detail_value(total_animals))}</span>'
                f'<span><strong>Surveyed area:</strong> {_html_escape(format_detail_value(area_surveyed))} km²</span>'
                f'<span><strong>Surveyed %:</strong> {_html_escape(format_detail_value(surveyed_pct))}%</span>'
                "</div>"
            )
            st.markdown(
                f"""
<div class="wll-basic-card">
<div class="wll-basic-row">
<div class="wll-basic-row-title">
<span>{wmu_label}</span>
<span class="wll-basic-stars">{stars_html}</span>
</div>
<div class="wll-basic-bar-label">Success chance</div>
<div class="wll-basic-track">
<div class="wll-basic-fill wll-basic-success" style="width:{success_pct:.1f}%;">{success_pct:.1f}%</div>
</div>
<div class="wll-basic-bar-label" style="margin-top:0.32rem;">Effort</div>
<div class="wll-basic-track">
<div class="wll-basic-fill wll-basic-effort" style="width:{effort_pct:.1f}%;">{effort_pct:.1f}%</div>
</div>
{draw_text_html}
{footer_html}
</div>
</div>
""",
                unsafe_allow_html=True,
            )

st.caption("Alberta WMUs Aerial Survey - 2025. Reference: https://github.com/wildlife-labs/alberta-wmu-survey-dashboard")
