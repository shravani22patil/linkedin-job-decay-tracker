"""
app.py
LinkedIn Job Decay Tracker — Streamlit Dashboard (Week 3)

Run:
    # Terminal 1 — backend
    uvicorn api_main:app --reload --port 8000

    # Terminal 2 — frontend
    streamlit run app.py

Design rules:
  - Zero statistics computed here. Every number comes from the FastAPI backend.
  - All DataFrames rebuilt from API JSON responses.
  - Charts use Plotly with the custom dark theme — no Matplotlib, no default
    Streamlit charts.
  - Layout: sidebar (controls) + main area (3 tab pages).
  - CSS overrides every default Streamlit colour.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from textwrap import dedent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Decay Tracker",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Colour constants (shared across all charts) ────────────────────────────────
C = {
    "bg":       "#0B0E14",
    "surface":  "#131720",
    "border":   "#1E2330",
    "text":     "#E8EAED",
    "muted":    "#6B7280",
    "blue":     "#4F8EF7",
    "green":    "#22C55E",
    "red":      "#EF4444",
    "amber":    "#F59E0B",
    "purple":   "#A78BFA",
    "teal":     "#2DD4BF",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#131720",
    font=dict(family="system-ui, -apple-system, sans-serif",
              color=C["muted"], size=11),
    title_font=dict(color=C["text"], size=13),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border"],
               tickfont=dict(color=C["muted"])),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border"],
               tickfont=dict(color=C["muted"])),
    margin=dict(l=16, r=16, t=44, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"])),
)

MONTH_ORDER  = ["April", "May", "June"]
MONTH_COLORS = {"April": C["blue"], "May": C["purple"], "June": C["teal"]}

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif}
.stApp{background:#0B0E14}
.block-container{padding:1.25rem 1.75rem 3rem !important;max-width:1380px}
#MainMenu,footer,header{visibility:hidden}
.stDeployButton{display:none}

/* Top nav */
.top-nav{display:flex;align-items:center;justify-content:space-between;
  padding:.6rem 0 1.1rem;border-bottom:1px solid #1E2330;margin-bottom:1.25rem}
.brand{font-size:1.05rem;font-weight:600;color:#E8EAED;letter-spacing:-.3px}
.brand-sub{font-size:.7rem;color:#6B7280;margin-top:1px}
.nav-pill{font-size:.68rem;font-weight:500;padding:3px 10px;border-radius:20px;
  background:#1A2744;color:#4F8EF7;border:1px solid #263659}

/* Metric cards */
.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1rem}
.mcard{background:#131720;border:1px solid #1E2330;border-radius:10px;padding:.9rem 1rem}
.mcard-label{font-size:.67rem;font-weight:500;color:#6B7280;text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:5px}
.mcard-value{font-size:1.55rem;font-weight:600;letter-spacing:-.8px;line-height:1}
.mcard-sub{font-size:.7rem;color:#6B7280;margin-top:4px}
.blue{color:#4F8EF7}.green{color:#22C55E}.red{color:#EF4444}
.amber{color:#F59E0B}.purple{color:#A78BFA}.white{color:#E8EAED}

/* Insight card */
.insight-wrap{background:#0E1420;border:1px solid #1E2B44;border-radius:12px;
  padding:1.1rem 1.25rem;margin-top:.5rem}
.insight-badge{font-size:.65rem;font-weight:600;padding:3px 9px;border-radius:20px;
  background:#1A2744;color:#4F8EF7;border:1px solid #263659;
  text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;display:inline-block}
.insight-text{font-size:.83rem;color:#C4CAD4;line-height:1.8}

/* Stat table */
.stat-table{width:100%;border-collapse:collapse;font-size:.8rem}
.stat-table th{text-align:left;padding:7px 10px;color:#6B7280;font-weight:500;
  font-size:.67rem;text-transform:uppercase;letter-spacing:.05em;
  border-bottom:1px solid #1E2330}
.stat-table td{padding:8px 10px;border-bottom:1px solid #141720;color:#E8EAED}
.stat-table tr:last-child td{border-bottom:none}
.pos{color:#22C55E;font-weight:600}.neg{color:#EF4444;font-weight:600}
.neu{color:#F59E0B}

/* Findings */
.finding{background:#131720;border:1px solid #1E2330;border-radius:10px;
  padding:.9rem 1rem;margin-bottom:.6rem}
.finding-num{font-size:.65rem;font-weight:600;color:#6B7280;
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.finding-text{font-size:.82rem;color:#C4CAD4;line-height:1.6}
.finding-stat{font-size:1.2rem;font-weight:600;color:#4F8EF7;
  letter-spacing:-.5px;margin-bottom:2px}

/* Section labels */
.slabel{font-size:.67rem;font-weight:500;color:#6B7280;text-transform:uppercase;
  letter-spacing:.07em;margin-bottom:.5rem;margin-top:.25rem}

/* India badge */
.warn-pill{display:inline-block;font-size:.67rem;font-weight:500;padding:2px 8px;
  border-radius:8px;background:#201A09;color:#F59E0B;border:1px solid #4A3B0D}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:4px;background:transparent;
  border-bottom:1px solid #1E2330;padding-bottom:0}
.stTabs [data-baseweb="tab"]{background:transparent;color:#6B7280;
  font-size:.8rem;font-weight:500;padding:.4rem .9rem;border-radius:6px 6px 0 0}
.stTabs [aria-selected="true"]{background:#131720 !important;
  color:#E8EAED !important;border-bottom:2px solid #4F8EF7}

/* Sidebar */
[data-testid="stSidebar"]{background:#0F1117;border-right:1px solid #1E2330}
[data-testid="stSidebar"] label{color:#9CA3AF !important;font-size:.8rem}

/* Selectbox / radio */
div[data-testid="stSelectbox"] select,
div[data-testid="stRadio"] label{color:#E8EAED !important}

/* Buttons */
.stButton>button{background:#1C2E52 !important;color:#4F8EF7 !important;
  border:1px solid #263659 !important;border-radius:8px !important;
  font-weight:500 !important;font-size:.8rem !important;
  padding:.4rem 1rem !important;transition:all .15s !important}
.stButton>button:hover{background:#22396B !important;border-color:#4F8EF7 !important}
.stButton>button[kind="primary"]{background:#1A44C8 !important;
  color:#fff !important;border-color:transparent !important}

/* Spinner */
.stSpinner>div{border-top-color:#4F8EF7 !important}
</style>
""", unsafe_allow_html=True)


# ── API helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch(endpoint: str) -> dict:
    """GET wrapper — caches for 5 minutes. Returns {} on error."""
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({endpoint}): {e}")
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_post(endpoint: str, body: dict) -> dict:
    """POST wrapper — caches for 5 minutes."""
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=body, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({endpoint}): {e}")
        return {}


def to_df(resp: dict) -> pd.DataFrame:
    """Converts a standard {data:[...]} API response to a DataFrame."""
    if not resp or "data" not in resp:
        return pd.DataFrame()
    return pd.DataFrame(resp["data"])


def api_ok() -> bool:
    """Returns True if the backend is reachable and data is loaded."""
    try:
        h = requests.get(f"{API_URL}/health", timeout=4).json()
        return h.get("data_loaded", False)
    except Exception:
        return False


# ── Chart builders ─────────────────────────────────────────────────────────────

def bar_chart(df: pd.DataFrame,
              x: str, y: str,
              color_col: str | None = None,
              color_map: dict | None = None,
              title: str = "",
              horizontal: bool = False,
              height: int = 320) -> go.Figure:
    """Generic bar chart with project dark theme applied."""
    if df.empty:
        return go.Figure().update_layout(**PLOTLY_LAYOUT, title=title)

    if color_col and color_map:
        colors = df[color_col].map(color_map).fillna(C["blue"]).tolist()
    else:
        colors = C["blue"]

    if horizontal:
        fig = go.Figure(go.Bar(
            y=df[x], x=df[y], orientation="h",
            marker_color=colors, marker_line_width=0,
        ))
    else:
        fig = go.Figure(go.Bar(
            x=df[x], y=df[y],
            marker_color=colors, marker_line_width=0,
        ))

    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height)
    fig.update_traces(marker_opacity=0.9)
    return fig


def line_chart(df: pd.DataFrame,
               x: str,
               y_cols: list[str],
               colors: list[str] | None = None,
               title: str = "",
               height: int = 300) -> go.Figure:
    fig = go.Figure()
    colors = colors or [C["blue"], C["purple"], C["teal"], C["green"]]
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], name=col,
            mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2.5),
            marker=dict(size=8, color=colors[i % len(colors)]),
        ))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height)
    return fig


def diverging_bar(df: pd.DataFrame,
                  label_col: str,
                  value_col: str,
                  title: str = "",
                  height: int = 400) -> go.Figure:
    """Horizontal bar chart that diverges at 0 — used for skill survival advantage."""
    df = df.sort_values(value_col)
    colors = [C["green"] if v >= 0 else C["red"] for v in df[value_col]]
    fig = go.Figure(go.Bar(
        y=df[label_col], x=df[value_col],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=df[value_col].apply(lambda v: f"{v:+.2f}pp"),
        textposition="outside",
        textfont=dict(color=C["muted"], size=9),
    ))
    fig.add_vline(x=0, line_color=C["border"], line_width=1.5)
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height,
                      xaxis_title="Survival advantage (pp)")
    return fig


def heatmap(df: pd.DataFrame,
            index_col: str,
            title: str = "",
            height: int = 300) -> go.Figure:
    """Skill × role heatmap."""
    matrix = df.set_index(index_col)
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=[[0, "#131720"], [0.5, "#1A3A6B"], [1, "#4F8EF7"]],
        showscale=True,
        text=matrix.values.round(0).astype(int).astype(str),
        texttemplate="%{text}%",
        textfont=dict(size=8.5, color=C["text"]),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=height,
    )

    fig.update_xaxes(
        tickangle=-30,
        tickfont=dict(size=9),
    )

    fig.update_layout(
        margin=dict(
            l=120,
            r=16,
            t=44,
            b=16,
        )
    )
    return fig


def metric_card(label: str, value: str,
                sub: str = "", color: str = "blue") -> str:
    return f"""<div class="mcard">
<div class="mcard-label">{label}</div>
<div class="mcard-value {color}">{value}</div>
{"<div class='mcard-sub'>" + sub + "</div>" if sub else ""}
</div>"""


# ── Backend check ──────────────────────────────────────────────────────────────

if not api_ok():
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem">
      <div style="font-size:2.5rem;margin-bottom:1rem">⚠️</div>
      <div style="font-size:1.1rem;font-weight:500;color:#E8EAED;margin-bottom:.5rem">
        Backend not running
      </div>
      <div style="font-size:.85rem;color:#6B7280;line-height:1.7">
        Start the FastAPI server first:<br>
        <code style="background:#1C1F26;padding:4px 10px;border-radius:6px;
          color:#4F8EF7;font-size:.8rem">
          uvicorn api_main:app --reload --port 8000
        </code>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")

    target_role = st.selectbox(
        "Your target role",
        ["Data Analyst", "Data Scientist", "Business Analyst",
         "Data Engineer", "ML Engineer", "Product Analyst"],
        index=0,
    )

    target_location = st.selectbox(
        "Target location",
        ["India", "United States", "United Kingdom",
         "Canada", "Australia", "Singapore", "Global"],
        index=0,
    )

    compare_from = st.selectbox(
        "Skill gap — current role",
        ["Data Analyst", "Business Analyst", "Data Scientist"],
        index=0,
    )
    compare_to = st.selectbox(
        "Skill gap — target role",
        ["Data Scientist", "Data Analyst", "Data Engineer",
         "ML Engineer", "Product Analyst"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.7rem;color:#374151;line-height:1.6'>"
        "Data: 60,413 LinkedIn job postings<br>"
        "Period: April–June 2026<br>"
        "Source: Kaggle public dataset<br>"
        "Last loaded: today"
        "</div>",
        unsafe_allow_html=True
    )


# ── Top nav ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="top-nav">
  <div>
    <div class="brand">📉 LinkedIn Job Decay Tracker</div>
    <div class="brand-sub">
      How fast do job postings disappear? · 60,413 postings · Apr–Jun 2026
    </div>
  </div>
  <span class="nav-pill">Gemini 1.5 Flash · AI Insights</span>
</div>
""", unsafe_allow_html=True)


# ── Load all data upfront ──────────────────────────────────────────────────────

with st.spinner("Loading data from backend..."):
    overview_raw  = fetch("/analytics/overview")
    velocity_raw  = fetch("/analytics/velocity")
    role_raw      = fetch("/analytics/survival/role")
    company_raw   = fetch("/analytics/survival/company")
    seniority_raw = fetch("/analytics/survival/seniority")
    worktype_raw  = fetch("/analytics/survival/worktype")
    sk_count_raw  = fetch("/analytics/survival/skill-count")
    sk_surv_raw   = fetch("/analytics/skills/survival")
    role_mat_raw  = fetch("/analytics/skills/role-matrix")
    sk_gap_raw    = fetch(
        f"/analytics/skills/gap"
        f"?from_role={compare_from.replace(' ','+')}"
        f"&to_role={compare_to.replace(' ','+')}",
    )
    cooccur_raw   = fetch("/analytics/skills/cooccurrence?min_corr=0.4")
    india_raw     = fetch("/analytics/india")
    companies_raw = fetch("/analytics/companies/top-surviving")

ov        = overview_raw.get("metrics", {})
vel_df    = to_df(velocity_raw)
role_df   = to_df(role_raw)
comp_df   = to_df(company_raw)
sen_df    = to_df(seniority_raw)
wt_df     = to_df(worktype_raw)
sk_cnt_df = to_df(sk_count_raw)
sk_surv_df = to_df(sk_surv_raw)
role_mat_df = to_df(role_mat_raw)
sk_gap_df   = to_df(sk_gap_raw)
cooccur_df  = to_df(cooccur_raw)
top_co_df   = to_df(companies_raw)


# ── 4 top metric cards ─────────────────────────────────────────────────────────

dis_rate   = ov.get("disappearance_rate_pct", 0)
mnc_mult   = ov.get("mnc_multiplier", 0)
may_surge  = ov.get("may_surge_pct", 0)
total_jobs = ov.get("total_unique_jobs", 0)

cards_html = dedent(f"""
<div class="metric-row">
  {metric_card("Disappearance Rate", f"{dis_rate}%", "of April jobs gone before May", "red")}
  {metric_card("Total Jobs Tracked", f"{total_jobs:,}", "across 3 monthly snapshots", "blue")}
  {metric_card("MNC vs Startup", f"{mnc_mult}x", "MNCs survive longer", "amber")}
  {metric_card("May Posting Surge", f"+{may_surge}%", "more new postings in May vs April", "green")}
</div>
""").strip()
st.markdown(cards_html, unsafe_allow_html=True)


# ── Tab pages ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Market Overview",
    "🧬  Skill Intelligence",
    "🌏  India Analysis",
    "✨  AI Insights",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # ── Row 1: Velocity + Survival Funnel ──────────────────────────────────────
    c1, c2 = st.columns([1.3, 1], gap="large")

    with c1:
        st.markdown('<div class="slabel">Monthly posting velocity</div>',
                    unsafe_allow_html=True)
        if not vel_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=vel_df["month"],
                y=vel_df["new_postings"],
                name="New postings",
                marker_color=C["blue"],
                marker_line_width=0,
            ))
            fig.add_trace(go.Bar(
                x=vel_df["month"],
                y=vel_df["carried_from_prev"],
                name="Carried from prev month",
                marker_color=C["purple"],
                marker_line_width=0,
            ))
            for _, row in vel_df.iterrows():
                if row["mom_change_pct"] is not None:
                    sign = "+" if row["mom_change_pct"] > 0 else ""
                    color = C["green"] if row["mom_change_pct"] > 0 else C["red"]

                    fig.add_annotation(
                        x=row["month"],
                        y=row["new_postings"] + 800, 
                        text=f"{sign}{row['mom_change_pct']:.0f}%",
                        font=dict(color=color, size=10),
                        showarrow=False,
                    )

        fig.update_layout(
          **PLOTLY_LAYOUT,
           barmode="stack",
           title="New job postings per month",
           height=310,
           )

        fig.update_layout(
         legend=dict(
           bgcolor="rgba(0,0,0,0)",
           font=dict(color=C["muted"]),
           orientation="h",
             y=-0.15,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="slabel">April cohort survival funnel</div>',
                    unsafe_allow_html=True)
        apr_size = ov.get("april_cohort_size", 0)
        survived_may  = ov.get("survived_to_may", 0)
        survived_june = ov.get("survived_to_june", 0)

        fig2 = go.Figure(go.Funnel(
            y=["Posted in April", "Still live in May", "Still live in June"],
            x=[apr_size, survived_may, survived_june],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=[C["blue"], C["amber"], C["green"]]),
            connector=dict(line=dict(color=C["border"], width=1)),
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title="April cohort — how many survived?",
            height=310,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Role survival + Company type ───────────────────────────────────
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown('<div class="slabel">30-day survival rate by role type</div>',
                    unsafe_allow_html=True)
        if not role_df.empty:
            rd = role_df.sort_values("survival_rate_pct")
            colors = [
                C["green"] if v >= 5
                else C["amber"] if v >= 1
                else C["red"]
                for v in rd["survival_rate_pct"]
            ]
            fig3 = go.Figure(go.Bar(
                y=rd["role_type"],
                x=rd["survival_rate_pct"],
                orientation="h",
                marker_color=colors,
                marker_line_width=0,
                text=[
                    f"{v:.1f}%  (n={n}{'*' if not r else ''})"
                    for v, n, r in zip(
                        rd["survival_rate_pct"],
                        rd["total"],
                        rd["sample_reliable"],
                    )
                ],
                textposition="outside",
                textfont=dict(color=C["muted"], size=9),
            ))
            fig3.update_layout(
                **PLOTLY_LAYOUT,
                title="Lower = disappears faster (harder to find = re-listed more)",
                height=320,
                xaxis_title="% still live in May",
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("* Small sample — treat as directional only")

    with c4:
        st.markdown('<div class="slabel">MNC vs Startup/SME survival</div>',
                    unsafe_allow_html=True)
        if not comp_df.empty:
            mnc_rate = comp_df.loc[
                comp_df["company_type"] == "MNC", "survival_rate_pct"
            ].values[0] if "MNC" in comp_df["company_type"].values else 0
            sme_rate = comp_df.loc[
                comp_df["company_type"] == "Startup/SME", "survival_rate_pct"
            ].values[0] if "Startup/SME" in comp_df["company_type"].values else 0

            fig4 = go.Figure()
            for _, row in comp_df.iterrows():
                clr = C["blue"] if row["company_type"] == "MNC" else C["amber"]
                fig4.add_trace(go.Bar(
                    name=row["company_type"],
                    x=[row["company_type"]],
                    y=[row["survival_rate_pct"]],
                    marker_color=clr,
                    marker_line_width=0,
                    text=f"{row['survival_rate_pct']:.1f}%<br>({row['total']:,} jobs)",
                    textposition="outside",
                    textfont=dict(color=C["text"], size=11),
                ))
            fig4.update_layout(
                **PLOTLY_LAYOUT,
                title=f"MNCs survive {ov.get('mnc_multiplier',0):.1f}x longer than Startups",
                height=320,
                showlegend=False,
                yaxis_title="30-day survival rate (%)",
                yaxis_range=[0, max(mnc_rate, sme_rate) * 1.5],
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Seniority + Worktype ───────────────────────────────────────────
    c5, c6 = st.columns(2, gap="large")

    with c5:
        st.markdown('<div class="slabel">Survival by seniority level</div>',
                    unsafe_allow_html=True)
        if not sen_df.empty:
            sd = sen_df.sort_values("survival_rate_pct", ascending=True)
            fig5 = bar_chart(
                sd, "seniority", "survival_rate_pct",
                title="Entry-level roles disappear fastest",
                horizontal=True, height=300,
            )
            st.plotly_chart(fig5, use_container_width=True)

    with c6:
        st.markdown('<div class="slabel">Survival by work type</div>',
                    unsafe_allow_html=True)
        if not wt_df.empty:
            wd = wt_df.sort_values("survival_rate_pct", ascending=True)
            fig6 = bar_chart(
                wd, "worktype", "survival_rate_pct",
                title="Internships survive longest (re-listed repeatedly)",
                horizontal=True, height=300,
            )
            st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4: Key findings ───────────────────────────────────────────────────
    st.markdown('<div class="slabel" style="margin-top:.75rem">Key findings</div>',
                unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown(f"""
        <div class="finding">
          <div class="finding-num">Finding 1</div>
          <div class="finding-stat">{dis_rate}%</div>
          <div class="finding-text">
            of April job postings disappeared before the May snapshot —
            the LinkedIn market turns over almost completely every 30 days.
            Only {ov.get('two_month_survival_pct',0)}% survived all 3 months.
          </div>
        </div>""", unsafe_allow_html=True)

    with fc2:
        st.markdown(f"""
        <div class="finding">
          <div class="finding-num">Finding 2</div>
          <div class="finding-stat">{mnc_mult}×</div>
          <div class="finding-text">
            MNC postings survive {mnc_mult}x longer than Startup/SME postings
            ({ov.get('mnc_survival_pct',0):.1f}% vs {ov.get('startup_survival_pct',0):.1f}%).
            Apply to startups within the first week — they move faster.
          </div>
        </div>""", unsafe_allow_html=True)

    with fc3:
        st.markdown(f"""
        <div class="finding">
          <div class="finding-num">Finding 3</div>
          <div class="finding-stat">+{may_surge:.0f}%</div>
          <div class="finding-text">
            new postings appeared in May vs April — a clear seasonal hiring wave.
            June then contracted {abs(ov.get('june_change_pct',0)):.0f}%,
            suggesting Q2 is peak hiring season.
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Skill Intelligence
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    # ── Skill survival advantage ───────────────────────────────────────────────
    st.markdown('<div class="slabel">Skill survival advantage (April cohort)</div>',
                unsafe_allow_html=True)
    st.caption(
        "Positive = skill appears more in **longer-lived** postings (role takes longer to fill).  "
        "Negative = skill appears more in **fast-disappearing** postings (fills quickly).  "
        "Higher survival ≠ more in demand — it means harder to fill OR repeatedly re-listed."
    )

    if not sk_surv_df.empty:
        fig_div = diverging_bar(
            sk_surv_df, "skill", "survival_advantage",
            title="Survival advantage by skill (pp = percentage points)",
            height=440,
        )
        st.plotly_chart(fig_div, use_container_width=True)

    # ── Role matrix heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="slabel" style="margin-top:.75rem">Skill demand by role type (% of postings)</div>',
                unsafe_allow_html=True)
    if not role_mat_df.empty:
        skill_cols_in_matrix = [c for c in role_mat_df.columns if c != "role_type"]
        # Sort skills by total demand across all roles
        col_order = (role_mat_df[skill_cols_in_matrix]
                     .mean()
                     .sort_values(ascending=False)
                     .index.tolist())
        role_mat_sorted = role_mat_df[["role_type"] + col_order]
        fig_heat = heatmap(
            role_mat_sorted, "role_type",
            title="% of each role's postings mentioning the skill",
            height=320,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Skill gap + Co-occurrence ─────────────────────────────────────────────
    sg1, sg2 = st.columns([1.2, 1], gap="large")

    with sg1:
        st.markdown(
            f'<div class="slabel">Skill gap: {compare_from} → {compare_to}</div>',
            unsafe_allow_html=True,
        )
        if not sk_gap_df.empty:
            from_col = f"{compare_from}_pct"
            to_col   = f"{compare_to}_pct"
            if from_col in sk_gap_df.columns and to_col in sk_gap_df.columns:
                top_gaps = sk_gap_df[sk_gap_df["gap"] > 5].head(10)
                colors_gap = [
                    C["green"] if g > 20 else C["blue"]
                    for g in top_gaps["gap"]
                ]
                fig_gap = go.Figure(go.Bar(
                    y=top_gaps["skill"],
                    x=top_gaps["gap"],
                    orientation="h",
                    marker_color=colors_gap,
                    marker_line_width=0,
                    text=[f"+{v:.0f}pp" for v in top_gaps["gap"]],
                    textposition="outside",
                    textfont=dict(color=C["muted"], size=9),
                ))
                fig_gap.update_layout(
                    **PLOTLY_LAYOUT,
                    title=f"Skills {compare_to} needs more than {compare_from}",
                    height=340,
                    xaxis_title="Percentage point gap",
                )
                st.plotly_chart(fig_gap, use_container_width=True)
        else:
            st.info(f"Skill gap data not available for this role combination.")

    with sg2:
        st.markdown('<div class="slabel">Top skill co-occurrence pairs</div>',
                    unsafe_allow_html=True)
        st.caption("Learn one → you likely need the other")
        if not cooccur_df.empty:
            top_pairs = cooccur_df.head(8)
            fig_cooc = go.Figure(go.Bar(
                y=top_pairs["pair_label"],
                x=top_pairs["correlation"],
                orientation="h",
                marker_color=[
                    C["green"] if v >= 0.8
                    else C["blue"] if v >= 0.6
                    else C["purple"]
                    for v in top_pairs["correlation"]
                ],
                marker_line_width=0,
                text=[f"r={v:.3f}  ({n} postings)"
                      for v, n in zip(top_pairs["correlation"],
                                      top_pairs["both_mentioned"])],
                textposition="outside",
                textfont=dict(color=C["muted"], size=9),
            ))
            fig_cooc.update_layout(
                **PLOTLY_LAYOUT,
                title="Skills that appear together (Pearson r)",
                height=340,
                xaxis_range=[0, 1.15],
            )
            st.plotly_chart(fig_cooc, use_container_width=True)

    # ── Skills count survival ─────────────────────────────────────────────────
    st.markdown('<div class="slabel" style="margin-top:.75rem">Does listing more skills help a posting survive?</div>',
                unsafe_allow_html=True)
    if not sk_cnt_df.empty:
        colors_cnt = [
            C["green"] if v == sk_cnt_df["survival_rate_pct"].max()
            else C["red"] if v == sk_cnt_df["survival_rate_pct"].min()
            else C["blue"]
            for v in sk_cnt_df["survival_rate_pct"]
        ]
        fig_cnt = go.Figure(go.Bar(
            x=sk_cnt_df["skill_bucket"],
            y=sk_cnt_df["survival_rate_pct"],
            marker_color=colors_cnt,
            marker_line_width=0,
            text=[f"{v:.1f}%<br>(n={n:,})"
                  for v, n in zip(sk_cnt_df["survival_rate_pct"],
                                  sk_cnt_df["total"])],
            textposition="outside",
            textfont=dict(color=C["text"], size=10),
        ))
        fig_cnt.update_layout(
            **PLOTLY_LAYOUT,
            title="2-skill postings survive longest · 4+ skill postings disappear fastest",
            height=280,
            yaxis_title="30-day survival rate (%)",
            yaxis_range=[0, sk_cnt_df["survival_rate_pct"].max() * 1.6],
        )
        st.plotly_chart(fig_cnt, use_container_width=True)
    st.caption(
        "4+ skill postings disappear fastest — very specialised roles either fill "
        "immediately (high demand) or get pulled when the ideal candidate can't be found."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — India Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    if india_raw:
        # Small sample warning
        if india_raw.get("small_sample_warning"):
            st.markdown(
                '<span class="warn-pill">⚠️ Small sample — India tech roles: '
                f'{india_raw.get("india_tech_april",0)} in April cohort. '
                'Directional signal only — not statistically robust.</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-bottom:.75rem'></div>",
                        unsafe_allow_html=True)

        # India metric cards
        i_surv = india_raw.get("india_survival_rate_pct", 0)
        i_tech = india_raw.get("india_tech_survival_pct", 0)
        i_jobs = india_raw.get("total_india_jobs", 0)
        i_tech_n = india_raw.get("india_tech_roles", 0)

        st.markdown(f"""
        <div class="metric-row">
          {metric_card("India total jobs", f"{i_jobs:,}", "across Apr–Jun 2026", "blue")}
          {metric_card("India tech roles", f"{i_tech_n:,}", "DA, DS, BA, DE, ML, PA", "purple")}
          {metric_card("India 30-day survival", f"{i_surv}%", "vs 2.4% global average", "green")}
          {metric_card("India tech survival", f"{i_tech}%", "tech roles only, small sample", "amber")}
        </div>
        """, unsafe_allow_html=True)

        # City distribution + role distribution
        ci1, ci2 = st.columns(2, gap="large")

        with ci1:
            st.markdown('<div class="slabel">Top India hiring cities</div>',
                        unsafe_allow_html=True)
            cities = india_raw.get("top_cities", {})
            if cities:
                city_df = pd.DataFrame(
                    list(cities.items()), columns=["city", "postings"]
                ).sort_values("postings", ascending=True)
                fig_city = bar_chart(
                    city_df, "city", "postings",
                    title="Postings per city (all 3 months)",
                    horizontal=True, height=300,
                )
                st.plotly_chart(fig_city, use_container_width=True)

        with ci2:
            st.markdown('<div class="slabel">India tech role distribution</div>',
                        unsafe_allow_html=True)
            roles_india = india_raw.get("role_distribution", {})
            if roles_india:
                role_pie_df = pd.DataFrame(
                    list(roles_india.items()), columns=["role", "count"]
                )
                fig_pie = go.Figure(go.Pie(
                    labels=role_pie_df["role"],
                    values=role_pie_df["count"],
                    hole=0.45,
                    marker=dict(
                        colors=[C["blue"], C["purple"], C["teal"],
                                C["green"], C["amber"], C["red"]],
                    ),
                    textfont=dict(color=C["text"], size=10),
                ))
                fig_pie.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Role types in India tech postings",
                    height=300,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # India vs global skill emphasis
        st.markdown('<div class="slabel" style="margin-top:.5rem">India skill emphasis vs global average</div>',
                    unsafe_allow_html=True)

        more_df = pd.DataFrame(india_raw.get("skills_india_emphasises", []))
        less_df = pd.DataFrame(india_raw.get("skills_india_deemphasises", []))

        ia1, ia2 = st.columns(2, gap="large")
        with ia1:
            st.caption("Skills India requires **more** than global average")
            if not more_df.empty and "india_vs_global" in more_df.columns:
                fig_more = go.Figure(go.Bar(
                    x=more_df["skill"],
                    y=more_df["india_vs_global"],
                    marker_color=C["green"],
                    marker_line_width=0,
                    text=[f"+{v:.1f}pp" for v in more_df["india_vs_global"]],
                    textposition="outside",
                    textfont=dict(color=C["text"], size=10),
                ))
                fig_more.update_layout(
                    **PLOTLY_LAYOUT,
                    title="India vs global (pp difference)",
                    height=280,
                    yaxis_range=[0, more_df["india_vs_global"].max() * 1.6],
                )
                st.plotly_chart(fig_more, use_container_width=True)

        with ia2:
            st.caption("Skills India requires **less** than global average")
            if not less_df.empty and "india_vs_global" in less_df.columns:
                fig_less = go.Figure(go.Bar(
                    x=less_df["skill"],
                    y=less_df["india_vs_global"],
                    marker_color=C["red"],
                    marker_line_width=0,
                    text=[f"{v:.1f}pp" for v in less_df["india_vs_global"]],
                    textposition="outside",
                    textfont=dict(color=C["text"], size=10),
                ))
                fig_less.update_layout(
                    **PLOTLY_LAYOUT,
                    title="India vs global (pp difference)",
                    height=280,
                    yaxis_range=[less_df["india_vs_global"].min() * 1.6, 0],
                )
                st.plotly_chart(fig_less, use_container_width=True)

        # India insight callout
        st.markdown("""
        <div class="finding" style="margin-top:.75rem">
          <div class="finding-num">India-specific takeaway</div>
          <div class="finding-text">
            India emphasises <strong style="color:#E8EAED">Excel (+8.4pp)</strong> and
            <strong style="color:#E8EAED">Tableau (+4.7pp)</strong> more than the global average,
            while de-emphasising Statistics (−10.3pp) and Machine Learning (−10.3pp).
            For India DA roles: prioritise SQL, Excel, Tableau, and Power BI.
            Add Statistics and Python to differentiate above the average applicant.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("India data not available — check backend connection.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Insights
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown(
        "AI insights are generated by **Gemini 1.5 Flash** using the real "
        "analysis numbers — not generic content. Each insight takes 2–4 seconds.",
    )
    st.markdown("---")

    ai1, ai2, ai3 = st.columns(3, gap="large")

    # ── Market Pulse ──────────────────────────────────────────────────────────
    with ai1:
        st.markdown("#### 📊 Market Pulse")
        st.caption("Overall market summary — what the data says this month.")
        if st.button("Generate Market Pulse", use_container_width=True):
            with st.spinner("Asking Gemini..."):
                pulse = fetch_post("/insights/market-pulse", {})
            if pulse:
                model_label = (
                    "Gemini 1.5 Flash"
                    if pulse.get("model") == "gemini-1.5-flash"
                    else "Preview (add API key)"
                )
                st.markdown(f"""
                <div class="insight-wrap">
                  <span class="insight-badge">{model_label}</span>
                  <div class="insight-text">{pulse.get('insight','').replace(chr(10),'<br>')}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Role Advice ───────────────────────────────────────────────────────────
    with ai2:
        st.markdown(f"#### 🎯 Role Advice")
        st.caption(
            f"Personalised advice for **{target_role}** roles in **{target_location}**."
        )
        if st.button("Generate Role Advice", use_container_width=True):
            with st.spinner("Asking Gemini..."):
                advice = fetch_post("/insights/role-advice", {
                    "target_role":     target_role,
                    "target_location": target_location,
                    "current_role":    target_role,
                })
            if advice:
                model_label = (
                    "Gemini 1.5 Flash"
                    if advice.get("model") == "gemini-1.5-flash"
                    else "Preview (add API key)"
                )
                st.markdown(f"""
                <div class="insight-wrap">
                  <span class="insight-badge">{model_label}</span>
                  <div class="insight-text">{advice.get('insight','').replace(chr(10),'<br>')}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Skill Roadmap ─────────────────────────────────────────────────────────
    with ai3:
        st.markdown(f"#### 🗺️ Skill Roadmap")
        st.caption(
            f"Learning path from **{compare_from}** to **{compare_to}** "
            "based on posting frequency gaps."
        )
        if st.button("Generate Skill Roadmap", use_container_width=True):
            with st.spinner("Asking Gemini..."):
                roadmap = fetch_post("/insights/skill-roadmap", {
                    "current_role":    compare_from,
                    "target_role":     compare_to,
                    "target_location": target_location,
                })
            if roadmap:
                model_label = (
                    "Gemini 1.5 Flash"
                    if roadmap.get("model") == "gemini-1.5-flash"
                    else "Preview (add API key)"
                )
                st.markdown(f"""
                <div class="insight-wrap">
                  <span class="insight-badge">{model_label}</span>
                  <div class="insight-text">{roadmap.get('insight','').replace(chr(10),'<br>')}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Prompt engineering note ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("How the AI layer works — prompt engineering details"):
        st.markdown("""
**Model:** Gemini 1.5 Flash (free tier via Google AI Studio)

**Why this isn't just "ask ChatGPT to summarise":**

Every prompt injects the real computed numbers — disappearance rates,
skill survival advantages, role-specific survival percentages — so the
model cannot hallucinate statistics. The prompt explicitly forbids bullet
points, requires 3-paragraph structure, and includes the data limitation
caveat so the model cites it automatically.

**Three prompt types:**

1. **Market Pulse** — persona: senior labour market analyst. Audience: non-technical
   hiring manager. Injects: velocity, survival rates, skill signals. Constraint:
   say "disappeared" not "filled."

2. **Role Advice** — persona: data analytics career coach. Audience: final-year
   student from Tier-3 college. Injects: target role survival rate, India-specific
   skill emphasis, city data. Constraint: every sentence grounded in a number.

3. **Skill Roadmap** — persona: curriculum designer. Injects: the actual skill gap
   percentages and co-occurrence correlations. Constraint: give a sequenced learning
   order, not a flat list.

**Temperature:** 0.55 — enough creativity for natural prose, low enough to stay
grounded in the numbers.
        """)
