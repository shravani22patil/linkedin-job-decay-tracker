"""
api/main.py
LinkedIn Job Decay Tracker — FastAPI Backend (Day 12–14)

Three endpoint groups:
  GET  /health                     — service health check
  GET  /analytics/*                — all dashboard data endpoints
  POST /insights/market-pulse      — Gemini market pulse
  POST /insights/role-advice       — Gemini personalised role advice
  POST /insights/skill-roadmap     — Gemini skill learning roadmap

Design decisions:
  1. All DataFrames are loaded ONCE at startup using lifespan events.
     Streamlit app calls these endpoints — no pandas in the frontend.
  2. Every analytics endpoint returns JSON. The Streamlit app rebuilds
     DataFrames from JSON using pd.DataFrame(response.json()["data"]).
  3. /health is a real health check — it reports whether Gemini is
     configured and when data was last loaded.
  4. Pydantic v2 models used for all request/response shapes.

Run:
    cd job_decay_tracker
    uvicorn api.main:app --reload --port 8000

Or from the project root:
    python -m uvicorn api.main:app --reload --port 8000
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

# Add project root to path so we can import analytics / insights
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from analytics import (
    load_all,
    get_overview_metrics,
    get_posting_velocity,
    get_survival_by_role,
    get_survival_by_company_type,
    get_survival_by_seniority,
    get_survival_by_worktype,
    get_skill_survival,
    get_skill_role_matrix,
    get_da_skill_gap,
    get_skill_cooccurrence,
    get_india_summary,
    get_top_surviving_companies,
    get_skill_count_survival,
    build_insight_payload,
)
from insights import (
    generate_market_pulse,
    generate_role_advice,
    generate_skill_roadmap,
    gemini_is_configured,
)

# ── App state ──────────────────────────────────────────────────────────────────
# Loaded once at startup — shared across all requests
_app_state: dict = {}

VALID_ROLES = [
    "Data Analyst", "Data Scientist", "Business Analyst",
    "Data Engineer", "ML Engineer", "Product Analyst", "Analytics Engineer",
]

VALID_LOCATIONS = [
    "India", "United States", "United Kingdom", "Canada",
    "Australia", "Singapore", "Global",
]


# ── Lifespan (replaces deprecated on_event) ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all DataFrames once at startup — not on every request."""
    print("[startup] Loading data files...")
    try:
        _app_state["data"] = load_all()
        _app_state["loaded_at"] = datetime.utcnow().isoformat()
        rows = {k: len(v) for k, v in _app_state["data"].items()}
        print(f"[startup] Loaded: {rows}")
    except FileNotFoundError as e:
        print(f"[startup] ERROR: {e}")
        print("[startup] Server will start but all analytics endpoints will return 503.")
        _app_state["data"] = None
        _app_state["loaded_at"] = None
    yield
    # Shutdown cleanup (nothing needed here)
    _app_state.clear()
    print("[shutdown] State cleared.")


# ── App init ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LinkedIn Job Decay Tracker API",
    description=(
        "Analytics backend for the LinkedIn Job Decay Tracker project. "
        "Analyses 60,413 job postings across 3 monthly snapshots (Apr–Jun 2026)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to specific URLs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_data():
    """Raises 503 if data failed to load at startup."""
    if _app_state.get("data") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Data files not loaded. "
                "Run load_data.py → survival.py → skill_extractor.py first."
            ),
        )
    return _app_state["data"]


def _sanitise(obj):
    """Replaces NaN/Inf floats with None for JSON compliance."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    return obj


def _df_to_response(df: pd.DataFrame, **kwargs) -> dict:
    """Converts a DataFrame to a JSON-serialisable response dict."""
    records = _sanitise(df.to_dict(orient="records"))
    return {
        "data":    records,
        "rows":    len(df),
        "columns": list(df.columns),
        **kwargs,
    }


# ── Pydantic models ────────────────────────────────────────────────────────────

class InsightRequest(BaseModel):
    target_role:     str = Field(default="Data Analyst")
    target_location: str = Field(default="India")
    current_role:    Optional[str] = Field(default="Data Analyst")

    class Config:
        json_schema_extra = {
            "example": {
                "target_role":     "Data Analyst",
                "target_location": "India",
                "current_role":    "Data Analyst",
            }
        }


class HealthResponse(BaseModel):
    status:          str
    version:         str
    data_loaded:     bool
    loaded_at:       Optional[str]
    gemini_ready:    bool
    rows_in_memory:  Optional[dict]
    timestamp:       str


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """
    Service health check. Call this first to verify the server is ready.
    Returns whether data is loaded and whether Gemini is configured.
    """
    data_ok = _app_state.get("data") is not None
    rows = (
        {k: len(v) for k, v in _app_state["data"].items()}
        if data_ok else None
    )
    return HealthResponse(
        status          = "ok" if data_ok else "degraded",
        version         = "1.0.0",
        data_loaded     = data_ok,
        loaded_at       = _app_state.get("loaded_at"),
        gemini_ready    = gemini_is_configured(),
        rows_in_memory  = rows,
        timestamp       = datetime.utcnow().isoformat(),
    )


# ── Analytics endpoints ────────────────────────────────────────────────────────

@app.get("/analytics/overview", tags=["Analytics"])
def analytics_overview():
    """
    Top-line metrics for the dashboard header cards.

    Returns 20 key numbers including:
    - 97.6% disappearance rate
    - 76.1% May posting surge
    - 2.6x MNC vs Startup survival multiplier
    """
    data     = _require_data()
    metrics  = get_overview_metrics(data["cohort"])
    return {"metrics": metrics, "computed_at": datetime.utcnow().isoformat()}


@app.get("/analytics/velocity", tags=["Analytics"])
def analytics_velocity():
    """
    Monthly posting velocity — new jobs per month with carry-over.

    April: 15,112 | May: 26,607 (+76.1%) | June: 18,694 (−29.7%)
    """
    data = _require_data()
    df   = get_posting_velocity(data["cohort"])
    return _df_to_response(df)


@app.get("/analytics/survival/role", tags=["Analytics"])
def survival_by_role():
    """
    30-day survival rate per tech role type (April cohort).

    Key finding: DA roles have 1.0% survival (fastest-disappearing tech category).
    DS roles have 5.2% survival — slower fill, likely more selective.
    """
    data = _require_data()
    df   = get_survival_by_role(data["cohort"])
    return _df_to_response(df)


@app.get("/analytics/survival/company", tags=["Analytics"])
def survival_by_company():
    """
    MNC vs Startup/SME survival rates.

    MNCs: 5.9% | Startups: 2.3% | Multiplier: 2.6x
    """
    data = _require_data()
    df   = get_survival_by_company_type(data["cohort"])
    return _df_to_response(df)


@app.get("/analytics/survival/seniority", tags=["Analytics"])
def survival_by_seniority():
    """Survival rates by seniority level."""
    data = _require_data()
    df   = get_survival_by_seniority(data["cohort"])
    return _df_to_response(df)


@app.get("/analytics/survival/worktype", tags=["Analytics"])
def survival_by_worktype():
    """Survival rates by employment type (Full-time, Contract, Internship etc.)."""
    data = _require_data()
    df   = get_survival_by_worktype(data["cohort"])
    return _df_to_response(df)


@app.get("/analytics/survival/skill-count", tags=["Analytics"])
def survival_by_skill_count():
    """
    Survival rate bucketed by number of skills mentioned in the posting.

    2-skill jobs survive slightly longest (3.3%).
    4+ skill jobs disappear fastest (1.4%) — very specialised roles move quickly.
    """
    data = _require_data()
    df   = get_skill_count_survival(data["cohort"])
    return _df_to_response(df)


@app.get("/analytics/skills/survival", tags=["Skills"])
def skill_survival():
    """
    Skill survival advantage — which skills appear in longer-lived vs fast-gone postings.

    Top: SQL (+1.78pp), Statistics (+1.42pp), Spark (+0.84pp)
    Bottom: Power BI (−0.96pp), R (−1.46pp), Machine Learning (−1.76pp)
    """
    data = _require_data()
    df   = get_skill_survival(data["skill_surv"])
    return _df_to_response(df)


@app.get("/analytics/skills/role-matrix", tags=["Skills"])
def skill_role_matrix():
    """
    Skill demand matrix: % of each role type's postings mentioning each skill.

    DA: SQL 60.1%, Stats 47.4%, Power BI 44.8%, Python 44.4%
    DS: Python 89.5%, ML 78.5%, Stats 76.6%, SQL 68.0%
    """
    data = _require_data()
    df   = get_skill_role_matrix(data["role_matrix"])
    return _df_to_response(df)


@app.get("/analytics/skills/gap", tags=["Skills"])
def skill_gap(
    from_role: str = Query(default="Data Analyst",   description="Source role"),
    to_role:   str = Query(default="Data Scientist", description="Target role"),
):
    """
    Skill gap between two roles based on posting frequency.

    E.g. DA → DS: Machine Learning +53.7pp, Python +45.1pp, Scikit-learn +43.6pp
    """
    data = _require_data()
    df   = get_da_skill_gap(data["role_matrix"], from_role=from_role, to_role=to_role)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for roles '{from_role}' → '{to_role}'. "
                   f"Valid roles: {VALID_ROLES}"
        )
    return _df_to_response(df, from_role=from_role, to_role=to_role)


@app.get("/analytics/skills/cooccurrence", tags=["Skills"])
def skill_cooccurrence(
    min_corr: float = Query(default=0.3, ge=0.0, le=1.0,
                             description="Minimum correlation threshold"),
):
    """
    Skill pairs that appear together frequently.
    Pandas + NumPy: r=0.934 | NumPy + Sklearn: r=0.804
    """
    data = _require_data()
    df   = get_skill_cooccurrence(data["cooccur"], min_corr=min_corr)
    return _df_to_response(df, min_corr=min_corr)


@app.get("/analytics/india", tags=["Analytics"])
def india_analysis():
    """
    India-specific survival and skill analysis.

    3,672 India jobs tracked. India tech survival: 7.0% (vs 2.4% global).
    India emphasises Excel (+8.4pp), Tableau (+4.7pp) more than global average.
    Note: 43 April tech roles in India — small sample, directional only.
    """
    data   = _require_data()
    result = get_india_summary(data["cohort"], data["india_prof"])

    # Convert DataFrames inside the dict to serialisable form
    serialised = {}
    for k, v in result.items():
        if isinstance(v, pd.DataFrame):
            serialised[k] = v.to_dict(orient="records")
        else:
            serialised[k] = v

    return serialised


@app.get("/analytics/companies/top-surviving", tags=["Analytics"])
def top_surviving_companies(
    min_postings: int = Query(default=5, ge=1, description="Min April postings"),
    top_n:        int = Query(default=10, ge=1, le=50),
):
    """
    Companies whose April postings survived longest into May.
    High survival = slower hiring pipeline OR repeated re-listing.
    """
    data = _require_data()
    df   = get_top_surviving_companies(data["cohort"], min_postings, top_n)
    return _df_to_response(df)


# ── Insight endpoints ──────────────────────────────────────────────────────────

@app.post("/insights/market-pulse", tags=["AI Insights"])
def insight_market_pulse():
    """
    Gemini-generated market summary covering all key findings.
    Returns fallback text if GEMINI_API_KEY is not configured.
    Typical response time: 2–4 seconds.
    """
    data    = _require_data()
    payload = build_insight_payload(
        data["cohort"], data["skill_surv"], data["india_prof"]
    )
    result = generate_market_pulse(payload)
    return result


@app.post("/insights/role-advice", tags=["AI Insights"])
def insight_role_advice(req: InsightRequest):
    """
    Personalised advice for a specific role + location combination.

    Example request body:
        {"target_role": "Data Analyst", "target_location": "India"}
    """
    if req.target_role not in VALID_ROLES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid target_role. Choose from: {VALID_ROLES}"
        )

    data    = _require_data()
    payload = build_insight_payload(
        data["cohort"], data["skill_surv"], data["india_prof"],
        target_role=req.target_role, target_location=req.target_location,
    )
    result = generate_role_advice(payload, req.target_role, req.target_location)
    return result


@app.post("/insights/skill-roadmap", tags=["AI Insights"])
def insight_skill_roadmap(req: InsightRequest):
    """
    Skill learning roadmap for transitioning from current_role to target_role.

    Example request body:
        {"current_role": "Data Analyst", "target_role": "Data Scientist"}
    """
    current = req.current_role or "Data Analyst"
    target  = req.target_role

    if target not in VALID_ROLES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid target_role. Choose from: {VALID_ROLES}"
        )
    if current == target:
        raise HTTPException(
            status_code=422,
            detail="current_role and target_role must be different."
        )

    data      = _require_data()
    payload   = build_insight_payload(
        data["cohort"], data["skill_surv"], data["india_prof"],
        target_role=target, target_location=req.target_location,
    )
    skill_gap = get_da_skill_gap(data["role_matrix"],
                                  from_role=current, to_role=target)

    if skill_gap.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Skill gap data not available for '{current}' → '{target}'."
        )

    result = generate_skill_roadmap(payload, skill_gap,
                                     current_role=current, target_role=target)
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
