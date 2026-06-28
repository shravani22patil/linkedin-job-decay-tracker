"""
analytics.py
LinkedIn Job Decay Tracker — Analytics Layer (Day 8–9)

This file is the single source of truth for every number shown on the dashboard.
It wraps the raw outputs from survival.py and skill_extractor.py into
dashboard-ready DataFrames and dicts.

Design principles:
  1. Every function is pure — takes DataFrames, returns DataFrames.
     No file I/O, no plotting, no printing inside any function.
  2. Functions are named after the dashboard widget they feed.
     If the function name changes, the dashboard call changes too.
  3. All expensive computation happens here — Streamlit calls these
     once and caches the result with @st.cache_data.

Confirmed numbers (from data audit — do not change without re-running):
  - April cohort:       15,112 unique jobs
  - Disappearance rate: 97.6%  (14,751 gone before May)
  - MNC survival:       5.9%   vs Startup 2.3% → 2.6x multiplier
  - DA survival:        1.0%   (fills very fast)
  - DS survival:        5.2%
  - Product Analyst:    14.3%  (slowest fill — small sample n=7)
  - SQL advantage:      +1.78pp (most common in long-lived postings)
  - ML advantage:       −1.76pp (fastest-disappearing skill)
  - Pandas+NumPy corr:  0.934  (almost always together)

Usage:
    from analytics import load_all, get_overview_metrics, get_survival_by_role, ...
    data = load_all()
    overview = get_overview_metrics(data['cohort'])
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── File paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

FILES = {
    "combined":   DATA_DIR / "combined_jobs.parquet",
    "cohort":     DATA_DIR / "survival_cohort.parquet",
    "skill_surv": DATA_DIR / "skill_survival_comparison.parquet",
    "role_matrix":DATA_DIR / "skill_role_matrix.parquet",
    "cooccur":    DATA_DIR / "skill_cooccurrence.parquet",
    "india_prof": DATA_DIR / "skill_india_profile.parquet",
}

MONTH_ORDER = ["April", "May", "June"]


# ── Data loader ────────────────────────────────────────────────────────────────

def load_all() -> dict[str, pd.DataFrame]:
    """
    Loads all parquet files and returns a dict.
    Call this once at Streamlit startup and pass the dict to every function.

    Usage in app.py:
        @st.cache_data
        def load_data():
            return load_all()
        data = load_data()
    """
    missing = [name for name, path in FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing data files: {missing}\n"
            f"Run in order: load_data.py → survival.py → skill_extractor.py"
        )
    return {name: pd.read_parquet(path) for name, path in FILES.items()}


# ── Section 1: Overview metrics (dashboard top cards) ─────────────────────────

def get_overview_metrics(cohort: pd.DataFrame) -> dict:
    """
    Returns the 8 top-line numbers shown as metric cards on the dashboard.

    Confirmed values:
      total_unique_jobs       = 60,413
      april_cohort_size       = 15,112
      disappearance_rate_pct  = 97.6
      survival_rate_pct       = 2.4
      may_surge_pct           = +76.1  (May had 76% more new postings than April)
      mnc_multiplier          = 2.6    (MNCs survive 2.6x longer than startups)
      tech_roles_total        = 1,036  (across all 3 months)
      india_jobs_total        = 3,672
    """
    april   = cohort[cohort["first_seen_month"] == "April"]
    survived = april["survived_next"].sum()
    gone     = len(april) - survived

    new_april = len(cohort[cohort["first_seen_month"] == "April"])
    new_may   = len(cohort[cohort["first_seen_month"] == "May"])
    new_june  = len(cohort[cohort["first_seen_month"] == "June"])

    mnc_surv = april[april["company_type"] == "MNC"]["survived_next"].mean() * 100
    sme_surv = april[april["company_type"] == "Startup/SME"]["survived_next"].mean() * 100
    multiplier = round(mnc_surv / sme_surv, 1) if sme_surv > 0 else None

    tech_all   = cohort[cohort["role_type"].notna()]
    india_all  = cohort[cohort["is_india"]]

    return {
        "total_unique_jobs":        int(cohort["jobid"].nunique()),
        "april_cohort_size":        int(len(april)),
        "disappeared_before_may":   int(gone),
        "survived_to_may":          int(survived),
        "disappearance_rate_pct":   round(gone / len(april) * 100, 1),
        "survival_rate_pct":        round(survived / len(april) * 100, 1),
        "survived_to_june":         int(april["survived_to_end"].sum()),
        "two_month_survival_pct":   round(april["survived_to_end"].mean() * 100, 1),
        "new_jobs_april":           int(new_april),
        "new_jobs_may":             int(new_may),
        "new_jobs_june":            int(new_june),
        "may_surge_pct":            round((new_may - new_april) / new_april * 100, 1),
        "june_change_pct":          round((new_june - new_may) / new_may * 100, 1),
        "mnc_survival_pct":         round(mnc_surv, 1),
        "startup_survival_pct":     round(sme_surv, 1),
        "mnc_multiplier":           multiplier,
        "tech_roles_total":         int(len(tech_all)),
        "india_jobs_total":         int(len(india_all)),
        "multi_month_jobs":         int((cohort["months_active"] > 1).sum()),
        "multi_month_rate_pct":     round((cohort["months_active"] > 1).mean() * 100, 1),
    }


# ── Section 2: Posting velocity ────────────────────────────────────────────────

def get_posting_velocity(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    New job postings per month, with carry-over from previous month.

    Confirmed:
      April: 15,112 new  |  0 carry-over  |  15,112 total visible
      May:   26,607 new  |  361 carry-over | 26,968 total visible
      June:  18,694 new  |  257 carry-over | 18,951 total visible

    The May surge (+76.1%) is the headline finding from velocity analysis.
    """
    rows = []
    for i, month in enumerate(MONTH_ORDER):
        new_n = int(len(cohort[cohort["first_seen_month"] == month]))

        if i == 0:
            carry = 0
        else:
            prev = MONTH_ORDER[i - 1]
            carry = int(cohort[
                (cohort["first_seen_month"] == prev) &
                cohort["months_list"].str.contains(month)
            ].shape[0])

        rows.append({
            "month":              month,
            "new_postings":       new_n,
            "carried_from_prev":  carry,
            "total_visible":      new_n + carry,
        })

    df = pd.DataFrame(rows)
    df["mom_change_pct"] = df["new_postings"].pct_change().mul(100).round(1)
    # Replace NaN (first row has no previous month) with None for JSON safety
    df["mom_change_pct"] = df["mom_change_pct"].where(df["mom_change_pct"].notna(), other=None)
    return df


# ── Section 3: Survival by role type ──────────────────────────────────────────

def get_survival_by_role(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Survival rates for each tech role type.

    Confirmed values (April cohort, n shown):
      Data Analyst:     1.0%  (n=103)  — fills very fast
      Business Analyst: 1.1%  (n=89)
      Data Scientist:   5.2%  (n=77)
      Product Analyst: 14.3%  (n=7, small sample)
      Data Engineer:    0.0%  (n=10, too small)
      ML Engineer:      0.0%  (n=11, too small)

    The interpretation column is pre-written for dashboard tooltips.
    """
    april_tech = cohort[
        (cohort["first_seen_month"] == "April") &
        cohort["role_type"].notna()
    ].copy()

    summary = (april_tech
        .groupby("role_type")["survived_next"]
        .agg(total="count", survived="sum", rate="mean")
        .reset_index())

    summary["survival_rate_pct"]      = (summary["rate"] * 100).round(1)
    summary["disappearance_rate_pct"] = (100 - summary["survival_rate_pct"]).round(1)

    summary["sample_reliable"] = summary["total"] >= 20
    summary["interpretation"] = summary.apply(_role_interpretation, axis=1)

    return (summary
            .drop(columns="rate")
            .sort_values("survival_rate_pct")
            .reset_index(drop=True))


def _role_interpretation(row) -> str:
    if not row["sample_reliable"]:
        return f"Small sample (n={row['total']}) — directional only"
    r = row["survival_rate_pct"]
    if r < 2:
        return "Very fast market — apply within days of posting"
    if r < 5:
        return "Moderate fill speed — 1-2 week application window"
    return "Slower fill — more time, likely re-listed multiple times"


# ── Section 4: Survival by company type ───────────────────────────────────────

def get_survival_by_company_type(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    MNC vs Startup/SME survival comparison.

    Confirmed:
      MNC:         5.9% survival  (n=562)
      Startup/SME: 2.3% survival  (n=14,464)
      Multiplier:  2.6x

    Strategy implication: apply to MNCs earlier in their posting cycle.
    Apply to startups within the first week — they move faster.
    """
    april = cohort[
        (cohort["first_seen_month"] == "April") &
        (cohort["company_type"] != "Unknown")
    ].copy()

    summary = (april
        .groupby("company_type")["survived_next"]
        .agg(total="count", survived="sum", rate="mean")
        .reset_index())

    summary["survival_rate_pct"]      = (summary["rate"] * 100).round(1)
    summary["disappearance_rate_pct"] = (100 - summary["survival_rate_pct"]).round(1)

    mnc = summary.loc[summary["company_type"] == "MNC", "survival_rate_pct"]
    sme = summary.loc[summary["company_type"] == "Startup/SME", "survival_rate_pct"]
    if len(mnc) and len(sme) and sme.values[0] > 0:
        mult = round(mnc.values[0] / sme.values[0], 1)
        summary["survival_multiplier"] = summary["company_type"].map(
            {"MNC": mult, "Startup/SME": 1.0}
        )

    return (summary
            .drop(columns="rate")
            .sort_values("survival_rate_pct", ascending=False)
            .reset_index(drop=True))


# ── Section 5: Survival by seniority ──────────────────────────────────────────

def get_survival_by_seniority(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Survival rates across seniority levels.

    Confirmed:
      Entry level:      1.7%  (n=1,995)
      Associate:        1.8%  (n=1,386)
      Mid-Senior level: 2.7%  (n=3,213)
      Director:         2.8%  (n=422)

    Note: 'Not Applicable' is the largest category (n=7,754) because
    many companies skip LinkedIn's seniority field.
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()

    summary = (april
        .groupby("seniority")["survived_next"]
        .agg(total="count", survived="sum", rate="mean")
        .reset_index())

    summary["survival_rate_pct"]      = (summary["rate"] * 100).round(1)
    summary["disappearance_rate_pct"] = (100 - summary["survival_rate_pct"]).round(1)

    return (summary
            .drop(columns="rate")
            .sort_values("survival_rate_pct")
            .reset_index(drop=True))


# ── Section 6: Survival by work type ──────────────────────────────────────────

def get_survival_by_worktype(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Full-time vs Contract vs Internship etc.

    Confirmed:
      Internship: 3.2% (highest — company re-lists internships repeatedly)
      Full-time:  2.5%
      Contract:   1.6%
      Part-time:  1.1%
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()

    summary = (april
        .groupby("worktype")["survived_next"]
        .agg(total="count", survived="sum", rate="mean")
        .reset_index())

    summary["survival_rate_pct"]      = (summary["rate"] * 100).round(1)
    summary["disappearance_rate_pct"] = (100 - summary["survival_rate_pct"]).round(1)

    return (summary
            .drop(columns="rate")
            .sort_values("total", ascending=False)
            .reset_index(drop=True))


# ── Section 7: Skill survival comparison ──────────────────────────────────────

def get_skill_survival(skill_surv: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-computed in skill_extractor.py — returned as-is with display labels.

    Confirmed top 3 (positive = longer-lived postings):
      SQL:        +1.78pp  Moderate signal
      Statistics: +1.42pp  Moderate signal
      Spark:      +0.84pp  Moderate signal

    Confirmed bottom 3 (negative = faster-disappearing):
      Power BI:        −0.96pp
      R:               −1.46pp
      Machine Learning:−1.76pp
    """
    df = skill_surv.copy()
    df["bar_color"] = df["survival_advantage"].apply(
        lambda x: "positive" if x > 0 else "negative"
    )
    df["abs_advantage"] = df["survival_advantage"].abs()
    return df


# ── Section 8: Skill frequency by role ────────────────────────────────────────

def get_skill_role_matrix(role_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the role × skill matrix exactly as computed.
    Used for the heatmap and DA/DS comparison chart.

    Key confirmed values:
      Data Analyst:    SQL 60.1%, Statistics 47.4%, Power BI 44.8%, Python 44.4%
      Data Scientist:  Python 89.5%, ML 78.5%, Statistics 76.6%, SQL 68.0%
      Business Analyst:Excel 35.3%, SQL 27.3%
      Product Analyst: SQL 82.6%, Power BI 65.2%, Tableau 65.2%
    """
    return role_matrix.copy()


def get_da_skill_gap(role_matrix: pd.DataFrame,
                     from_role: str = "Data Analyst",
                     to_role: str = "Data Scientist") -> pd.DataFrame:
    """
    Shows the skill gap between two role types.
    Positive gap = to_role needs it more, negative = from_role needs it more.

    Used for the "upgrade your skillset" widget in the dashboard.
    """
    from_row = role_matrix[role_matrix["role_type"] == from_role]
    to_row   = role_matrix[role_matrix["role_type"] == to_role]

    if from_row.empty or to_row.empty:
        return pd.DataFrame()

    skills = [c for c in role_matrix.columns if c != "role_type"]
    from_vals = from_row[skills].values[0]
    to_vals   = to_row[skills].values[0]

    gap_df = pd.DataFrame({
        "skill":      skills,
        f"{from_role}_pct": from_vals,
        f"{to_role}_pct":   to_vals,
        "gap":        to_vals - from_vals,
    })
    gap_df["gap_direction"] = gap_df["gap"].apply(
        lambda g: f"{to_role} needs more" if g > 5
        else f"{from_role} needs more" if g < -5
        else "Similar demand"
    )
    return gap_df.sort_values("gap", ascending=False).reset_index(drop=True)


# ── Section 9: Skill co-occurrence ────────────────────────────────────────────

def get_skill_cooccurrence(cooccur: pd.DataFrame,
                            min_corr: float = 0.3) -> pd.DataFrame:
    """
    Returns skill pairs above the minimum correlation threshold.

    Confirmed top pairs:
      Pandas + NumPy:          r=0.934  (123 postings)
      NumPy + Scikit-learn:    r=0.804  (105 postings)
      Pandas + Scikit-learn:   r=0.789  (108 postings)
      AWS + GCP:               r=0.672  (41 postings)
      BigQuery + Snowflake:    r=0.627  (47 postings)
      Power BI + Tableau:      r=0.610  (data viz pair)
    """
    return cooccur[cooccur["correlation"] >= min_corr].copy().reset_index(drop=True)


# ── Section 10: India analysis ─────────────────────────────────────────────────

def get_india_summary(cohort: pd.DataFrame,
                      india_prof: pd.DataFrame) -> dict:
    """
    All India-specific numbers in one dict.

    Confirmed:
      total india jobs across 3 months: 3,672
      april india cohort:               1,061
      india tech roles:                 43  (small sample — always mention)
      india tech survival:              7.0%
      top cities: Bengaluru, Hyderabad, Mumbai, Gurugram

    India skill emphasis vs global:
      More: Excel (+8.4pp), Tableau (+4.7pp), BigQuery (+3.4pp)
      Less: Statistics (−10.3pp), ML (−10.3pp), Pandas (−8.1pp)
    """
    india_all   = cohort[cohort["is_india"]]
    india_april = cohort[(cohort["first_seen_month"] == "April") & cohort["is_india"]]
    india_tech  = cohort[cohort["is_india"] & cohort["role_type"].notna()]
    india_tech_april = india_tech[india_tech["first_seen_month"] == "April"]

    more_emphasis = india_prof[india_prof["india_vs_global"] > 2].sort_values(
        "india_vs_global", ascending=False
    )[["skill", "india_pct", "global_tech_pct", "india_vs_global"]].reset_index(drop=True)

    less_emphasis = india_prof[india_prof["india_vs_global"] < -2].sort_values(
        "india_vs_global"
    )[["skill", "india_pct", "global_tech_pct", "india_vs_global"]].reset_index(drop=True)

    return {
        "total_india_jobs":          int(len(india_all)),
        "india_april_cohort":        int(len(india_april)),
        "india_tech_roles":          int(len(india_tech)),
        "india_tech_april":          int(len(india_tech_april)),
        "india_survival_rate_pct":   round(india_april["survived_next"].mean() * 100, 1)
                                     if len(india_april) > 0 else None,
        "india_tech_survival_pct":   round(india_tech_april["survived_next"].mean() * 100, 1)
                                     if len(india_tech_april) > 0 else None,
        "top_cities":                india_all["city"].value_counts().head(6).to_dict(),
        "role_distribution":         india_tech["role_type"].value_counts().to_dict(),
        "skills_india_emphasises":   more_emphasis,
        "skills_india_deemphasises": less_emphasis,
        "small_sample_warning":      len(india_tech_april) < 50,
        "sample_caveat":             f"India tech roles: {len(india_tech_april)} in April cohort. "
                                     "Directional signal only — not statistically robust.",
    }


# ── Section 11: Top companies ──────────────────────────────────────────────────

def get_top_surviving_companies(cohort: pd.DataFrame,
                                 min_postings: int = 5,
                                 top_n: int = 10) -> pd.DataFrame:
    """
    Companies whose postings survive longest.

    Filters to min_postings to avoid single-post outliers inflating the rate.
    Result is used in the dashboard's "Who's hiring slowly?" section.

    Note: high survival rate means SLOWER hiring process, not better company.
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()

    summary = (april
        .groupby("company")
        .agg(
            total        = ("jobid",        "count"),
            survived     = ("survived_next", "sum"),
            company_type = ("company_type",  "first"),
        )
        .reset_index())

    summary = summary[summary["total"] >= min_postings]
    summary["survival_rate_pct"] = (summary["survived"] / summary["total"] * 100).round(1)

    return (summary
            .sort_values("survival_rate_pct", ascending=False)
            .head(top_n)
            .reset_index(drop=True))


# ── Section 12: Skills count vs survival ──────────────────────────────────────

def get_skill_count_survival(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Do jobs requiring more skills survive longer?

    Confirmed:
      0 skills: 2.3% survival  (10,502 jobs)
      1 skill:  2.5% survival  (3,547 jobs)
      2 skills: 3.3% survival  (584 jobs)
      3 skills: 2.7% survival  (185 jobs)
      4+ skills:1.4% survival  (294 jobs)

    Insight: 2-skill jobs survive longest. 4+ skill jobs disappear
    fastest — very specialised roles either fill immediately or get pulled.
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()

    april["skill_bucket"] = pd.cut(
        april["skills_count"],
        bins=[-1, 0, 1, 2, 3, 100],
        labels=["0 skills", "1 skill", "2 skills", "3 skills", "4+ skills"],
    )

    result = (april
        .groupby("skill_bucket", observed=True)["survived_next"]
        .agg(total="count", survived="sum", rate="mean")
        .reset_index())

    result["survival_rate_pct"]      = (result["rate"] * 100).round(1)
    result["disappearance_rate_pct"] = (100 - result["survival_rate_pct"]).round(1)

    return result.drop(columns="rate").reset_index(drop=True)


# ── Section 13: Insight payload for Gemini ────────────────────────────────────

def build_insight_payload(cohort: pd.DataFrame,
                           skill_surv: pd.DataFrame,
                           india_prof: pd.DataFrame,
                           target_role: str = "Data Analyst",
                           target_location: str = "India") -> dict:
    """
    Assembles the complete context dict passed to insights.py → Gemini.

    target_role and target_location come from the dashboard user inputs.
    This allows the AI insight to be personalised per user's goal.
    """
    overview = get_overview_metrics(cohort)
    by_role  = get_survival_by_role(cohort)
    by_co    = get_survival_by_company_type(cohort)
    india    = get_india_summary(cohort, india_prof)

    # Top 3 and bottom 3 skills
    top_skills = skill_surv.head(3)[["skill", "survival_advantage",
                                      "signal_strength"]].to_dict(orient="records")
    fast_skills = skill_surv.tail(3)[["skill", "survival_advantage",
                                       "signal_strength"]].to_dict(orient="records")

    # Target role specific numbers
    role_row = by_role[by_role["role_type"] == target_role]
    target_survival = role_row["survival_rate_pct"].values[0] if len(role_row) else None
    target_n        = role_row["total"].values[0] if len(role_row) else None

    return {
        # Global market
        "total_jobs_analysed":          overview["total_unique_jobs"],
        "months_covered":               "April, May, June 2026",
        "april_cohort_size":            overview["april_cohort_size"],
        "disappearance_rate_pct":       overview["disappearance_rate_pct"],
        "survival_rate_pct":            overview["survival_rate_pct"],
        "two_month_survival_pct":       overview["two_month_survival_pct"],
        "may_posting_surge_pct":        overview["may_surge_pct"],
        "june_posting_change_pct":      overview["june_change_pct"],
 
        # ADD THESE THREE ↓↓↓
        "new_jobs_april":               overview["new_jobs_april"],
        "new_jobs_may":                 overview["new_jobs_may"],
        "new_jobs_june":                overview["new_jobs_june"],

        # Company type
        "mnc_survival_pct":             overview["mnc_survival_pct"],
        "startup_survival_pct":         overview["startup_survival_pct"],
        "mnc_multiplier":               overview["mnc_multiplier"],

        # Role-specific
        "target_role":                  target_role,
        "target_role_survival_pct":     target_survival,
        "target_role_sample_n":         target_n,
        "all_role_survival_rates":      by_role.set_index("role_type")
                                               ["survival_rate_pct"].to_dict(),

        # Skills
        "skills_in_long_lived_postings":  top_skills,
        "skills_in_fast_fill_postings":   fast_skills,

        # India
        "target_location":              target_location,
        "india_tech_survival_pct":      india["india_tech_survival_pct"],
        "india_top_cities":             india["top_cities"],
        "india_role_distribution":      india["role_distribution"],
        "india_small_sample_warning":   india["small_sample_warning"],
        "india_sample_caveat":          india["sample_caveat"],

        # Honest limitations
        "data_limitation":
            "Survival = jobid present in next monthly snapshot. "
            "Disappearance ≠ filled — could also mean expired or withdrawn. "
            "Say 'disappeared' not 'filled' when presenting.",
    }


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading all data files...")
    data = load_all()
    print(f"  Loaded {len(data)} DataFrames\n")

    print("=== Overview metrics ===")
    overview = get_overview_metrics(data["cohort"])
    for k, v in overview.items():
        print(f"  {k:<40s} {v}")

    print("\n=== Posting velocity ===")
    print(get_posting_velocity(data["cohort"]).to_string(index=False))

    print("\n=== Survival by role ===")
    print(get_survival_by_role(data["cohort"])
          [["role_type","total","survival_rate_pct","interpretation"]].to_string(index=False))

    print("\n=== Survival by company type ===")
    print(get_survival_by_company_type(data["cohort"]).to_string(index=False))

    print("\n=== DA → DS skill gap ===")
    print(get_da_skill_gap(data["role_matrix"])[["skill","gap","gap_direction"]]
          .head(10).to_string(index=False))

    print("\n=== Skill count vs survival ===")
    print(get_skill_count_survival(data["cohort"]).to_string(index=False))

    print("\n=== India summary ===")
    india = get_india_summary(data["cohort"], data["india_prof"])
    for k, v in india.items():
        if not isinstance(v, pd.DataFrame):
            print(f"  {k:<40s} {v}")
    print("  skills_india_emphasises:")
    print(india["skills_india_emphasises"].to_string(index=False))

    print("\n=== Insight payload (keys) ===")
    payload = build_insight_payload(
        data["cohort"], data["skill_surv"], data["india_prof"],
        target_role="Data Analyst", target_location="India"
    )
    for k, v in payload.items():
        print(f"  {k:<45s} {str(v)[:80]}")

    print("\nAll analytics functions verified.")
