"""
survival.py
LinkedIn Job Decay Tracker — Survival & Decay Analysis

Reads the cleaned combined_jobs.parquet produced by load_data.py
and computes all survival metrics used by the dashboard and AI insight layer.

Every function returns a clean DataFrame — no plotting, no printing.
That separation means you can test each function independently
and swap the frontend (Streamlit → Next.js, etc.) without touching this file.

Run standalone to verify all metrics print correctly:
    python survival.py

Design note on "survival":
    A job posting is considered to have "survived" if its jobid appears
    in a later month's snapshot. Disappearance = either filled OR expired
    OR delisted for other reasons. We use "disappeared" not "filled"
    when presenting results — it's more accurate.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

INPUT_FILE  = Path("data/combined_jobs.parquet")
OUTPUT_FILE = Path("data/survival_cohort.parquet")

# Month order matters for survival logic
MONTH_ORDER = ["April", "May", "June"]


# ── Core cohort builder ────────────────────────────────────────────────────────

def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    The heart of the project. For every unique jobid, determines:
      - first_seen_month: earliest month it appears in
      - last_seen_month:  latest month it appears in
      - months_active:    number of distinct months it appeared in
      - survival_status:  human-readable outcome label
      - survived_next:    boolean — did it appear in the very next month?
      - survived_to_end:  boolean — was it still live in June (the last snapshot)?
      - months_list:      sorted list of months it appeared in

    Returns one row per unique jobid — the "cohort table".
    """
    # Build a lookup: jobid -> set of months it appeared in
    # Use the deduplicated view (one row per jobid per month)
    deduped = (
        df[["jobid", "snapshot_month"]]
        .drop_duplicates()                          # one row per jobid × month
    )

    job_months = (
        deduped
        .groupby("jobid")["snapshot_month"]
        .apply(set)
        .reset_index()
        .rename(columns={"snapshot_month": "months_set"})
    )

    # Attach first appearance metadata (title, company, role_type etc.)
    # Join from the first-seen occurrence of each job
    meta_cols = [
        "jobid", "snapshot_month", "title", "company", "company_type",
        "role_type", "location", "city", "country", "is_india", "is_remote",
        "seniority", "worktype", "skills_str", "skills_count", "posted_days_ago",
    ]
    # Add individual skill flags if they exist
    skill_flag_cols = [c for c in df.columns if c.startswith("skill_")]
    meta_cols += skill_flag_cols

    # Keep only columns that actually exist (defensive — May/June have extra cols)
    meta_cols = [c for c in meta_cols if c in df.columns]

    # For each jobid, take the row from the earliest month
    df_sorted = df.copy()
    df_sorted["_month_ord"] = df_sorted["snapshot_month"].map(
        {m: i for i, m in enumerate(MONTH_ORDER)}
    )
    first_occurrence = (
        df_sorted
        .sort_values("_month_ord")
        .drop_duplicates("jobid", keep="first")
        [meta_cols]
        .rename(columns={"snapshot_month": "first_seen_month"})
    )

    # Merge months_set onto metadata
    cohort = first_occurrence.merge(job_months, on="jobid", how="left")

    # Derived columns
    cohort["months_list"] = cohort["months_set"].apply(
        lambda s: sorted(s, key=lambda m: MONTH_ORDER.index(m) if m in MONTH_ORDER else 99)
    )
    cohort["months_active"] = cohort["months_set"].apply(len)
    cohort["last_seen_month"] = cohort["months_list"].apply(lambda lst: lst[-1])

    # survival_status — the main outcome label
    cohort["survival_status"] = cohort.apply(_assign_survival_status, axis=1)

    # survived_next: appeared in the month immediately after first_seen
    cohort["survived_next"] = cohort.apply(
        lambda row: _next_month(row["first_seen_month"]) in row["months_set"],
        axis=1
    )

    # survived_to_end: still present in June (the final snapshot)
    cohort["survived_to_end"] = cohort["months_set"].apply(lambda s: "June" in s)

    # Drop the intermediate set column (not serialisable to parquet as-is)
    cohort = cohort.drop(columns=["months_set"])

    # Convert months_list to string for parquet compatibility
    cohort["months_list"] = cohort["months_list"].apply(lambda lst: " → ".join(lst))

    return cohort.reset_index(drop=True)


def _next_month(month: str) -> str | None:
    """Returns the month after the given one, or None if it's the last."""
    idx = MONTH_ORDER.index(month) if month in MONTH_ORDER else -1
    if idx == -1 or idx >= len(MONTH_ORDER) - 1:
        return None
    return MONTH_ORDER[idx + 1]


def _assign_survival_status(row) -> str:
    """
    Assigns a human-readable survival label based on which months
    the posting appeared in.

    Labels are designed to be honest about what we know:
    "Disappeared" not "Filled" — because we can't confirm it was filled.
    """
    months = row["months_set"]
    first  = row["first_seen_month"]

    if months == {"April"}:
        return "Disappeared by May"                    # seen Apr only
    if months == {"May"}:
        return "Disappeared by June"                   # seen May only
    if months == {"June"}:
        return "June only (latest snapshot)"           # seen Jun only — can't track
    if months == {"April", "May"}:
        return "Survived April→May, gone by June"
    if months == {"May", "June"}:
        return "Survived May→June"
    if months == {"April", "June"}:
        return "April + June (skipped May snapshot)"   # unusual — note it
    if months == {"April", "May", "June"}:
        return "Survived all 3 months"
    return "Unknown"                                   # shouldn't happen


# ── Survival metric functions ──────────────────────────────────────────────────
# Each function takes the cohort DataFrame and returns a summary DataFrame.
# Designed so the dashboard can call each one independently.

def survival_overview(cohort: pd.DataFrame) -> dict:
    """
    Top-level summary numbers for the dashboard metric cards.

    Returns a dict so values are easy to access by name in Streamlit:
        metrics = survival_overview(cohort)
        st.metric("Disappearance rate", metrics["disappearance_rate_pct"])
    """
    april_cohort = cohort[cohort["first_seen_month"] == "April"]
    may_cohort   = cohort[cohort["first_seen_month"] == "May"]

    # April cohort survival rate (most reliable — 2 follow-up months)
    apr_survived_may  = april_cohort["survived_next"].sum()
    apr_survived_june = april_cohort["survived_to_end"].sum()
    apr_total         = len(april_cohort)
    apr_disappeared   = apr_total - apr_survived_may

    # Overall across all months
    total_jobs    = len(cohort)
    survived_any  = cohort["months_active"].gt(1).sum()

    return {
        # Core disappearance metric
        "total_unique_jobs":       total_jobs,
        "april_cohort_size":       apr_total,
        "april_disappeared_by_may": apr_disappeared,
        "april_survival_rate_pct": round(apr_survived_may / apr_total * 100, 1),
        "april_disappearance_rate_pct": round(apr_disappeared / apr_total * 100, 1),
        "april_survived_to_june":  apr_survived_june,
        "april_2month_survival_pct": round(apr_survived_june / apr_total * 100, 1),

        # Posting velocity
        "new_jobs_april": apr_total,
        "new_jobs_may":   len(may_cohort),
        "new_jobs_june":  len(cohort[cohort["first_seen_month"] == "June"]),

        # Cross-month survivors
        "jobs_survived_at_least_1_month": survived_any,
        "overall_multi_month_rate_pct":
            round(survived_any / total_jobs * 100, 1),
    }


def survival_by_role(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Survival rates broken down by role_type (DA vs DS vs BA etc.).

    Only includes jobs with a classified role_type (tech roles).
    Uses April cohort as the base — most reliable because it has
    two follow-up snapshots to track against.
    """
    tech = cohort[
        cohort["role_type"].notna() &
        (cohort["first_seen_month"] == "April")
    ].copy()

    if len(tech) == 0:
        return pd.DataFrame()

    summary = (
        tech
        .groupby("role_type")
        .agg(
            total          = ("jobid", "count"),
            survived_may   = ("survived_next", "sum"),
            survived_june  = ("survived_to_end", "sum"),
        )
        .reset_index()
    )

    summary["survival_rate_pct"]     = (summary["survived_may"] / summary["total"] * 100).round(1)
    summary["disappearance_rate_pct"] = (100 - summary["survival_rate_pct"]).round(1)
    summary["2month_survival_pct"]   = (summary["survived_june"] / summary["total"] * 100).round(1)

    # Interpretation column — useful for dashboard tooltips
    summary["interpretation"] = summary["survival_rate_pct"].apply(
        lambda r: "Fills very fast (< 5% survive)" if r < 5
        else "Moderate fill rate" if r < 10
        else "Slower to fill (likely re-listed)"
    )

    return summary.sort_values("survival_rate_pct").reset_index(drop=True)


def survival_by_seniority(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Survival rates by seniority level.
    Filters out 'Not Applicable' only if it dominates to the point of
    obscuring the other categories — otherwise keeps it (it's real data).
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()

    summary = (
        april
        .groupby("seniority")
        .agg(
            total        = ("jobid", "count"),
            survived_may = ("survived_next", "sum"),
        )
        .reset_index()
    )
    summary["survival_rate_pct"]      = (summary["survived_may"] / summary["total"] * 100).round(1)
    summary["disappearance_rate_pct"] = (100 - summary["survival_rate_pct"]).round(1)

    return summary.sort_values("survival_rate_pct").reset_index(drop=True)


def survival_by_company_type(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    MNC vs Startup/SME survival comparison.

    Key finding from the data audit:
    MNCs have 7.0% survival rate vs 2.8% for Startup/SME.
    This means MNC postings stay live 2.5x longer — either because
    MNCs fill roles slower, or because they re-list the same position repeatedly.
    """
    april = cohort[
        (cohort["first_seen_month"] == "April") &
        (cohort["company_type"] != "Unknown")
    ].copy()

    summary = (
        april
        .groupby("company_type")
        .agg(
            total        = ("jobid", "count"),
            survived_may = ("survived_next", "sum"),
        )
        .reset_index()
    )
    summary["survival_rate_pct"]      = (summary["survived_may"] / summary["total"] * 100).round(1)
    summary["disappearance_rate_pct"] = (100 - summary["survival_rate_pct"]).round(1)

    # Compute the multiplier for the interview story
    if len(summary) == 2:
        mnc_rate     = summary.loc[summary["company_type"] == "MNC", "survival_rate_pct"].values
        startup_rate = summary.loc[summary["company_type"] == "Startup/SME", "survival_rate_pct"].values
        if len(mnc_rate) and len(startup_rate) and startup_rate[0] > 0:
            summary["survival_multiplier_vs_startup"] = (mnc_rate[0] / startup_rate[0]).round(1)

    return summary.sort_values("survival_rate_pct", ascending=False).reset_index(drop=True)


def survival_by_worktype(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Survival by employment type (Full-time, Contract, Internship etc.).
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()

    summary = (
        april
        .groupby("worktype")
        .agg(
            total        = ("jobid", "count"),
            survived_may = ("survived_next", "sum"),
        )
        .reset_index()
    )
    summary["survival_rate_pct"]      = (summary["survived_may"] / summary["total"] * 100).round(1)
    summary["disappearance_rate_pct"] = (100 - summary["survival_rate_pct"]).round(1)

    return summary.sort_values("total", ascending=False).reset_index(drop=True)


def skill_decay_comparison(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    For each skill, computes what % of survived vs disappeared April jobs
    mentioned that skill in their description.

    A positive 'survival_advantage' means the skill appears more often
    in postings that survived longer — suggesting those roles take longer
    to fill (harder to find the right candidate).

    Note on interpretation:
    Higher survival rate for a skill does NOT mean "this skill is more in demand."
    It means postings requiring that skill take longer to fill. Both interpretations
    are worth presenting — document this nuance in your README.
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()
    survived  = april[april["survived_next"]]
    gone      = april[~april["survived_next"]]

    skill_flag_cols = [c for c in cohort.columns if c.startswith("skill_")]
    if not skill_flag_cols:
        return pd.DataFrame({"note": ["No skill columns found — run load_data.py first"]})

    rows = []
    for col in skill_flag_cols:
        skill_name = col.replace("skill_", "").replace("_", " ").title()
        # Fix specific casing issues from column naming
        skill_name = skill_name.replace("Bi", "BI").replace("Gcp", "GCP") \
                               .replace("Aws", "AWS").replace("Dbt", "dbt") \
                               .replace("Ml", "ML").replace("Numpy", "NumPy") \
                               .replace("Scikit Learn", "Scikit-learn")

        s_pct = survived[col].mean() * 100 if len(survived) > 0 else 0
        g_pct = gone[col].mean()    * 100 if len(gone)     > 0 else 0
        diff  = s_pct - g_pct

        s_count = survived[col].sum()
        g_count = gone[col].sum()

        rows.append({
            "skill":              skill_name,
            "in_survived_pct":    round(s_pct, 2),
            "in_disappeared_pct": round(g_pct, 2),
            "survival_advantage": round(diff, 2),
            "survived_count":     int(s_count),
            "disappeared_count":  int(g_count),
            "total_mentions":     int(s_count + g_count),
        })

    df = pd.DataFrame(rows).sort_values("survival_advantage", ascending=False)
    return df.reset_index(drop=True)


def posting_velocity(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    How many NEW job postings appeared each month (excluding re-appearances).
    Also shows how many from the previous month survived into this one.

    This is the "market activity" chart — useful for showing whether
    the job market was growing or contracting over the 3 months.
    """
    rows = []
    for month in MONTH_ORDER:
        new_this_month = cohort[cohort["first_seen_month"] == month]

        # Jobs that first appeared in the previous month and survived to this one
        prev_month = _prev_month(month)
        if prev_month:
            survived_from_prev = cohort[
                (cohort["first_seen_month"] == prev_month) &
                cohort["months_list"].str.contains(month)
            ]
            survivors = len(survived_from_prev)
        else:
            survivors = 0

        rows.append({
            "month":           month,
            "new_postings":    len(new_this_month),
            "survived_from_prev": survivors,
            "total_visible":   len(new_this_month) + survivors,
        })

    return pd.DataFrame(rows)


def _prev_month(month: str) -> str | None:
    idx = MONTH_ORDER.index(month) if month in MONTH_ORDER else -1
    if idx <= 0:
        return None
    return MONTH_ORDER[idx - 1]


def top_surviving_companies(cohort: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Companies whose postings survive longest (highest survival rate).
    Filtered to companies with at least 5 postings in April
    to avoid single-post outliers skewing the rate.

    Useful for the dashboard — shows which employers have slower
    hiring processes (or re-list aggressively).
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()

    summary = (
        april
        .groupby("company")
        .agg(
            total         = ("jobid", "count"),
            survived      = ("survived_next", "sum"),
            company_type  = ("company_type", "first"),
        )
        .reset_index()
    )

    # Only companies with enough postings for the rate to be meaningful
    summary = summary[summary["total"] >= 5]
    summary["survival_rate_pct"] = (summary["survived"] / summary["total"] * 100).round(1)

    return (
        summary
        .sort_values("survival_rate_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def india_subanalysis(cohort: pd.DataFrame) -> dict:
    """
    India-specific survival metrics.

    Sample size caveat:
    India tech roles in April = approximately 30-40 jobs.
    This is enough to show directional patterns but not enough
    for statistically robust conclusions. Always mention this
    when presenting India-specific numbers.
    """
    india = cohort[cohort["is_india"]].copy()
    india_april = india[india["first_seen_month"] == "April"]
    india_tech  = india_april[india_april["role_type"].notna()]

    return {
        "total_india_jobs":      len(india),
        "india_april_cohort":    len(india_april),
        "india_tech_roles":      len(india_tech),
        "india_survival_rate_pct":
            round(india_april["survived_next"].mean() * 100, 1)
            if len(india_april) > 0 else None,
        "india_tech_survival_pct":
            round(india_tech["survived_next"].mean() * 100, 1)
            if len(india_tech) > 0 else None,
        "india_top_cities":
            india["city"].value_counts().head(5).to_dict(),
        "india_role_distribution":
            india_tech["role_type"].value_counts().to_dict(),
        "small_sample_warning":
            len(india_tech) < 50,
    }


def generate_insight_payload(cohort: pd.DataFrame) -> dict:
    """
    Bundles all key metrics into a single dict for the Gemini insight layer.

    This is what you pass to insights.py when calling the LLM.
    Structured so the LLM gets concrete numbers, not DataFrames.
    """
    overview   = survival_overview(cohort)
    by_role    = survival_by_role(cohort)
    by_company = survival_by_company_type(cohort)
    skills     = skill_decay_comparison(cohort)
    india      = india_subanalysis(cohort)
    velocity   = posting_velocity(cohort)

    # Top 3 skills with positive survival advantage
    top_skills = skills[skills["survival_advantage"] > 0].head(3)
    top_skill_names = top_skills["skill"].tolist()
    top_skill_advantages = top_skills["survival_advantage"].tolist()

    # Bottom 3 skills (appear more in short-lived postings)
    bottom_skills = skills[skills["survival_advantage"] < 0].tail(3)
    fast_fill_skills = bottom_skills["skill"].tolist()

    # Company type rates
    mnc_rate = by_company.loc[
        by_company["company_type"] == "MNC", "survival_rate_pct"
    ].values[0] if "MNC" in by_company["company_type"].values else None

    startup_rate = by_company.loc[
        by_company["company_type"] == "Startup/SME", "survival_rate_pct"
    ].values[0] if "Startup/SME" in by_company["company_type"].values else None

    return {
        "total_jobs_analysed":          overview["total_unique_jobs"],
        "april_cohort_size":            overview["april_cohort_size"],
        "disappearance_rate_pct":       overview["april_disappearance_rate_pct"],
        "survival_rate_pct":            overview["april_survival_rate_pct"],
        "two_month_survival_pct":       overview["april_2month_survival_pct"],
        "new_jobs_april":               overview["new_jobs_april"],
        "new_jobs_may":                 overview["new_jobs_may"],
        "new_jobs_june":                overview["new_jobs_june"],
        "mnc_survival_rate_pct":        mnc_rate,
        "startup_survival_rate_pct":    startup_rate,
        "skills_with_longer_survival":  top_skill_names,
        "skill_survival_advantages":    top_skill_advantages,
        "skills_in_fast_fill_roles":    fast_fill_skills,
        "role_survival_rates":          by_role.set_index("role_type")["survival_rate_pct"].to_dict()
                                        if len(by_role) else {},
        "india_survival_rate_pct":      india["india_survival_rate_pct"],
        "india_top_cities":             india["india_top_cities"],
        "india_small_sample_warning":   india["small_sample_warning"],
        "data_limitation_note":
            "Survival = jobid still present in next monthly snapshot. "
            "This is a proxy for posting longevity — it cannot confirm "
            "whether a role was filled, re-listed, or simply expired.",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    """
    Loads the combined parquet, builds the cohort table,
    saves it, and prints a summary of all key metrics.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            f"Run load_data.py first to generate it."
        )

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  {len(df):,} rows, {df['jobid'].nunique():,} unique jobs\n")

    print("Building survival cohort...")
    cohort = build_cohort(df)
    print(f"  Cohort size: {len(cohort):,} unique jobs\n")

    # Save cohort
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    cohort.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved cohort to: {OUTPUT_FILE}\n")

    # Print all metrics
    print("=" * 60)
    print("SURVIVAL OVERVIEW")
    print("=" * 60)
    overview = survival_overview(cohort)
    for k, v in overview.items():
        print(f"  {k:45s}: {v}")

    print("\n" + "=" * 60)
    print("SURVIVAL BY ROLE TYPE (April cohort)")
    print("=" * 60)
    role_df = survival_by_role(cohort)
    if len(role_df):
        print(role_df[["role_type", "total", "survived_may",
                       "survival_rate_pct", "disappearance_rate_pct"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("SURVIVAL BY COMPANY TYPE (April cohort)")
    print("=" * 60)
    company_df = survival_by_company_type(cohort)
    print(company_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("SKILL DECAY COMPARISON")
    print("=" * 60)
    skill_df = skill_decay_comparison(cohort)
    print(skill_df[["skill", "in_survived_pct", "in_disappeared_pct",
                    "survival_advantage", "total_mentions"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("POSTING VELOCITY")
    print("=" * 60)
    print(posting_velocity(cohort).to_string(index=False))

    print("\n" + "=" * 60)
    print("INDIA SUB-ANALYSIS")
    print("=" * 60)
    india = india_subanalysis(cohort)
    for k, v in india.items():
        print(f"  {k:40s}: {v}")

    return cohort


if __name__ == "__main__":
    cohort = run()
