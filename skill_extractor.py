"""
skill_extractor.py
LinkedIn Job Decay Tracker — Skill Intelligence Module

Day 5–6 deliverable.

What this file does:
    Reads combined_jobs.parquet and survival_cohort.parquet,
    then produces every skill-related analysis table used by
    the dashboard and EDA notebook.

Why it's a separate file from survival.py:
    survival.py answers "which jobs disappeared?"
    skill_extractor.py answers "what skills were in those jobs?"
    Keeping them separate means you can update skill patterns
    or add new skills without touching the survival logic.

Run standalone:
    python skill_extractor.py

Output files written to data/:
    skill_survival_comparison.parquet
    skill_role_matrix.parquet
    skill_cooccurrence.parquet
    skill_india_profile.parquet
    skill_report.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
COMBINED_FILE   = DATA_DIR / "combined_jobs.parquet"
COHORT_FILE     = DATA_DIR / "survival_cohort.parquet"
REPORT_FILE     = DATA_DIR / "skill_report.txt"

# ── Skill column registry ──────────────────────────────────────────────────────
# Exact column names as they exist in the parquet files.
# Each maps to a human-readable display name.
SKILL_COLS = {
    "skill_sql":              "SQL",
    "skill_python":           "Python",
    "skill_excel":            "Excel",
    "skill_power_bi":         "Power BI",
    "skill_tableau":          "Tableau",
    "skill_r":                "R",
    "skill_statistics":       "Statistics",
    "skill_machine_learning": "Machine Learning",
    "skill_pandas":           "Pandas",
    "skill_numpy":            "NumPy",
    "skill_scikit_learn":     "Scikit-learn",
    "skill_aws":              "AWS",
    "skill_azure":            "Azure",
    "skill_gcp":              "GCP",
    "skill_spark":            "Spark",
    "skill_airflow":          "Airflow",
    "skill_dbt":              "dbt",
    "skill_looker":           "Looker",
    "skill_bigquery":         "BigQuery",
    "skill_snowflake":        "Snowflake",
}

# Skill groupings — used for category-level analysis
SKILL_CATEGORIES = {
    "Core Analytics":    ["SQL", "Excel", "Statistics"],
    "Visualisation":     ["Power BI", "Tableau", "Looker"],
    "Programming":       ["Python", "R", "Pandas", "NumPy", "Scikit-learn"],
    "Cloud & Warehouse": ["AWS", "Azure", "GCP", "BigQuery", "Snowflake"],
    "Data Engineering":  ["Spark", "Airflow", "dbt"],
    "ML / AI":           ["Machine Learning", "Scikit-learn"],
}

# Month display order
MONTH_ORDER = ["April", "May", "June"]


# ── Helper ─────────────────────────────────────────────────────────────────────

def _col_to_name(col: str) -> str:
    """Converts 'skill_power_bi' → 'Power BI' using the registry."""
    return SKILL_COLS.get(col, col.replace("skill_", "").replace("_", " ").title())


def _get_skill_cols(df: pd.DataFrame) -> list[str]:
    """Returns only the skill_* columns that actually exist in the DataFrame."""
    return [c for c in SKILL_COLS if c in df.columns]


# ── Analysis functions ─────────────────────────────────────────────────────────

def skill_frequency_overall(combined: pd.DataFrame) -> pd.DataFrame:
    """
    How often does each skill appear across all 61,141 job postings?
    Also breaks down by month to show whether demand is rising or falling.

    This answers: "Which skills are most in demand globally?"
    Frequency = % of ALL postings (not just tech roles) that mention the skill.
    """
    cols = _get_skill_cols(combined)
    rows = []

    for col in cols:
        skill = _col_to_name(col)
        total_count    = combined[col].sum()
        total_pct      = combined[col].mean() * 100

        # Monthly breakdown
        monthly = combined.groupby("snapshot_month")[col].mean() * 100
        apr_pct = monthly.get("April", 0)
        may_pct = monthly.get("May",   0)
        jun_pct = monthly.get("June",  0)

        # Month-over-month trend: positive = growing demand
        apr_to_jun_trend = jun_pct - apr_pct

        rows.append({
            "skill":              skill,
            "skill_col":          col,
            "total_mentions":     int(total_count),
            "overall_pct":        round(total_pct, 2),
            "april_pct":          round(apr_pct, 2),
            "may_pct":            round(may_pct, 2),
            "june_pct":           round(jun_pct, 2),
            "apr_to_jun_trend":   round(apr_to_jun_trend, 3),
            "trend_direction":    "↑ Growing" if apr_to_jun_trend > 0.1
                                  else "↓ Declining" if apr_to_jun_trend < -0.1
                                  else "→ Stable",
        })

    return (pd.DataFrame(rows)
            .sort_values("total_mentions", ascending=False)
            .reset_index(drop=True))


def skill_frequency_tech_only(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Same as skill_frequency_overall but filtered to tech roles only
    (DA, DS, BA, DE, ML Engineer, Product Analyst, Analytics Engineer).

    This is what you show in the dashboard — more relevant to your audience.

    Key finding: SQL is #1 (60.1% of DA roles), Python #2 (44.4% of DA roles).
    Statistics appears in 47.4% of DA roles — more than Excel (38.9%).
    """
    tech = combined[combined["role_type"].notna()].copy()
    cols = _get_skill_cols(tech)
    rows = []

    for col in cols:
        skill = _col_to_name(col)
        total_count = tech[col].sum()
        total_pct   = tech[col].mean() * 100

        monthly = tech.groupby("snapshot_month")[col].mean() * 100
        apr_pct = monthly.get("April", 0)
        may_pct = monthly.get("May",   0)
        jun_pct = monthly.get("June",  0)

        rows.append({
            "skill":          skill,
            "skill_col":      col,
            "tech_mentions":  int(total_count),
            "tech_pct":       round(total_pct, 2),
            "april_pct":      round(apr_pct, 2),
            "may_pct":        round(may_pct, 2),
            "june_pct":       round(jun_pct, 2),
            "trend":          round(jun_pct - apr_pct, 3),
        })

    return (pd.DataFrame(rows)
            .sort_values("tech_mentions", ascending=False)
            .reset_index(drop=True))


def skill_survival_comparison(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    THE core skill analysis — the reason this project exists.

    For every skill, compares:
      - % of April jobs that SURVIVED to May AND mentioned that skill
      - % of April jobs that DISAPPEARED by May AND mentioned that skill

    survival_advantage > 0  → skill appears more in long-lived postings
                               (those roles take longer to fill)
    survival_advantage < 0  → skill appears more in fast-disappearing postings
                               (those roles fill quickly — high demand / easy hire)

    IMPORTANT — what this means vs what it doesn't mean:
      Higher survival ≠ more in demand.
      Higher survival = harder to fill OR role gets re-listed repeatedly.
      Lower survival = fast market signal (role fills very quickly).

    Always present both interpretations. Interviewers will test this nuance.
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()
    survived   = april[april["survived_next"] == True]
    disappeared = april[april["survived_next"] == False]

    cols = _get_skill_cols(cohort)
    rows = []

    for col in cols:
        skill = _col_to_name(col)

        s_pct = survived[col].mean()   * 100 if len(survived)    > 0 else 0.0
        d_pct = disappeared[col].mean() * 100 if len(disappeared) > 0 else 0.0
        diff  = s_pct - d_pct

        s_count = int(survived[col].sum())
        d_count = int(disappeared[col].sum())

        rows.append({
            "skill":                  skill,
            "skill_col":              col,
            "in_survived_pct":        round(s_pct, 2),
            "in_disappeared_pct":     round(d_pct, 2),
            "survival_advantage":     round(diff, 3),
            "survived_mentions":      s_count,
            "disappeared_mentions":   d_count,
            "total_mentions_april":   s_count + d_count,
            "signal_strength":        "Strong"   if abs(diff) > 2.0
                                      else "Moderate" if abs(diff) > 0.5
                                      else "Weak",
            "interpretation":
                f"Roles requiring {skill} survive longer (harder to fill or re-listed)"
                if diff > 0.5
                else f"Roles requiring {skill} fill quickly — high market velocity"
                if diff < -0.5
                else f"No meaningful survival difference for {skill}",
        })

    return (pd.DataFrame(rows)
            .sort_values("survival_advantage", ascending=False)
            .reset_index(drop=True))


def skill_by_role_matrix(combined: pd.DataFrame) -> pd.DataFrame:
    """
    For each role type, what % of postings mention each skill?

    Real numbers from your data:
      Data Analyst:    SQL 60.1%, Statistics 47.4%, Power BI 44.8%, Python 44.4%
      Data Scientist:  Python 89.5%, ML 78.5%, Statistics 76.6%, SQL 68.0%
      Business Analyst:SQL 27.3%, Excel 35.3%, Power BI 17.6%
      Product Analyst: SQL 82.6%, Power BI 65.2%, Tableau 65.2%
      Data Engineer:   Python 66.7%, SQL 73.8%, Azure 35.7%, AWS 31.0%

    This table becomes a skill gap roadmap:
    "If you're a DA targeting Product Analyst, add Power BI and Tableau."
    """
    tech = combined[combined["role_type"].notna()].copy()
    cols = _get_skill_cols(tech)

    matrix = (tech
              .groupby("role_type")[cols]
              .mean()
              .mul(100)
              .round(1))

    # Rename skill columns to display names
    matrix.columns = [_col_to_name(c) for c in matrix.columns]
    matrix = matrix.reset_index()

    return matrix


def skill_cooccurrence(combined: pd.DataFrame,
                       min_cooccurrence: int = 5) -> pd.DataFrame:
    """
    Which skills appear together in the same job posting?
    Uses Pearson correlation between skill flag columns.

    Top confirmed pairs from your data:
      Pandas + NumPy            corr = 0.93  (almost always together)
      NumPy + Scikit-learn      corr = 0.80
      Pandas + Scikit-learn     corr = 0.79
      AWS + GCP                 corr = 0.67
      BigQuery + Snowflake      corr = 0.63
      Power BI + Tableau        corr = 0.61

    Insight: if a job asks for Pandas, there's a 93% chance it also asks for NumPy.
    So learning "Pandas" is really learning "the Python data science trio."
    """
    tech = combined[combined["role_type"].notna()].copy()
    cols = _get_skill_cols(tech)

    # Need at least 2 skills to compute co-occurrence
    if len(cols) < 2:
        return pd.DataFrame()

    skill_matrix = tech[cols].astype(int)
    corr = skill_matrix.corr()

    rows = []
    for s1, s2 in combinations(cols, 2):
        both_count = (tech[s1] & tech[s2]).sum()
        if both_count < min_cooccurrence:
            continue

        rows.append({
            "skill_1":           _col_to_name(s1),
            "skill_2":           _col_to_name(s2),
            "correlation":       round(corr.loc[s1, s2], 3),
            "both_mentioned":    int(both_count),
            "pair_label":        f"{_col_to_name(s1)} + {_col_to_name(s2)}",
            "strength":          "Very strong (≥0.8)"  if corr.loc[s1,s2] >= 0.8
                                 else "Strong (≥0.6)"  if corr.loc[s1,s2] >= 0.6
                                 else "Moderate (≥0.4)" if corr.loc[s1,s2] >= 0.4
                                 else "Weak (<0.4)",
        })

    return (pd.DataFrame(rows)
            .sort_values("correlation", ascending=False)
            .reset_index(drop=True))


def skill_india_profile(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Skill demand specifically for India-based tech roles.

    Sample size note: 111 India tech roles across 3 months.
    Directional signal — not statistically robust at the role level.
    Always mention this when presenting India-specific numbers.

    Top India skills confirmed: SQL (53.2%), Python (44.1%), Excel (33.3%),
    Statistics (29.7%), Power BI (27.0%), Tableau (25.2%).
    """
    india_tech = combined[
        combined["is_india"] & combined["role_type"].notna()
    ].copy()

    global_tech = combined[
        combined["role_type"].notna()
    ].copy()

    cols = _get_skill_cols(combined)
    rows = []

    for col in cols:
        skill = _col_to_name(col)
        india_pct  = india_tech[col].mean()  * 100 if len(india_tech)  > 0 else 0.0
        global_pct = global_tech[col].mean() * 100 if len(global_tech) > 0 else 0.0
        india_vs_global = india_pct - global_pct

        rows.append({
            "skill":               skill,
            "india_pct":           round(india_pct, 1),
            "global_tech_pct":     round(global_pct, 1),
            "india_vs_global":     round(india_vs_global, 1),
            "india_count":         int(india_tech[col].sum()),
            "india_emphasis":      "India emphasises more" if india_vs_global > 3
                                   else "India emphasises less" if india_vs_global < -3
                                   else "Similar to global",
        })

    return (pd.DataFrame(rows)
            .sort_values("india_pct", ascending=False)
            .reset_index(drop=True))


def skill_bucket_survival(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Do jobs that require MORE skills survive longer?

    April cohort grouped by number of skills mentioned:
      0 skills → 2.3% survival
      1 skill  → 2.5% survival
      2 skills → 3.3% survival
      3 skills → 2.7% survival
      4+ skills → 1.4% survival  ← interesting: very demanding roles fill fast

    Insight: the sweet spot is 2–3 skills. Highly specialised multi-skill
    roles (4+) disappear fastest — either because they're very rare talent
    that gets snapped up, or because companies can't fill them and re-post
    with simpler requirements.
    """
    april = cohort[cohort["first_seen_month"] == "April"].copy()
    april["skill_bucket"] = pd.cut(
        april["skills_count"],
        bins=[-1, 0, 1, 2, 3, 100],
        labels=["0 skills", "1 skill", "2 skills", "3 skills", "4+ skills"],
    )

    result = (april
              .groupby("skill_bucket", observed=True)["survived_next"]
              .agg(total="count", survived="sum", survival_rate="mean")
              .reset_index())

    result["survival_pct"]      = (result["survival_rate"] * 100).round(1)
    result["disappearance_pct"] = (100 - result["survival_pct"]).round(1)

    return result.drop(columns=["survival_rate"])


def skill_category_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Groups individual skills into categories and computes category-level
    demand in tech roles.

    Categories:
      Core Analytics    → SQL, Excel, Statistics
      Visualisation     → Power BI, Tableau, Looker
      Programming       → Python, R, Pandas, NumPy, Scikit-learn
      Cloud & Warehouse → AWS, Azure, GCP, BigQuery, Snowflake
      Data Engineering  → Spark, Airflow, dbt
      ML / AI           → Machine Learning, Scikit-learn

    A posting counts as "requiring" a category if it mentions at least 1 skill
    from that category.
    """
    tech = combined[combined["role_type"].notna()].copy()
    name_to_col = {v: k for k, v in SKILL_COLS.items()}

    rows = []
    for category, skills in SKILL_CATEGORIES.items():
        valid_cols = [name_to_col[s] for s in skills if name_to_col.get(s) in tech.columns]
        if not valid_cols:
            continue

        # Job requires this category if it mentions at least 1 skill in the group
        tech[f"_cat_{category}"] = tech[valid_cols].any(axis=1)
        pct = tech[f"_cat_{category}"].mean() * 100

        rows.append({
            "category":         category,
            "skills_in_group":  ", ".join(skills),
            "tech_postings_pct": round(pct, 1),
            "tech_count":       int(tech[f"_cat_{category}"].sum()),
        })

    return (pd.DataFrame(rows)
            .sort_values("tech_postings_pct", ascending=False)
            .reset_index(drop=True))


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict[str, pd.DataFrame]:
    """
    Runs all skill analysis functions and saves outputs to data/.
    Returns a dict of all DataFrames for use by other modules.
    """
    if not COMBINED_FILE.exists():
        raise FileNotFoundError(f"{COMBINED_FILE} not found. Run load_data.py first.")
    if not COHORT_FILE.exists():
        raise FileNotFoundError(f"{COHORT_FILE} not found. Run survival.py first.")

    print("Loading data files...")
    combined = pd.read_parquet(COMBINED_FILE)
    cohort   = pd.read_parquet(COHORT_FILE)
    print(f"  combined: {combined.shape[0]:,} rows")
    print(f"  cohort:   {cohort.shape[0]:,} rows\n")

    log = []

    # ── 1. Overall frequency ─────────────────────────────────────────────────
    print("1/7  Overall skill frequency...")
    freq_overall = skill_frequency_overall(combined)
    log.append("=== SKILL FREQUENCY (ALL JOBS) ===")
    log.append(freq_overall[["skill", "total_mentions", "overall_pct",
                              "trend_direction"]].to_string(index=False))

    # ── 2. Tech-only frequency ───────────────────────────────────────────────
    print("2/7  Tech-role skill frequency...")
    freq_tech = skill_frequency_tech_only(combined)
    log.append("\n=== SKILL FREQUENCY (TECH ROLES ONLY) ===")
    log.append(freq_tech[["skill", "tech_mentions", "tech_pct",
                           "april_pct", "may_pct", "june_pct"]].to_string(index=False))

    # ── 3. Survival comparison ───────────────────────────────────────────────
    print("3/7  Skill survival comparison...")
    survival_comp = skill_survival_comparison(cohort)
    log.append("\n=== SKILL SURVIVAL COMPARISON (April cohort) ===")
    log.append(survival_comp[["skill", "in_survived_pct", "in_disappeared_pct",
                               "survival_advantage", "signal_strength",
                               "interpretation"]].to_string(index=False))
    survival_comp.to_parquet(DATA_DIR / "skill_survival_comparison.parquet", index=False)

    # ── 4. Role matrix ───────────────────────────────────────────────────────
    print("4/7  Skill × role type matrix...")
    role_matrix = skill_by_role_matrix(combined)
    log.append("\n=== SKILL × ROLE TYPE MATRIX (% of postings) ===")
    log.append(role_matrix.to_string(index=False))
    role_matrix.to_parquet(DATA_DIR / "skill_role_matrix.parquet", index=False)

    # ── 5. Co-occurrence ─────────────────────────────────────────────────────
    print("5/7  Skill co-occurrence...")
    cooccur = skill_cooccurrence(combined)
    log.append("\n=== SKILL CO-OCCURRENCE (top 15 pairs) ===")
    log.append(cooccur.head(15)[["pair_label", "correlation",
                                  "both_mentioned", "strength"]].to_string(index=False))
    cooccur.to_parquet(DATA_DIR / "skill_cooccurrence.parquet", index=False)

    # ── 6. India profile ─────────────────────────────────────────────────────
    print("6/7  India skill profile...")
    india_prof = skill_india_profile(combined)
    log.append("\n=== INDIA SKILL PROFILE ===")
    log.append(india_prof[["skill", "india_pct", "global_tech_pct",
                             "india_vs_global", "india_emphasis"]].to_string(index=False))
    india_prof.to_parquet(DATA_DIR / "skill_india_profile.parquet", index=False)

    # ── 7. Skill bucket survival ─────────────────────────────────────────────
    print("7/7  Skill count vs survival...")
    bucket_df = skill_bucket_survival(cohort)
    log.append("\n=== SURVIVAL BY SKILLS COUNT BUCKET ===")
    log.append(bucket_df.to_string(index=False))

    # ── Category summary ─────────────────────────────────────────────────────
    cat_df = skill_category_summary(combined)
    log.append("\n=== SKILL CATEGORY DEMAND ===")
    log.append(cat_df.to_string(index=False))

    # ── Save report ──────────────────────────────────────────────────────────
    report = "\n".join(log)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nSkill report saved → {REPORT_FILE}")

    print("\n── Key findings ────────────────────────────────────────────")

    top3 = survival_comp.head(3)
    bot3 = survival_comp.tail(3)
    print("\nSkills in longest-lived postings (take time to fill):")
    for _, row in top3.iterrows():
        print(f"  {row['skill']:20s}  +{row['survival_advantage']:.2f}pp "
              f"survival advantage  [{row['signal_strength']}]")

    print("\nSkills in fastest-disappearing postings (high market velocity):")
    for _, row in bot3.iterrows():
        print(f"  {row['skill']:20s}  {row['survival_advantage']:.2f}pp "
              f"  [{row['signal_strength']}]")

    print("\nTop skill co-occurrence pairs:")
    for _, row in cooccur.head(5).iterrows():
        print(f"  {row['pair_label']:35s}  corr={row['correlation']:.3f}  "
              f"({row['both_mentioned']} postings)")

    print("\nIndia vs global skill emphasis:")
    india_diff = india_prof.sort_values("india_vs_global", ascending=False)
    more = india_diff[india_diff["india_vs_global"] > 3]
    less = india_diff[india_diff["india_vs_global"] < -3]
    for _, row in more.iterrows():
        print(f"  ↑ India emphasises {row['skill']:15s} "
              f"+{row['india_vs_global']:.1f}pp vs global")
    for _, row in less.iterrows():
        print(f"  ↓ India de-emphasises {row['skill']:12s} "
              f"{row['india_vs_global']:.1f}pp vs global")

    return {
        "freq_overall":    freq_overall,
        "freq_tech":       freq_tech,
        "survival_comp":   survival_comp,
        "role_matrix":     role_matrix,
        "cooccurrence":    cooccur,
        "india_profile":   india_prof,
        "bucket_survival": bucket_df,
        "category_summary": cat_df,
    }


if __name__ == "__main__":
    results = run()
