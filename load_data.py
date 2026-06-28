"""
load_data.py
LinkedIn Job Decay Tracker — Data Loading & Cleaning

Reads three monthly CSV snapshots (April, May, June 2026),
cleans them, classifies roles and companies, extracts skills,
and saves a single combined Parquet file for all downstream analysis.

Run:
    python load_data.py

Output:
    data/combined_jobs.parquet   — full cleaned dataset
    data/load_report.txt         — audit log of every decision made
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/raw")
RAW_DIR     = DATA_DIR / "demo"
OUTPUT_FILE = DATA_DIR / "combined_jobs.parquet"
REPORT_FILE = DATA_DIR / "load_report.txt"

# These are the three files from Kaggle — put them in data/raw/
RAW_FILES = {
    "April": RAW_DIR / "LinkedIn_April.csv",
    "May":   RAW_DIR / "LinkedIn_May_2026.csv",
    "June":  RAW_DIR / "June_2026.csv",
}

# ── Column contract ────────────────────────────────────────────────────────────
# Confirmed from actual data inspection — do not change these names
REQUIRED_COLS = [
    "collectedat",      # ISO timestamp string e.g. "2026-04-27T04:44:38.539Z"
    "company",          # Company name string
    "jobid",            # Integer job ID (unique per posting)
    "title",            # Job title string
    "location",         # Location string e.g. "Bengaluru, Karnataka, India"
    "seniority",        # One of: Entry level, Mid-Senior level, Associate,
                        #         Director, Executive, Internship, Not Applicable
    "worktype",         # One of: Full-time, Part-time, Contract, Internship,
                        #         Temporary, Volunteer, Other
    "descriptiontext",  # Raw job description (used for skill extraction)
    "postedat",         # Relative string e.g. "14 hours ago", "3 days ago"
]

# Columns that exist in May/June but NOT in April — handled gracefully
OPTIONAL_COLS = ["source_file", "descriptionhtml", "companyurl",
                 "joburl", "searchlocation", "searchquery", "searchtimerange"]


# ── Skill patterns ─────────────────────────────────────────────────────────────
# Validated against real descriptiontext — frequencies confirmed in data audit
SKILL_PATTERNS = {
    "SQL":              r"\bsql\b",
    "Python":           r"\bpython\b",
    "Excel":            r"\bexcel\b",
    "Power BI":         r"power\s*bi",
    "Tableau":          r"\btableau\b",
    "R":                r"\br\b",
    "Spark":            r"\bspark\b",
    "dbt":              r"\bdbt\b",
    "Airflow":          r"\bairflow\b",
    "Machine Learning": r"machine\s+learning|\bml\b",
    "Statistics":       r"\bstatistic",
    "AWS":              r"\baws\b",
    "Azure":            r"\bazure\b",
    "GCP":              r"\bgcp\b",
    "Looker":           r"\blooker\b",
    "BigQuery":         r"\bbigquery\b",
    "Snowflake":        r"\bsnowflake\b",
    "Pandas":           r"\bpandas\b",
    "NumPy":            r"\bnumpy\b",
    "Scikit-learn":     r"scikit[\-\s]*learn|sklearn",
}

# ── MNC company names ──────────────────────────────────────────────────────────
# Used for company_type classification (MNC vs Startup/SME)
MNC_KEYWORDS = [
    "google", "microsoft", "amazon", "meta", "apple", "ibm",
    "accenture", "deloitte", "wipro", "infosys", "tcs", "cognizant",
    "capgemini", "kpmg", "pwc", "ernst & young", "ey ", " ey,",
    "mckinsey", "bcg", "bain", "oracle", "sap", "salesforce",
    "jp morgan", "jpmorgan", "goldman sachs", "morgan stanley",
    "barclays", "hsbc", "citibank", "citi ", "ubs", "deutsche bank",
    "johnson & johnson", "pfizer", "novartis", "unilever", "nestle",
    "procter", "samsung", "sony", "lg ", "siemens", "bosch",
    "toyota", "ford ", "general motors", "boeing", "airbus",
    "linkedin", "twitter", "uber", "netflix", "adobe",
]


# ── Pure functions (no side effects — easy to test) ───────────────────────────

def classify_role(title: str) -> str | None:
    """
    Returns a standardised role category from a job title string.
    Returns None if the title doesn't match any known tech/DA category.

    Designed to be called with .apply() on a Series.
    The order of checks matters — more specific patterns first.
    """
    if not isinstance(title, str):
        return None
    t = title.lower()

    if re.search(r"analytics engineer", t):
        return "Analytics Engineer"
    if re.search(r"data engineer|etl developer|pipeline engineer", t):
        return "Data Engineer"
    if re.search(r"machine learning engineer|ml engineer|mlops", t):
        return "ML Engineer"
    if re.search(r"data scien", t):
        return "Data Scientist"
    if re.search(r"product analyst", t):
        return "Product Analyst"
    if re.search(r"business analyst|business analysis", t):
        return "Business Analyst"
    if re.search(r"data analyst|data analysis|analytics analyst", t):
        return "Data Analyst"

    return None


def classify_company(company: str) -> str:
    """
    Returns 'MNC' if the company name matches a known large corporation,
    otherwise 'Startup/SME'.

    Limitation: this is keyword-based, not a verified company database.
    A company called 'Google Analytics Consulting Ltd' would be classified
    as MNC. Document this assumption when presenting.
    """
    if not isinstance(company, str) or company.strip() == "":
        return "Unknown"
    c = company.lower()
    if any(keyword in c for keyword in MNC_KEYWORDS):
        return "MNC"
    return "Startup/SME"


def extract_skills(text: str) -> list[str]:
    """
    Returns a list of skills detected in a job description.
    Regex patterns validated against actual descriptiontext values.
    Returns empty list (not None) for missing/empty descriptions.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    t = text.lower()
    return [skill for skill, pat in SKILL_PATTERNS.items()
            if re.search(pat, t)]


def parse_postedat_days(text: str) -> float | None:
    """
    Converts a relative LinkedIn 'posted at' string to approximate days ago.

    The postedat field is relative to the scrape time, not an absolute date.
    Values like "14 hours ago", "3 days ago", "2 weeks ago" are typical.
    Sub-hour values ("31 minutes ago") return ~0.0 days.

    Returns None if the pattern doesn't match (treat as unknown age).
    """
    if not isinstance(text, str):
        return None
    t = text.lower().strip()

    # Handle "X minutes ago" -> fraction of a day
    m = re.match(r"(\d+)\s*minute", t)
    if m:
        return round(int(m.group(1)) / 1440, 2)

    m = re.match(r"(\d+)\s*(hour|day|week|month)", t)
    if not m:
        return None

    n, unit = int(m.group(1)), m.group(2)
    if unit == "hour":  return round(n / 24, 2)
    if unit == "day":   return float(n)
    if unit == "week":  return float(n * 7)
    if unit == "month": return float(n * 30)
    return None


def extract_country(location: str) -> str:
    """
    Extracts the last comma-separated part of a location string as country.
    e.g. "Bengaluru, Karnataka, India" -> "India"

    Note: US states (TX, CA, NY) appear as the last segment for US locations
    because many US postings don't include 'United States'. This is a known
    limitation of the source data.
    """
    if not isinstance(location, str) or location.strip() == "":
        return "Unknown"
    parts = [p.strip() for p in location.split(",")]
    return parts[-1] if parts else "Unknown"


def extract_city(location: str) -> str:
    """
    Extracts the first comma-separated segment as city.
    e.g. "Bengaluru, Karnataka, India" -> "Bengaluru"
    """
    if not isinstance(location, str) or location.strip() == "":
        return "Unknown"
    return location.split(",")[0].strip()


# ── Loading ────────────────────────────────────────────────────────────────────

def load_single_file(path: Path, month_label: str, log: list) -> pd.DataFrame:
    """
    Loads one CSV file, validates required columns exist,
    adds snapshot_month, and returns a clean DataFrame.

    Uses latin1 encoding — confirmed necessary for these files
    (UTF-8 fails on byte 0x92, a Windows curly-apostrophe).
    """
    log.append(f"\n[{month_label}] Loading: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Place your Kaggle CSV files in: {RAW_DIR.resolve()}"
        )

    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    log.append(f"  Raw rows: {len(df):,} | Columns: {list(df.columns)}")

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{month_label}] Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df["snapshot_month"] = month_label
    return df


def deduplicate(df: pd.DataFrame, month_label: str, log: list) -> pd.DataFrame:
    """
    Removes duplicate jobids within a single month's snapshot.

    Cause: the scraper sometimes collects the same job twice within the
    same batch run (confirmed — same jobid, same collectedat minute).
    Strategy: keep the first occurrence (earliest collectedat).

    Deduplication happens per-month BEFORE combining, so that a job
    appearing in both April and May is NOT treated as a duplicate —
    those cross-month appearances are the survival signal we want to keep.
    """
    before = len(df)
    df = df.sort_values("collectedat").drop_duplicates("jobid", keep="first")
    after = len(df)
    removed = before - after
    log.append(f"  Dedup: removed {removed:,} duplicate jobids within {month_label} "
               f"({before:,} → {after:,} rows)")
    return df


def parse_timestamps(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    Parses the collectedat ISO timestamp string into proper datetime.
    Adds collected_date (date only) for grouping.

    collectedat format: "2026-04-27T04:44:38.539Z" (UTC)
    """
    df["collected_at"] = pd.to_datetime(
        df["collectedat"], utc=True, errors="coerce"
    )
    null_ts = df["collected_at"].isna().sum()
    if null_ts > 0:
        log.append(f"  Warning: {null_ts} rows with unparseable collectedat — kept as NaT")

    df["collected_date"] = df["collected_at"].dt.date
    df["collected_month"] = df["collected_at"].dt.to_period("M").astype(str)
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    Adds all derived columns to the combined DataFrame.
    Every new column is documented with its derivation logic.
    """
    n = len(df)
    log.append(f"\n[Feature Engineering] Processing {n:,} rows")

    # Role type classification
    df["role_type"] = df["title"].apply(classify_role)
    role_counts = df["role_type"].value_counts(dropna=False)
    log.append(f"  role_type distribution:\n{role_counts.to_string()}")

    # Company type classification
    df["company_type"] = df["company"].apply(classify_company)
    log.append(f"  company_type: {df['company_type'].value_counts().to_dict()}")

    # Skill extraction — stored as both list (for analysis) and string (for display)
    log.append("  Extracting skills from descriptiontext...")
    df["skills_list"] = df["descriptiontext"].apply(extract_skills)
    df["skills_str"]  = df["skills_list"].apply(
        lambda x: ", ".join(x) if x else ""
    )
    df["skills_count"] = df["skills_list"].apply(len)
    has_skills = (df["skills_count"] > 0).sum()
    log.append(f"  Jobs with ≥1 skill detected: {has_skills:,} of {n:,} "
               f"({has_skills/n*100:.1f}%)")

    # Individual skill flags — one boolean column per skill
    # Enables fast groupby without re-parsing the list
    for skill in SKILL_PATTERNS:
        col = f"skill_{skill.lower().replace(' ', '_').replace('-', '_')}"
        df[col] = df["skills_list"].apply(lambda lst: skill in lst)

    # Posting age in days (relative to scrape time — see parse_postedat_days docstring)
    df["posted_days_ago"] = df["postedat"].apply(parse_postedat_days)
    log.append(f"  posted_days_ago nulls: {df['posted_days_ago'].isna().sum():,} "
               f"(sub-minute posts and unparseable values)")

    # Geography
    df["country"] = df["location"].apply(extract_country)
    df["city"]    = df["location"].apply(extract_city)

    # is_india / is_remote flags — useful for sub-analysis
    df["is_india"] = (
        df["location"].str.lower().str.contains("india", na=False)
    )
    df["is_remote"] = (
        df["location"].str.lower().str.contains("remote", na=False) |
        df["title"].str.lower().str.contains("remote", na=False)
    )
    log.append(f"  India jobs: {df['is_india'].sum():,} | "
               f"Remote jobs: {df['is_remote'].sum():,}")

    return df


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(raw_files: dict = RAW_FILES) -> pd.DataFrame:
    """
    Full loading pipeline. Returns the cleaned combined DataFrame
    and writes it to OUTPUT_FILE.

    Parameters
    ----------
    raw_files : dict
        Mapping of month_label -> Path to CSV file.
        Override in tests to point at fixtures.
    """
    log = [f"LinkedIn Job Decay Tracker — Data Load Report",
           f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
           "=" * 60]

    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    # ── Step 1: Load each file ─────────────────────────────────────────────────
    monthly_dfs = []
    for month_label, path in raw_files.items():
        df = load_single_file(path, month_label, log)
        df = deduplicate(df, month_label, log)
        df = parse_timestamps(df, log)
        monthly_dfs.append(df)

    # ── Step 2: Combine ────────────────────────────────────────────────────────
    combined = pd.concat(monthly_dfs, ignore_index=True)
    log.append(f"\n[Combine] Total rows after concat: {len(combined):,}")
    log.append(f"  Total unique jobids: {combined['jobid'].nunique():,}")

    # Sanity check: a jobid appearing in both April AND May is correct
    # (it's our survival signal). Only same-month dupes were removed above.
    same_month_dupes = combined.duplicated(subset=["jobid", "snapshot_month"]).sum()
    if same_month_dupes > 0:
        log.append(f"  WARNING: {same_month_dupes} same-month duplicate jobids remain — investigate")

    # ── Step 3: Add features ───────────────────────────────────────────────────
    combined = add_features(combined, log)

    # ── Step 4: Column ordering ────────────────────────────────────────────────
    # Put the most important columns first for readability
    priority_cols = [
        "jobid", "snapshot_month", "title", "company", "company_type",
        "role_type", "location", "city", "country", "is_india", "is_remote",
        "seniority", "worktype", "skills_str", "skills_count",
        "posted_days_ago", "collected_at", "collected_date",
    ]
    other_cols = [c for c in combined.columns if c not in priority_cols]
    combined = combined[priority_cols + other_cols]

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    combined.to_parquet(OUTPUT_FILE, index=False)
    log.append(f"\n[Output] Saved to: {OUTPUT_FILE}")
    log.append(f"  Shape: {combined.shape[0]:,} rows × {combined.shape[1]} columns")

    # Print and save load report
    report = "\n".join(log)
    print(report)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nLoad report saved to: {REPORT_FILE}")

    return combined


# ── Quick validation helper ────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """
    Runs basic sanity checks on the loaded DataFrame.
    Prints warnings for anything suspicious.
    Call this after run() to confirm the output looks correct.
    """
    print("\n── Validation ──────────────────────────────────────────")

    checks = {
        "No null jobids":
            df["jobid"].isna().sum() == 0,
        "snapshot_month has exactly 3 values":
            df["snapshot_month"].nunique() == 3,
        "collectedat parsed successfully (< 1% nulls)":
            df["collected_at"].isna().mean() < 0.01,
        "role_type classified for at least 1% of jobs":
            df["role_type"].notna().mean() > 0.01,
        "At least 1 skill detected in > 20% of jobs":
            (df["skills_count"] > 0).mean() > 0.20,
        "company_type has no 'Unknown' majority":
            (df["company_type"] == "Unknown").mean() < 0.50,
    }

    all_passed = True
    for check, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        if not result:
            all_passed = False
        print(f"  {status}  {check}")

    if all_passed:
        print("\n  All checks passed. Data is ready for survival.py")
    else:
        print("\n  Some checks failed — review load_report.txt before proceeding")

    print(f"\n  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Months: {sorted(df['snapshot_month'].unique())}")
    print(f"  Tech roles: {df['role_type'].notna().sum():,}")
    print(f"  India jobs: {df['is_india'].sum():,}")


if __name__ == "__main__":
    df = run()
    validate(df)
