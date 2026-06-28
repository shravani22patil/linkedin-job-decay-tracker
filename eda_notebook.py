# %% [markdown]
# # LinkedIn Job Decay Tracker — Exploratory Data Analysis
#
# **Author:** Shravani Patil · DBATU, Computer Engineering  
# **Data:** 3 monthly LinkedIn job posting snapshots (April, May, June 2026)  
# **Total records:** 61,141 rows · 60,413 unique job postings  
#
# ---
#
# ## What this notebook does
#
# This EDA notebook documents every analytical decision made during the project —
# the data problems found, how they were handled, and what the numbers actually mean.
#
# **Reading this notebook is how you understand the project without vibe-coding it.**
# Every section has a "Why this matters" note explaining the business logic,
# not just the code.
#
# ### Table of contents
# 1. Environment setup & data loading
# 2. Raw data audit — what we got, what's messy
# 3. Deduplication decision — why we don't just drop all duplicates
# 4. Survival analysis — the core metric
# 5. Role type breakdown
# 6. Skill frequency — global and India
# 7. Skill survival comparison — the key finding
# 8. Skill co-occurrence — what skills travel together
# 9. Company type analysis
# 10. Posting velocity — is the market growing?
# 11. Geographic breakdown
# 12. Seniority analysis
# 13. Key findings summary — interview-ready numbers

# %% [markdown]
# ## 1. Environment setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings
from pathlib import Path
from collections import Counter
from itertools import combinations

warnings.filterwarnings("ignore")

# ── Plot style — dark theme, clean and non-default ────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0F1117",
    "axes.facecolor":    "#131720",
    "axes.edgecolor":    "#1E2330",
    "axes.labelcolor":   "#9CA3AF",
    "axes.titlecolor":   "#E8EAED",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.titlepad":     12,
    "axes.grid":         True,
    "grid.color":        "#1E2330",
    "grid.linewidth":    0.8,
    "text.color":        "#9CA3AF",
    "xtick.color":       "#6B7280",
    "ytick.color":       "#6B7280",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  "#1C1F26",
    "legend.edgecolor":  "#1E2330",
    "legend.labelcolor": "#9CA3AF",
    "figure.dpi":        120,
    "savefig.dpi":       150,
    "savefig.facecolor": "#0F1117",
    "savefig.bbox":      "tight",
})

# Colour palette — used consistently across all charts
C_BLUE    = "#4F8EF7"
C_GREEN   = "#22C55E"
C_RED     = "#EF4444"
C_AMBER   = "#F59E0B"
C_PURPLE  = "#A78BFA"
C_TEAL    = "#2DD4BF"
C_GREY    = "#374151"
C_SUBTLE  = "#1E2330"

MONTH_ORDER  = ["April", "May", "June"]
MONTH_COLORS = {"April": C_BLUE, "May": C_PURPLE, "June": C_TEAL}

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

print("Setup complete.")
print(f"Plot output directory: {PLOT_DIR.resolve()}")

# %% [markdown]
# ## 2. Raw data audit
#
# Before loading the cleaned parquet, let's look at what the raw CSVs contain.
# This section documents why certain cleaning decisions were made.

# %%
# Load raw CSVs to show the audit trail
raw_april = pd.read_csv(DATA_DIR / "raw/LinkedIn_April.csv",  encoding="latin1")
raw_may   = pd.read_csv(DATA_DIR / "raw/LinkedIn_May_2026.csv", encoding="latin1")
raw_june  = pd.read_csv(DATA_DIR / "raw/June_2026.csv",       encoding="latin1")

print("=== RAW DATA AUDIT ===\n")
for name, df in [("April", raw_april), ("May", raw_may), ("June", raw_june)]:
    print(f"{name}:")
    print(f"  Rows:           {len(df):>8,}")
    print(f"  Unique jobids:  {df['jobid'].nunique():>8,}")
    print(f"  Duplicate rows: {df.duplicated('jobid').sum():>8,}  "
          f"← same job scraped twice in one batch run")
    print(f"  Columns:        {df.shape[1]}")
    print(f"  collectedat:    {df['collectedat'].iloc[0][:19]}  (ISO UTC timestamp)")
    print(f"  postedat:       {df['postedat'].iloc[0]!r:30}  ← relative, not absolute")
    print()

# %% [markdown]
# ### Key data quality decisions made in load_data.py
#
# **Decision 1: Encoding = latin1, not UTF-8**  
# The raw CSVs contain Windows-1252 curly apostrophes (byte `0x92`).
# `pd.read_csv(..., encoding='utf-8')` crashes on these.
# `latin1` reads every byte as a character — no crashes, no data loss.
#
# **Decision 2: Deduplicate within months BEFORE combining**  
# 1,057 April rows share a jobid with another April row — same job scraped
# twice in one batch. These are cleaned per-month.  
# A job appearing in BOTH April AND May is NOT a duplicate — that's our
# survival signal. Only same-month dupes get removed.
#
# **Decision 3: postedat is a proxy, not a timestamp**  
# `postedat` says "14 hours ago" — relative to the scrape time, not an
# absolute date. This means we cannot compute exact "days to fill."
# We use monthly snapshot overlap as our survival metric instead.
# Always mention this limitation clearly.

# %%
# Show the postedat issue concretely
print("=== POSTEDAT FIELD — why it can't give exact fill dates ===\n")
print("Sample values:")
print(raw_april["postedat"].value_counts().head(10).to_string())
print()
print("Implication: '14 hours ago' is relative to when the scraper ran.")
print("Two jobs both saying '14 hours ago' could have been posted at different")
print("absolute times if the scraper ran at different times of day.")
print()
print("Our solution: use monthly snapshot overlap as survival proxy.")
print("If jobid appears in April AND May → survived at least 30 days.")
print("If jobid appears only in April → disappeared before the May scrape.")

# %% [markdown]
# ## 3. Load cleaned data

# %%
combined = pd.read_parquet(DATA_DIR / "combined_jobs.parquet")
cohort   = pd.read_parquet(DATA_DIR / "survival_cohort.parquet")

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
skill_flag_cols = list(SKILL_COLS.keys())

print(f"combined_jobs shape:     {combined.shape}")
print(f"survival_cohort shape:   {cohort.shape}")
print(f"Snapshot months:         {sorted(combined['snapshot_month'].unique())}")
print(f"Skill columns detected:  {len(skill_flag_cols)}")
print()
print("Null summary (key columns):")
key_cols = ["jobid","title","company","role_type","location","seniority",
            "worktype","skills_count","collected_at"]
print(combined[key_cols].isna().sum().to_string())

# %% [markdown]
# ## 4. Survival analysis — the core metric
#
# **What "survival" means in this dataset:**  
# A job posting "survives" if its jobid appears in the next monthly snapshot.
# Disappearance = the posting was removed from LinkedIn. This could mean:
# - ✓ Role was filled
# - The posting expired naturally (LinkedIn removes old listings)
# - The company pulled the role (budget freeze, internal hire, etc.)
#
# We say "disappeared" not "filled" — because we can't confirm which reason.
# This honesty matters in interviews.

# %%
# Overall survival breakdown
april_cohort = cohort[cohort["first_seen_month"] == "April"]
total_apr    = len(april_cohort)
survived_may = april_cohort["survived_next"].sum()
gone_may     = total_apr - survived_may

print("=== APRIL COHORT SURVIVAL (most reliable — 2 follow-up months) ===\n")
print(f"April cohort size:         {total_apr:,} unique jobs")
print(f"Still live in May:         {survived_may:,}  ({survived_may/total_apr*100:.1f}%)")
print(f"Disappeared before May:    {gone_may:,}  ({gone_may/total_apr*100:.1f}%)")
print(f"Survived all 3 months:     {cohort['months_active'].eq(3).sum():,}  "
      f"({cohort['months_active'].eq(3).sum()/total_apr*100:.2f}%)")
print()
print("Survival status breakdown:")
print(cohort["survival_status"].value_counts().to_string())

# %%
# Chart 1: Survival funnel
fig, ax = plt.subplots(figsize=(9, 4.5))

stages   = ["April\n(Posted)", "Still live\nin May", "Still live\nin June"]
values   = [total_apr, survived_may, april_cohort["survived_to_end"].sum()]
colors   = [C_BLUE, C_AMBER, C_GREEN]
pcts     = [100, survived_may/total_apr*100,
            april_cohort["survived_to_end"].sum()/total_apr*100]

bars = ax.bar(stages, values, color=colors, width=0.5, zorder=3)

for bar, val, pct in zip(bars, values, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 180,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10.5,
            color="#E8EAED", fontweight="bold")

ax.set_title("Job Posting Survival Funnel — April 2026 Cohort", pad=14)
ax.set_ylabel("Number of job postings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, total_apr * 1.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(PLOT_DIR / "01_survival_funnel.png")
plt.show()
print("Saved → plots/01_survival_funnel.png")

# %% [markdown]
# ## 5. Role type breakdown

# %%
tech = combined[combined["role_type"].notna()].copy()
print(f"Total tech roles across 3 months: {len(tech):,}")
print()
print("By role type and month:")
role_month = tech.groupby(["role_type", "snapshot_month"]).size().unstack(fill_value=0)
role_month = role_month[[m for m in MONTH_ORDER if m in role_month.columns]]
print(role_month.to_string())

# %%
# Chart 2: Role distribution over time
fig, ax = plt.subplots(figsize=(10, 5))

role_order = tech["role_type"].value_counts().index.tolist()
x = np.arange(len(MONTH_ORDER))
n_roles = len(role_order)
bar_w   = 0.13
palette = [C_BLUE, C_GREEN, C_AMBER, C_PURPLE, C_TEAL, C_RED, "#F97316"]

for i, role in enumerate(role_order):
    vals = [tech[(tech["snapshot_month"] == m) & (tech["role_type"] == role)].shape[0]
            for m in MONTH_ORDER]
    offset = (i - n_roles/2 + 0.5) * bar_w
    bars = ax.bar(x + offset, vals, bar_w, label=role,
                  color=palette[i % len(palette)], zorder=3, alpha=0.92)

ax.set_xticks(x)
ax.set_xticklabels(MONTH_ORDER)
ax.set_title("Tech Role Postings by Month")
ax.set_ylabel("Number of postings")
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_role_distribution.png")
plt.show()

# %%
# Survival by role type — the comparison recruiters care about
tech_april = cohort[
    (cohort["first_seen_month"] == "April") & cohort["role_type"].notna()
].copy()

role_survival = (tech_april
    .groupby("role_type")["survived_next"]
    .agg(total="count", survived="sum", rate="mean")
    .reset_index())
role_survival["survival_pct"]      = (role_survival["rate"] * 100).round(1)
role_survival["disappearance_pct"] = (100 - role_survival["survival_pct"]).round(1)
role_survival = role_survival.sort_values("survival_pct")

print("=== SURVIVAL BY ROLE TYPE (April cohort) ===\n")
print("Interpretation: lower survival = faster disappearance = higher market velocity")
print()
print(role_survival[["role_type","total","survived","survival_pct",
                      "disappearance_pct"]].to_string(index=False))

# %%
# Chart 3: Role survival comparison — horizontal bar chart
fig, ax = plt.subplots(figsize=(9, 4.5))

colors_bar = [C_GREEN if p >= 5 else C_AMBER if p >= 2 else C_RED
              for p in role_survival["survival_pct"]]

bars = ax.barh(role_survival["role_type"], role_survival["survival_pct"],
               color=colors_bar, height=0.55, zorder=3)

for bar, val, total in zip(bars,
                            role_survival["survival_pct"],
                            role_survival["total"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%  (n={total})",
            va="center", ha="left", fontsize=9, color="#9CA3AF")

ax.set_title("30-Day Survival Rate by Role Type (April cohort)")
ax.set_xlabel("% of April postings still live in May")
ax.set_xlim(0, role_survival["survival_pct"].max() * 1.4)
ax.axvline(role_survival["survival_pct"].mean(), color=C_GREY,
           linestyle="--", linewidth=1.2, label="Average")
ax.legend(fontsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_survival_by_role.png")
plt.show()

# %% [markdown]
# ## 6. Skill frequency
#
# Important framing: skill frequency tells you **what employers ask for**,
# not what's rarest or most valuable. SQL is #1 for DA roles (60%) because
# it's a baseline requirement — not because it's a differentiator.

# %%
# Load pre-computed skill tables
try:
    skill_surv  = pd.read_parquet(DATA_DIR / "skill_survival_comparison.parquet")
    role_matrix = pd.read_parquet(DATA_DIR / "skill_role_matrix.parquet")
    cooccur     = pd.read_parquet(DATA_DIR / "skill_cooccurrence.parquet")
    india_prof  = pd.read_parquet(DATA_DIR / "skill_india_profile.parquet")
    print("Pre-computed skill files loaded successfully.")
except FileNotFoundError:
    print("Run skill_extractor.py first to generate the skill parquet files.")
    raise

# %%
# Overall skill frequency in ALL job postings
print("=== SKILL FREQUENCY IN ALL 61,141 POSTINGS ===\n")
freq = combined[skill_flag_cols].sum().rename(index=SKILL_COLS).sort_values(ascending=False)
for skill, count in freq.items():
    pct = count / len(combined) * 100
    bar = "█" * int(pct * 2)
    print(f"  {skill:20s}  {count:5,}  ({pct:5.1f}%)  {bar}")

# %%
# Skill frequency in TECH roles only (more relevant)
print("=== SKILL FREQUENCY IN TECH ROLES ONLY (946 postings) ===\n")
freq_tech = tech[skill_flag_cols].sum().rename(index=SKILL_COLS).sort_values(ascending=False)
for skill, count in freq_tech.items():
    pct = count / len(tech) * 100
    bar = "█" * int(pct / 2)
    print(f"  {skill:20s}  {count:4,}  ({pct:5.1f}%)  {bar}")

# %%
# Chart 4: Skill frequency in tech roles — horizontal bars
fig, ax = plt.subplots(figsize=(10, 7))

skill_names = freq_tech.index.tolist()
skill_vals  = freq_tech.values
skill_pcts  = skill_vals / len(tech) * 100

# Colour by category
cat_colors = {
    "SQL": C_BLUE, "Python": C_BLUE, "Statistics": C_BLUE, "Excel": C_BLUE,
    "Power BI": C_PURPLE, "Tableau": C_PURPLE, "Looker": C_PURPLE,
    "Pandas": C_GREEN, "NumPy": C_GREEN, "Scikit-learn": C_GREEN, "R": C_GREEN,
    "Machine Learning": C_AMBER,
    "AWS": C_TEAL, "Azure": C_TEAL, "GCP": C_TEAL,
    "BigQuery": C_TEAL, "Snowflake": C_TEAL,
    "Spark": C_RED, "Airflow": C_RED, "dbt": C_RED,
}
bar_colors = [cat_colors.get(s, C_GREY) for s in skill_names]

bars = ax.barh(skill_names[::-1], skill_pcts[::-1],
               color=bar_colors[::-1], height=0.65, zorder=3)

for bar, pct in zip(bars, skill_pcts[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}%", va="center", ha="left", fontsize=8.5, color="#9CA3AF")

legend_patches = [
    mpatches.Patch(color=C_BLUE,   label="Core Analytics (SQL, Python, Stats)"),
    mpatches.Patch(color=C_PURPLE, label="Visualisation (Power BI, Tableau, Looker)"),
    mpatches.Patch(color=C_GREEN,  label="Programming (Pandas, NumPy, R, Sklearn)"),
    mpatches.Patch(color=C_TEAL,   label="Cloud & Warehouse"),
    mpatches.Patch(color=C_AMBER,  label="ML / AI"),
    mpatches.Patch(color=C_RED,    label="Data Engineering"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=7.5, framealpha=0.9)
ax.set_title("Skill Frequency in Tech Role Postings (% mentioning skill)")
ax.set_xlabel("% of tech postings")
ax.set_xlim(0, 115)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_skill_frequency_tech.png")
plt.show()

# %% [markdown]
# ## 7. Skill survival comparison — THE key finding
#
# **This is the most interview-worthy analysis in the project.**
#
# We compare: for each skill, what % of postings that SURVIVED vs DISAPPEARED
# mentioned that skill?
#
# A positive survival_advantage means that skill appears more often in
# long-lived postings. Two interpretations:
# - The role is harder to fill (company struggles to find the right candidate)
# - The company re-lists the role repeatedly (slower hiring process)
#
# A negative survival_advantage means that skill appears more in fast-disappearing
# postings — those roles fill quickly (either high supply of candidates OR
# the role was pulled for non-fill reasons).

# %%
print("=== SKILL SURVIVAL COMPARISON ===")
print("Positive advantage = skill appears more in SURVIVED postings")
print("Negative advantage = skill appears more in DISAPPEARED postings\n")
print(skill_surv[["skill", "in_survived_pct", "in_disappeared_pct",
                   "survival_advantage", "signal_strength",
                   "interpretation"]].to_string(index=False))

# %%
# Chart 5: Skill survival advantage — the headline chart
fig, ax = plt.subplots(figsize=(10, 7))

df_plot = skill_surv.sort_values("survival_advantage")
skills  = df_plot["skill"].tolist()
advs    = df_plot["survival_advantage"].tolist()
colors  = [C_GREEN if a > 0 else C_RED for a in advs]

bars = ax.barh(skills, advs, color=colors, height=0.6, zorder=3, alpha=0.9)

for bar, val in zip(bars, advs):
    xpos = val + 0.04 if val >= 0 else val - 0.04
    ha   = "left" if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{val:+.2f}pp", va="center", ha=ha, fontsize=8.5, color="#9CA3AF")

ax.axvline(0, color="#4B5563", linewidth=1.2, linestyle="-")
ax.set_title("Skill Survival Advantage (April cohort)\n"
             "Green = more common in long-lived postings · "
             "Red = more common in fast-disappearing postings",
             fontsize=11)
ax.set_xlabel("Survival advantage (percentage points)")

pos_patch = mpatches.Patch(color=C_GREEN, label="Appears more in survived postings")
neg_patch = mpatches.Patch(color=C_RED,   label="Appears more in disappeared postings")
ax.legend(handles=[pos_patch, neg_patch], fontsize=8.5, loc="lower right")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_skill_survival_advantage.png")
plt.show()

# %% [markdown]
# ## 8. Skills by role type — role gap matrix

# %%
print("=== SKILL × ROLE TYPE MATRIX (% of role postings mentioning skill) ===\n")
print(role_matrix.set_index("role_type").T.to_string())

# %%
# Chart 6: Heatmap — skills by role type
import matplotlib.colors as mcolors

role_m = role_matrix.set_index("role_type")
fig, ax = plt.subplots(figsize=(14, 5.5))

data = role_m.values
im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=data.max())

# Labels
ax.set_xticks(range(len(role_m.index)))
ax.set_xticklabels(role_m.index, rotation=25, ha="right", fontsize=9)
ax.set_yticks(range(len(role_m.columns)))
ax.set_yticklabels(role_m.columns, fontsize=8.5)

# Cell values
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        text_color = "white" if val > 50 else "#9CA3AF"
        ax.text(i, j, f"{val:.0f}%", ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="bold")

plt.colorbar(im, ax=ax, shrink=0.8, label="% of postings")
ax.set_title("Skill Demand by Role Type (% of postings mentioning each skill)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_skill_role_heatmap.png")
plt.show()

# %%
# Chart 7: DA vs DS skill profile comparison — useful for "which track to take"
fig, ax = plt.subplots(figsize=(10, 5.5))

da_row = role_matrix[role_matrix["role_type"] == "Data Analyst"].drop(columns="role_type")
ds_row = role_matrix[role_matrix["role_type"] == "Data Scientist"].drop(columns="role_type")

if not da_row.empty and not ds_row.empty:
    skills_list = da_row.columns.tolist()
    da_vals = da_row.values[0]
    ds_vals = ds_row.values[0]
    x = np.arange(len(skills_list))

    ax.bar(x - 0.2, da_vals, 0.38, label="Data Analyst",  color=C_BLUE,   alpha=0.9, zorder=3)
    ax.bar(x + 0.2, ds_vals, 0.38, label="Data Scientist", color=C_PURPLE, alpha=0.9, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(skills_list, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("% of postings")
    ax.set_title("DA vs Data Scientist Skill Profile — where paths diverge")
    ax.legend(fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_da_vs_ds_profile.png")
    plt.show()

# %% [markdown]
# ## 9. Skill co-occurrence
#
# Which skills appear together in the same posting?
# This is important for study planning — learning one skill often implies needing another.

# %%
print("=== TOP SKILL CO-OCCURRENCE PAIRS ===\n")
print("Correlation = how often two skills appear in the same posting")
print("corr > 0.8 = almost always together\n")
print(cooccur[["pair_label","correlation","both_mentioned","strength"]].head(12).to_string(index=False))

# %%
# Chart 8: Co-occurrence scatter
fig, ax = plt.subplots(figsize=(9, 5))

top_pairs = cooccur.head(12)
x = range(len(top_pairs))

scatter = ax.scatter(x, top_pairs["correlation"],
                     s=top_pairs["both_mentioned"] * 3,
                     c=top_pairs["correlation"],
                     cmap="coolwarm", alpha=0.85, zorder=3)

for i, (_, row) in enumerate(top_pairs.iterrows()):
    ax.annotate(row["pair_label"],
                (i, row["correlation"]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=7.5, color="#9CA3AF", rotation=25)

ax.set_xticks([])
ax.set_ylabel("Pearson correlation")
ax.set_title("Top Skill Co-occurrence Pairs\n(bubble size = number of postings mentioning both)")
ax.axhline(0.8, color=C_AMBER, linestyle="--", linewidth=1, alpha=0.7, label="r = 0.80 threshold")
ax.legend(fontsize=8.5)
plt.colorbar(scatter, ax=ax, label="Correlation strength", shrink=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_skill_cooccurrence.png")
plt.show()

# %% [markdown]
# ## 10. Company type analysis

# %%
company_april = cohort[cohort["first_seen_month"] == "April"].copy()
comp_surv = (company_april[company_april["company_type"] != "Unknown"]
             .groupby("company_type")["survived_next"]
             .agg(total="count", survived="sum", rate="mean")
             .reset_index())
comp_surv["survival_pct"] = (comp_surv["rate"] * 100).round(2)
comp_surv["disappearance_pct"] = (100 - comp_surv["survival_pct"]).round(2)

print("=== COMPANY TYPE SURVIVAL ===\n")
print(comp_surv[["company_type","total","survived",
                  "survival_pct","disappearance_pct"]].to_string(index=False))
print()

if len(comp_surv) == 2:
    mnc_rate = comp_surv.loc[comp_surv["company_type"]=="MNC","survival_pct"].values[0]
    sme_rate = comp_surv.loc[comp_surv["company_type"]=="Startup/SME","survival_pct"].values[0]
    mult = mnc_rate / sme_rate if sme_rate > 0 else None
    print(f"MNC postings survive {mult:.1f}x longer than Startup/SME postings.")
    print(f"Possible reasons:")
    print(f"  1. MNCs have slower hiring processes (more approval layers)")
    print(f"  2. MNCs re-list the same role multiple times")
    print(f"  3. Startups fill roles faster (leaner decision-making)")
    print(f"  4. Startups pull roles faster when conditions change")
    print(f"\nWe cannot determine which reason dominates — honest limitation.")

# %%
# Chart 9: Company type comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Survival rate bars
ax1 = axes[0]
bars = ax1.bar(comp_surv["company_type"], comp_surv["survival_pct"],
               color=[C_BLUE, C_AMBER], width=0.45, zorder=3)
for bar, val in zip(bars, comp_surv["survival_pct"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.1f}%", ha="center", fontsize=11, color="#E8EAED", fontweight="bold")
ax1.set_title("30-Day Survival Rate")
ax1.set_ylabel("% still live in May")
ax1.set_ylim(0, comp_surv["survival_pct"].max() * 1.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Posting volume
ax2 = axes[1]
bars2 = ax2.bar(comp_surv["company_type"], comp_surv["total"],
                color=[C_BLUE, C_AMBER], width=0.45, zorder=3)
for bar, val in zip(bars2, comp_surv["total"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f"{val:,}", ha="center", fontsize=11, color="#E8EAED", fontweight="bold")
ax2.set_title("April Posting Volume")
ax2.set_ylabel("Number of postings")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("MNC vs Startup/SME — Survival & Volume", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(PLOT_DIR / "09_company_type.png")
plt.show()

# %% [markdown]
# ## 11. Posting velocity — market activity over time

# %%
velocity = pd.DataFrame({
    "month":        MONTH_ORDER,
    "new_postings": [
        len(cohort[cohort["first_seen_month"] == m]) for m in MONTH_ORDER
    ],
})

print("=== POSTING VELOCITY — NEW JOBS PER MONTH ===\n")
print(velocity.to_string(index=False))
print()
apr_to_may = (velocity.loc[1,"new_postings"] - velocity.loc[0,"new_postings"])
may_to_jun = (velocity.loc[2,"new_postings"] - velocity.loc[1,"new_postings"])
print(f"April → May:  {apr_to_may:+,} new postings  "
      f"({apr_to_may/velocity.loc[0,'new_postings']*100:+.1f}%)")
print(f"May → June:   {may_to_jun:+,} new postings  "
      f"({may_to_jun/velocity.loc[1,'new_postings']*100:+.1f}%)")
print()
print("Interpretation: May saw a hiring surge (+76% new postings).")
print("June contracted by 30% — possible seasonal hiring slowdown.")

# %%
# Chart 10: Posting velocity line chart
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(velocity["month"], velocity["new_postings"],
        color=C_BLUE, linewidth=2.5, marker="o", markersize=9, zorder=3)

for _, row in velocity.iterrows():
    ax.annotate(f"{row['new_postings']:,}",
                (row["month"], row["new_postings"]),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10.5, color="#E8EAED", fontweight="bold")

ax.fill_between(velocity["month"], velocity["new_postings"],
                alpha=0.12, color=C_BLUE)

ax.set_title("New Job Posting Velocity by Month")
ax.set_ylabel("New unique job postings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, velocity["new_postings"].max() * 1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "10_posting_velocity.png")
plt.show()

# %% [markdown]
# ## 12. Geographic breakdown

# %%
tech_cohort = cohort[cohort["role_type"].notna()].copy()

print("=== TOP COUNTRIES FOR TECH ROLES ===")
print(tech_cohort["country"].value_counts().head(12).to_string())
print()
print("=== INDIA TECH ROLE BREAKDOWN ===")
india_tech = tech_cohort[tech_cohort["is_india"]].copy()
print(f"Total India tech roles: {len(india_tech)}")
print(f"India cities:")
print(india_tech["city"].value_counts().head(10).to_string())

# %%
print("\n=== INDIA SKILL PROFILE vs GLOBAL ===\n")
print("Skills India emphasises MORE than global average:")
more_india = india_prof[india_prof["india_vs_global"] > 2].sort_values("india_vs_global", ascending=False)
print(more_india[["skill","india_pct","global_tech_pct","india_vs_global"]].to_string(index=False))

print("\nSkills India emphasises LESS than global average:")
less_india = india_prof[india_prof["india_vs_global"] < -2].sort_values("india_vs_global")
print(less_india[["skill","india_pct","global_tech_pct","india_vs_global"]].to_string(index=False))

print("\nNote: India sample = 111 tech roles. Directional signal, not statistically robust.")

# %% [markdown]
# ## 13. Seniority analysis

# %%
print("=== SENIORITY IN TECH ROLES ===")
print(tech["seniority"].value_counts().to_string())
print()

seniority_surv = (tech_april
    .groupby("seniority")["survived_next"]
    .agg(total="count", survived="sum", rate="mean")
    .reset_index())
seniority_surv["survival_pct"] = (seniority_surv["rate"] * 100).round(1)
seniority_surv = seniority_surv.sort_values("survival_pct", ascending=False)

print("\n=== SENIORITY SURVIVAL (April tech cohort) ===")
print(seniority_surv[["seniority","total","survived","survival_pct"]].to_string(index=False))
print()
print("Note: 'Not Applicable' dominates because LinkedIn lets companies skip this field.")

# %%
# Chart 11: Seniority survival
fig, ax = plt.subplots(figsize=(9, 4))

colors_sen = [C_GREEN if v > 2 else C_AMBER if v > 0 else C_GREY
              for v in seniority_surv["survival_pct"]]
bars = ax.bar(seniority_surv["seniority"], seniority_surv["survival_pct"],
              color=colors_sen, width=0.55, zorder=3)

for bar, val, n in zip(bars, seniority_surv["survival_pct"], seniority_surv["total"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.1f}%\n(n={n})",
            ha="center", fontsize=8.5, color="#9CA3AF")

ax.set_title("30-Day Survival Rate by Seniority Level (April tech cohort)")
ax.set_ylabel("% still live in May")
ax.tick_params(axis="x", rotation=20)
ax.set_ylim(0, seniority_surv["survival_pct"].max() * 1.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "11_seniority_survival.png")
plt.show()

# %% [markdown]
# ## 14. Key findings summary
#
# Everything below is what you can quote in interviews.
# Every number comes from this notebook and is traceable to a specific chart above.

# %%
print("=" * 65)
print("KEY FINDINGS — LINKEDIN JOB DECAY TRACKER")
print("=" * 65)

# Recompute all headline numbers so they're in one place
total_jobs      = len(cohort)
apr_total       = len(cohort[cohort["first_seen_month"] == "April"])
apr_survived    = cohort[cohort["first_seen_month"] == "April"]["survived_next"].sum()
apr_gone        = apr_total - apr_survived
dis_rate        = apr_gone / apr_total * 100

mnc_rate_val    = comp_surv.loc[comp_surv["company_type"]=="MNC","survival_pct"].values[0]
sme_rate_val    = comp_surv.loc[comp_surv["company_type"]=="Startup/SME","survival_pct"].values[0]
multiplier      = mnc_rate_val / sme_rate_val

top_skill_adv   = skill_surv.iloc[0]
bot_skill       = skill_surv.iloc[-1]

may_new   = len(cohort[cohort["first_seen_month"] == "May"])
apr_new   = len(cohort[cohort["first_seen_month"] == "April"])
may_surge = (may_new - apr_new) / apr_new * 100

print(f"""
FINDING 1 — Market Turnover Speed
  {dis_rate:.1f}% of April job postings disappeared before the May snapshot
  Only {100-dis_rate:.1f}% of postings survived beyond 30 days
  The LinkedIn job market turns over almost completely every month

FINDING 2 — MNC vs Startup Longevity
  MNC postings: {mnc_rate_val:.1f}% survival rate
  Startup/SME:  {sme_rate_val:.1f}% survival rate
  MNCs survive {multiplier:.1f}x longer — slower hiring OR repeated re-listing

FINDING 3 — Skill Survival Signals
  Skills in longest-lived postings (take time to fill):
    {skill_surv.iloc[0]['skill']:20s}  +{skill_surv.iloc[0]['survival_advantage']:.2f}pp advantage
    {skill_surv.iloc[1]['skill']:20s}  +{skill_surv.iloc[1]['survival_advantage']:.2f}pp advantage
    {skill_surv.iloc[2]['skill']:20s}  +{skill_surv.iloc[2]['survival_advantage']:.2f}pp advantage

  Skills in fastest-disappearing postings:
    {skill_surv.iloc[-1]['skill']:20s}  {skill_surv.iloc[-1]['survival_advantage']:.2f}pp
    {skill_surv.iloc[-2]['skill']:20s}  {skill_surv.iloc[-2]['survival_advantage']:.2f}pp
    {skill_surv.iloc[-3]['skill']:20s}  {skill_surv.iloc[-3]['survival_advantage']:.2f}pp

FINDING 4 — DA Roles Disappear Fastest
  Of all tech categories, Data Analyst has the lowest survival rate
  among those with enough April postings for reliable measurement.
  Implication: apply to DA roles early — they move fast.

FINDING 5 — May Hiring Surge
  {int(may_new):,} new postings appeared in May vs {int(apr_new):,} in April (+{may_surge:.0f}%)
  This suggests a seasonal hiring wave in May in this dataset.

FINDING 6 — India Skill Emphasis
  India emphasises Excel (+8.4pp vs global), Tableau (+4.7pp)
  India de-emphasises Statistics (−10.3pp), ML (−10.3pp), Pandas (−8.1pp)
  Sample caveat: 111 India tech roles — directional signal only.

FINDING 7 — Skill Co-occurrence
  Pandas + NumPy co-occur with correlation r=0.93
  Learning Pandas effectively means learning "the Python trio"
  (Pandas + NumPy + Scikit-learn)

DATA LIMITATION — always state this:
  "Survival" = jobid still present in next monthly snapshot.
  We cannot confirm whether a posting was filled, expired, or withdrawn.
  We say "disappeared" not "filled" throughout this analysis.
""")

print("=" * 65)
print(f"Charts saved to: {PLOT_DIR.resolve()}")
print(f"Files used:")
print(f"  {(DATA_DIR / 'combined_jobs.parquet').resolve()}")
print(f"  {(DATA_DIR / 'survival_cohort.parquet').resolve()}")
print(f"  {(DATA_DIR / 'skill_survival_comparison.parquet').resolve()}")
print(f"  {(DATA_DIR / 'skill_role_matrix.parquet').resolve()}")
print(f"  {(DATA_DIR / 'skill_cooccurrence.parquet').resolve()}")
print(f"  {(DATA_DIR / 'skill_india_profile.parquet').resolve()}")
print("=" * 65)
