"""
insights.py
LinkedIn Job Decay Tracker — AI Insight Layer (Day 10–11)

Calls Gemini 1.5 Flash with carefully engineered prompts to turn
the analytics numbers into stakeholder-ready natural language insights.

Three insight types:
  1. market_pulse    — Weekly market summary (main dashboard insight)
  2. role_advice     — Personalised advice for a specific role + location
  3. skill_roadmap   — What to learn next based on target role

Why Gemini 1.5 Flash (not GPT-4):
  - Free tier via Google AI Studio
  - Fast (< 3 seconds per call)
  - Sufficient quality for structured analytical summaries
  - No cost during project development or demo

Prompt engineering principles used here:
  1. Persona — "senior labour market analyst" sets the tone
  2. Structured input — real numbers injected into every prompt
  3. Output constraints — paragraph count, word limit, no bullet points
  4. Caveat injection — data limitations written into the prompt
     so the model cites them automatically
  5. Audience awareness — "non-technical hiring manager" vs "final-year student"

Setup:
  1. Get free API key: https://aistudio.google.com/app/apikey
  2. Add to .env: GEMINI_API_KEY=your_key_here
  3. pip install google-generativeai python-dotenv

Usage:
  from insights import generate_market_pulse, generate_role_advice
  pulse = generate_market_pulse(payload)
  advice = generate_role_advice(payload, "Data Analyst", "India")
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ── Gemini model setup ─────────────────────────────────────────────────────────

_model = None  # lazy-loaded on first call

def _get_model():
    """
    Lazy-loads the Gemini model on first call.
    Returns None (with a clear error message) if the API key is missing.
    """
    global _model
    if _model is not None:
        return _model

    if not GENAI_AVAILABLE:
        return None

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    print("=" * 60)
    print("GENAI_AVAILABLE:", GENAI_AVAILABLE)
    print("Current working directory:", os.getcwd())
    print("API key exists:", bool(api_key))
    print("FULL API KEY:", repr(api_key) if api_key else "NONE")
    print("=" * 60)

    if not api_key or api_key == "your_gemini_api_key_here":
        return None

    try:
        genai.configure(api_key=api_key)

        # Conservative safety settings — analytical content shouldn't trigger any
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety = {
            HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        _model = genai.GenerativeModel(
            model_name="gemini-flash-latest",
            safety_settings=safety,
            generation_config=genai.GenerationConfig(
                temperature=0.55,       # Slightly creative but factually grounded
                max_output_tokens=600,
                top_p=0.92,
            ),
        )
        return _model
    except Exception as e:
        print(f"[insights.py] Gemini init error: {e}")
        return None


# ── Fallback responses (shown when API key missing or call fails) ──────────────

_FALLBACKS = {
    "market_pulse": (
        "⚠️ Gemini API key not configured — add GEMINI_API_KEY to your .env file.\n\n"
        "**What the AI would say based on your data:**\n\n"
        "Across 60,413 LinkedIn job postings tracked from April to June 2026, "
        "97.6% of April postings disappeared before the May snapshot — "
        "meaning the market turns over almost completely every 30 days. "
        "MNC postings survive 2.6x longer than startup postings (5.9% vs 2.3% survival rate), "
        "suggesting either slower hiring pipelines or repeated re-listing behaviour at large companies. "
        "May saw a 76.1% surge in new postings, followed by a 29.7% contraction in June, "
        "indicating a seasonal hiring wave in Q2 2026."
    ),
    "role_advice": (
        "⚠️ Gemini API key not configured — add GEMINI_API_KEY to your .env file.\n\n"
        "**Sample advice for Data Analyst roles:**\n\n"
        "Data Analyst postings have a 1.0% survival rate in this dataset — "
        "they disappear faster than any other tech role category. "
        "This means you should apply within 48–72 hours of a posting appearing. "
        "SQL appears in 60.1% of DA postings and has the highest survival advantage (+1.78pp), "
        "suggesting employers struggle most to find candidates with deep SQL skills. "
        "If you're targeting India specifically, Excel and Tableau are emphasised "
        "more than the global average, while Statistics and ML are less commonly required."
    ),
    "skill_roadmap": (
        "⚠️ Gemini API key not configured — add GEMINI_API_KEY to your .env file.\n\n"
        "**Sample skill roadmap for Data Analyst → Data Scientist:**\n\n"
        "The largest skill gaps between DA and DS roles are Machine Learning (+53.7pp), "
        "Python (+45.1pp), and Scikit-learn (+43.6pp). "
        "Start with Python's data science stack (Pandas, NumPy, Scikit-learn — "
        "they co-occur at r=0.93 so learning one means learning all three). "
        "Then add one ML framework (Scikit-learn for classical ML, then optionally PyTorch/TF). "
        "SQL remains important in DS roles (68% of postings) — don't deprioritise it."
    ),
}


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_market_pulse_prompt(payload: dict) -> str:
    """
    Constructs the market pulse prompt — the main dashboard insight.

    Designed for a non-technical hiring manager or job seeker.
    Injects all confirmed numbers so the model cannot hallucinate stats.
    """
    skills_long  = ", ".join(
        f"{s['skill']} (+{s['survival_advantage']:.2f}pp)"
        for s in payload.get("skills_in_long_lived_postings", [])
    )
    skills_fast  = ", ".join(
        f"{s['skill']} ({s['survival_advantage']:.2f}pp)"
        for s in payload.get("skills_in_fast_fill_postings", [])
    )

    role_rates = payload.get("all_role_survival_rates", {})
    role_lines = "\n".join(
        f"  {role}: {rate}% survive 30 days"
        for role, rate in sorted(role_rates.items(), key=lambda x: x[1])
        if role_rates  # only if data exists
    )

    return f"""You are a senior labour market analyst who has just finished analysing a dataset
of {payload['total_jobs_analysed']:,} LinkedIn job postings tracked monthly from {payload['months_covered']}.

Write a market pulse summary (3 tight paragraphs, 200–230 words total) for a non-technical
hiring manager or job seeker. Your writing must be clear, direct, and grounded in these confirmed figures.

KEY METRICS:
- April cohort: {payload['april_cohort_size']:,} unique job postings
- Disappeared before May: {payload['disappearance_rate_pct']}% (only {payload['survival_rate_pct']}% survived)
- Survived 2+ months: {payload['two_month_survival_pct']}% of April jobs
- New postings by month: April {payload['new_jobs_april']:,} → May {payload['new_jobs_may']:,} (+{payload['may_posting_surge_pct']}%) → June {payload['new_jobs_june']:,} ({payload['june_posting_change_pct']}%)
- MNC posting survival: {payload['mnc_survival_pct']}% vs Startup/SME {payload['startup_survival_pct']}% ({payload['mnc_multiplier']}x difference)

TECH ROLE SURVIVAL RATES (April cohort):
{role_lines}

SKILL SIGNALS:
- Skills in longer-lived postings (take time to fill): {skills_long}
- Skills in fast-disappearing postings: {skills_fast}

DATA LIMITATION TO MENTION:
Disappearance ≠ filled — a posting could have been filled, expired naturally,
or withdrawn. Use the word "disappeared" not "filled" throughout.

STRUCTURE:
Paragraph 1: What happened to the market — the headline disappearance finding with exact numbers.
Paragraph 2: Who survives longer and what the skill signals mean.
Paragraph 3: One concrete, specific action a job seeker should take based on these numbers.

Rules: No bullet points. No headers. No em-dashes. Flowing paragraphs only.
Mention the data limitation naturally in paragraph 1.
Be direct — give the recommendation, don't hedge."""


def _build_role_advice_prompt(payload: dict,
                               target_role: str,
                               target_location: str) -> str:
    """
    Personalised advice for a specific role + location combination.
    Designed for a final-year student applying for jobs.
    """
    target_surv = payload.get("target_role_survival_pct")
    target_n    = payload.get("target_role_sample_n")
    india_surv  = payload.get("india_tech_survival_pct")
    india_caveat = payload.get("india_sample_caveat", "")

    cities_str = ", ".join(
        f"{city} ({count} postings)"
        for city, count in list(payload.get("india_top_cities", {}).items())[:5]
    )

    skills_long  = ", ".join(
        f"{s['skill']} (+{s['survival_advantage']:.2f}pp)"
        for s in payload.get("skills_in_long_lived_postings", [])
    )
    skills_fast  = ", ".join(
        f"{s['skill']} ({s['survival_advantage']:.2f}pp)"
        for s in payload.get("skills_in_fast_fill_postings", [])
    )

    india_more = ""
    india_less = ""
    if target_location.lower() in ["india", "bengaluru", "hyderabad", "mumbai",
                                    "gurugram", "chennai", "pune"]:
        india_more = "Skills India emphasises more than global: Excel (+8.4pp), Tableau (+4.7pp), BigQuery (+3.4pp)"
        india_less = "Skills India emphasises less: Statistics (−10.3pp), ML (−10.3pp), Pandas (−8.1pp)"

    return f"""You are a senior data analytics career coach advising a final-year Computer
Engineering student from a Tier-3 college in India who is applying for {target_role} roles.

Write a personalised career insight (3 paragraphs, 200–230 words) based on this real data.

TARGET PROFILE:
- Role: {target_role}
- Location preference: {target_location}
- Stage: Final-year student, applying for internships and entry-level roles

MARKET DATA FOR THIS ROLE:
- {target_role} 30-day survival rate: {target_surv}% (based on {target_n} April postings)
- Overall market disappearance rate: {payload['disappearance_rate_pct']}%
- MNC vs Startup hiring speed: MNCs {payload['mnc_survival_pct']}% survival vs Startups {payload['startup_survival_pct']}%

SKILL SIGNALS:
- Skills in longer-lived postings (harder roles to fill): {skills_long}
- Skills in fast-fill postings: {skills_fast}
{india_more}
{india_less}

INDIA CONTEXT:
- India tech role 30-day survival: {india_surv}% (Note: {india_caveat})
- Top India hiring cities: {cities_str}
- India role distribution: {payload.get('india_role_distribution', {})}

STRUCTURE:
Paragraph 1: What the market data says specifically about {target_role} roles — speed, volume, competition.
Paragraph 2: Which skills to prioritise given the survival data and India-specific emphasis.
Paragraph 3: Specific application strategy — when to apply, which company type to target first,
             and one honest caveat about the data's limitations.

Rules: No bullet points. No headers. No generic advice. Every sentence must be grounded
in the numbers above. Speak directly to the student ("you"), not about them."""


def _build_skill_roadmap_prompt(payload: dict,
                                 current_role: str,
                                 target_role: str,
                                 skill_gap_df) -> str:
    """
    Skill learning roadmap based on the role matrix gap analysis.
    skill_gap_df comes from analytics.get_da_skill_gap().
    """
    # Top gaps (target needs more)
    top_gaps = skill_gap_df[skill_gap_df["gap"] > 10].head(6)
    gap_lines = "\n".join(
        f"  {row['skill']}: +{row['gap']:.0f}pp more common in {target_role} postings"
        for _, row in top_gaps.iterrows()
    )

    # Skills current role already has strong
    strong_skills = skill_gap_df[skill_gap_df["gap"] < -5].head(4)
    strong_lines = ", ".join(
        f"{row['skill']} (+{abs(row['gap']):.0f}pp in {current_role})"
        for _, row in strong_skills.iterrows()
    ) if len(strong_skills) else "None significant"

    return f"""You are a senior data science curriculum designer advising a {current_role}
who wants to transition to {target_role} roles.

Write a practical skill roadmap (3 paragraphs, 180–210 words) based on job posting data.

SKILL GAP ANALYSIS (from {payload['total_jobs_analysed']:,} LinkedIn postings):
Skills you need to add for {target_role} (% more common than in {current_role}):
{gap_lines}

Skills you already have an advantage in ({current_role} postings):
{strong_lines}

KEY CO-OCCURRENCE INSIGHT:
Pandas + NumPy co-occur in postings with correlation r=0.934
NumPy + Scikit-learn: r=0.804
→ Learning Pandas effectively means learning the full Python DS trio

STRUCTURE:
Paragraph 1: The biggest skill gaps and why they matter (ground in the posting frequency data).
Paragraph 2: A sequenced learning order — what to learn first, and why that sequence
             makes sense given how the skills co-occur in postings.
Paragraph 3: What NOT to deprioritise from your current {current_role} skillset,
             and one honest note about what job posting frequency can and cannot tell you
             about actual skill importance.

Rules: No bullet points. No numbered lists. Flowing paragraphs only.
Be specific about the numbers. Don't hedge — give a clear recommendation."""


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_market_pulse(payload: dict) -> dict:
    """
    Generates the main market pulse insight for the dashboard.

    Returns:
        {
          "insight":      str,   # The generated text
          "model":        str,   # "gemini-1.5-flash" or "fallback"
          "generated_at": str,   # ISO timestamp
          "prompt_tokens":int,   # Approximate (for transparency)
          "success":      bool,
        }
    """
    model = _get_model()
    if model is None:
        return _make_fallback_response("market_pulse")

    prompt = _build_market_pulse_prompt(payload)
    return _call_gemini(model, prompt, "market_pulse")


def generate_role_advice(payload: dict,
                          target_role: str = "Data Analyst",
                          target_location: str = "India") -> dict:
    """
    Generates personalised role + location advice.
    target_role and target_location come from the dashboard selectors.
    """
    model = _get_model()
    if model is None:
        return _make_fallback_response("role_advice")

    prompt = _build_role_advice_prompt(payload, target_role, target_location)
    return _call_gemini(model, prompt, "role_advice")


def generate_skill_roadmap(payload: dict,
                            skill_gap_df,
                            current_role: str = "Data Analyst",
                            target_role: str = "Data Scientist") -> dict:
    """
    Generates a skill learning roadmap for a role transition.
    skill_gap_df comes from analytics.get_da_skill_gap().
    """
    model = _get_model()
    if model is None:
        return _make_fallback_response("skill_roadmap")

    prompt = _build_skill_roadmap_prompt(payload, current_role, target_role, skill_gap_df)
    return _call_gemini(model, prompt, "skill_roadmap")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _call_gemini(model, prompt: str, insight_type: str,
                  max_retries: int = 2) -> dict:
    """
    Calls Gemini with retry logic and structured error handling.
    Returns a consistent dict regardless of success or failure.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt)

            # Extract text safely — response.text raises if blocked
            text = response.text.strip()
            if not text:
                raise ValueError("Gemini returned empty response")

            return {
                "insight":       text,
                "model":         "gemini-flash-latest",
                "insight_type":  insight_type,
                "generated_at":  datetime.utcnow().isoformat(),
                "prompt_chars":  len(prompt),
                "success":       True,
                "error":         None,
            }

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                wait = 2 ** attempt  # exponential back-off: 1s, 2s
                time.sleep(wait)

    # All retries exhausted — return fallback with error info
    result = _make_fallback_response(insight_type)
    result["error"] = last_error
    result["success"] = False
    return result


def _make_fallback_response(insight_type: str) -> dict:
    return {
        "insight":      _FALLBACKS.get(insight_type, "AI insight unavailable."),
        "model":        "fallback",
        "insight_type": insight_type,
        "generated_at": datetime.utcnow().isoformat(),
        "prompt_chars": 0,
        "success":      False,
        "error":        "Gemini API key not configured or unavailable",
    }


def gemini_is_configured() -> bool:
    """
    Returns True if Gemini is ready to use.
    Call this at dashboard startup to decide whether to show the AI buttons.
    """
    if not GENAI_AVAILABLE:
        return False
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(key) and key != "your_gemini_api_key_here"


# ── Test harness ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from analytics import load_all, build_insight_payload, get_da_skill_gap

    print("=== insights.py test harness ===\n")
    print(f"Gemini configured: {gemini_is_configured()}")
    print(f"google-generativeai installed: {GENAI_AVAILABLE}")
    print()

    # Load real data
    data    = load_all()
    payload = build_insight_payload(
        data["cohort"], data["skill_surv"], data["india_prof"],
        target_role="Data Analyst", target_location="India"
    )
    skill_gap = get_da_skill_gap(data["role_matrix"])

    print("--- Test 1: Market Pulse ---")
    result = generate_market_pulse(payload)
    print(f"Model:    {result['model']}")
    print(f"Success:  {result['success']}")
    if result.get("error"):
        print(f"Error:    {result['error']}")
    print(f"\nInsight preview (first 400 chars):\n{result['insight'][:400]}...")

    print("\n--- Test 2: Role Advice ---")
    result2 = generate_role_advice(payload, "Data Analyst", "India")
    print(f"Model:    {result2['model']}")
    print(f"Success:  {result2['success']}")
    print(f"\nInsight preview:\n{result2['insight'][:400]}...")

    print("\n--- Test 3: Skill Roadmap ---")
    result3 = generate_skill_roadmap(payload, skill_gap,
                                      current_role="Data Analyst",
                                      target_role="Data Scientist")
    print(f"Model:    {result3['model']}")
    print(f"Success:  {result3['success']}")
    print(f"\nInsight preview:\n{result3['insight'][:400]}...")

    print("\n=== All three insight types tested ===")
    print("If model='fallback', add GEMINI_API_KEY to your .env file.")
    print("Get your free key at: https://aistudio.google.com/app/apikey")
