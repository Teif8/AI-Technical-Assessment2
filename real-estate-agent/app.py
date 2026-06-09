"""
PropFind AI — Saudi Real Estate Recommendation Agent
=====================================================
Project 1 of 2 | AI Technical Assessment
Uses OpenAI API (gpt-4o-mini) + Python-based matching logic.
Property data is synthetic/demo — clearly labeled.
"""

import json
import os
import requests
import streamlit as st
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# ──────────────────────────────────────────────

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = "https://api.openai.com/v1/chat/completions"
MODEL   = "gpt-4o-mini"

# Scoring weights for property matching (must sum to 100)
SCORE_WEIGHTS = {
    "city":         30,
    "purpose":      20,
    "budget":       20,
    "bedrooms":     10,
    "property_type": 8,
    "parking":       6,
    "furnishing":    6,
}

st.set_page_config(
    page_title="PropFind AI",
    page_icon="🏙️",
    layout="wide"
)

# ──────────────────────────────────────────────
# SECTION 2: DATA LOADING
# ──────────────────────────────────────────────

@st.cache_data
def load_properties():
    """Load property dataset from JSON file. Cached for performance."""
    with open("properties.json", "r", encoding="utf-8") as f:
        return json.load(f)

PROPERTIES = load_properties()

def get_property_by_id(pid: str) -> dict | None:
    """Return a full property dict by its ID, or None if not found."""
    return next((p for p in PROPERTIES if p["id"] == pid), None)

# ──────────────────────────────────────────────
# SECTION 3: AI FUNCTIONS
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Saudi real estate consultant AI.
Your ONLY job is to extract search criteria from the user's message.

You must reply with ONLY valid JSON — no markdown, no backticks, no extra text.
Every field must use EXACTLY the allowed values listed below or null.

JSON structure:
{
  "criteria": {
    "city":          one of [Riyadh, Jeddah, Al Khobar, Dammam, Dhahran, Medina, Makkah, Abha, Tabuk] or null,
    "property_type": one of [Apartment, Villa, Studio, Duplex, Chalet, Loft, Townhouse] or null,
    "purpose":       one of [Rent, Sale] or null,
    "budget":        a number (max price the user will pay) or null,
    "bedrooms":      a number or null,
    "bathrooms":     a number or null,
    "parking":       true or false or null,
    "furnished":     one of [Furnished, Unfurnished, Semi-Furnished] or null
  },
  "message": "A friendly 1-2 sentence acknowledgement of the request",
  "followUpQuestion": "One focused question if city or purpose is still null after extraction, otherwise null"
}

STRICT RULES — never break these:
- purpose must be exactly "Rent" or "Sale". Never "rent", "buy", "for sale", "purchase" or any other form.
- property_type must be exactly one of the listed types. Never "house", "flat", "condo", etc.
- city must be exactly one of the listed cities. Match user input to the closest listed city.
- budget is always a number. "100000 SAR", "100k", "one hundred thousand" all become 100000.
- If the user says "buy" or "purchase" or "for sale", purpose = "Sale".
- If the user says "rent" or "lease" or "per month", purpose = "Rent".
- Do NOT recommend or invent properties — matching is handled separately.\nTreat every new user message as a completely new property search unless it is clearly answering a follow-up question.\nNever reuse city, budget, purpose, or property type from previous messages."""


def extract_criteria(user_message: str, history: list, api_key: str) -> dict:
    """
    Call the LLM to extract structured search criteria from the user's message.
    Returns a parsed dict with 'criteria', 'message', and 'followUpQuestion'.
    Falls back to a safe default dict on any error.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Treat every message as a fresh search unless it is a follow-up.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": MODEL, "messages": messages, "temperature": 0.2},
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]

        # Clean any accidental markdown wrappers
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)

    except json.JSONDecodeError:
        return _fallback_response("I had trouble understanding the response format. Please try again.")
    except requests.exceptions.Timeout:
        return _fallback_response("The request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        return _fallback_response(f"API error: {e.response.status_code}. Please check your API key.")
    except Exception as e:
        return _fallback_response(f"Something went wrong: {str(e)}")


def _fallback_response(message: str) -> dict:
    """Return a safe default response structure when the AI call fails."""
    return {
        "criteria": {
            "city": None, "property_type": None, "purpose": None,
            "budget": None, "bedrooms": None, "bathrooms": None,
            "parking": None, "furnished": None,
        },
        "message": message,
        "followUpQuestion": None,
    }


def normalize_criteria(criteria: dict) -> dict:
    """
    Sanitise and normalise raw LLM output before it reaches hard_filter().
    Ensures all values use exact casing expected by the dataset so that
    string comparisons in hard_filter() always work correctly.
    """
    # Allowed value maps — keys are lowercase variants, values are canonical forms
    PURPOSE_MAP = {
        "rent": "Rent", "rental": "Rent", "lease": "Rent", "monthly": "Rent",
        "sale": "Sale", "sell": "Sale", "buy": "Sale", "purchase": "Sale",
        "buying": "Sale", "for sale": "Sale", "forsale": "Sale",
    }
    TYPE_MAP = {
        "apartment": "Apartment", "flat": "Apartment", "condo": "Apartment",
        "villa": "Villa", "house": "Villa", "home": "Villa",
        "studio": "Studio",
        "duplex": "Duplex",
        "chalet": "Chalet", "cabin": "Chalet",
        "loft": "Loft",
        "townhouse": "Townhouse", "town house": "Townhouse",
    }
    CITY_MAP = {
        "riyadh": "Riyadh",
        "jeddah": "Jeddah", "jidda": "Jeddah", "jiddah": "Jeddah",
        "al khobar": "Al Khobar", "khobar": "Al Khobar", "alkhobar": "Al Khobar",
        "dammam": "Dammam",
        "dhahran": "Dhahran", "al dhahran": "Dhahran",
        "medina": "Medina", "madinah": "Medina", "al madinah": "Medina",
        "makkah": "Makkah", "mecca": "Makkah", "mekka": "Makkah",
        "abha": "Abha",
        "tabuk": "Tabuk",
    }
    FURNISH_MAP = {
        "furnished": "Furnished", "fully furnished": "Furnished", "full": "Furnished",
        "unfurnished": "Unfurnished", "un-furnished": "Unfurnished", "empty": "Unfurnished",
        "semi-furnished": "Semi-Furnished", "semi furnished": "Semi-Furnished", "semi": "Semi-Furnished",
    }

    normalised = dict(criteria)

    # Normalise purpose
    raw_purpose = (normalised.get("purpose") or "").strip().lower()
    if raw_purpose:
        normalised["purpose"] = PURPOSE_MAP.get(raw_purpose, None)
        if normalised["purpose"] is None:
            # Try partial match — e.g. "for sale" contains "sale"
            for key, val in PURPOSE_MAP.items():
                if key in raw_purpose:
                    normalised["purpose"] = val
                    break

    # Normalise property_type
    raw_type = (normalised.get("property_type") or "").strip().lower()
    if raw_type:
        normalised["property_type"] = TYPE_MAP.get(raw_type, None)
        if normalised["property_type"] is None:
            for key, val in TYPE_MAP.items():
                if key in raw_type:
                    normalised["property_type"] = val
                    break

    # Normalise city
    raw_city = (normalised.get("city") or "").strip().lower()
    if raw_city:
        normalised["city"] = CITY_MAP.get(raw_city, None)
        if normalised["city"] is None:
            for key, val in CITY_MAP.items():
                if key in raw_city or raw_city in key:
                    normalised["city"] = val
                    break

    # Normalise furnishing
    raw_furnish = (normalised.get("furnished") or "").strip().lower()
    if raw_furnish:
        normalised["furnished"] = FURNISH_MAP.get(raw_furnish, None)
        if normalised["furnished"] is None:
            for key, val in FURNISH_MAP.items():
                if key in raw_furnish:
                    normalised["furnished"] = val
                    break

    # Ensure budget is a plain number (strip any accidental string)
    raw_budget = normalised.get("budget")
    if raw_budget is not None:
        try:
            normalised["budget"] = float(str(raw_budget).replace(",", "").replace(" ", ""))
        except (ValueError, TypeError):
            normalised["budget"] = None

    return normalised


def generate_match_explanation(property_data: dict, criteria: dict, api_key: str) -> list[str]:
    """
    Ask the LLM to generate 2-3 concise match reasons for a specific property.
    Falls back to Python-generated reasons if the call fails.
    """
    prompt = f"""Given this property:
{json.dumps(property_data, ensure_ascii=False)}

And these user criteria:
{json.dumps(criteria, ensure_ascii=False)}

Reply ONLY with a JSON array of 2-3 short match reason strings. Example:
["Matches budget of 7,000 SAR", "Furnished as requested", "Has parking"]
No extra text, no markdown."""

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        reasons = json.loads(cleaned)
        return reasons if isinstance(reasons, list) else _python_reasons(property_data, criteria)
    except Exception:
        return _python_reasons(property_data, criteria)


def _python_reasons(prop: dict, criteria: dict) -> list[str]:
    """Fallback: generate match reasons using Python logic."""
    reasons = []
    if criteria.get("city") and prop["city"].lower() == criteria["city"].lower():
        reasons.append(f"Located in {prop['city']}")
    if criteria.get("budget") and prop["price"] <= criteria["budget"]:
        reasons.append(f"Within budget ({prop['price']:,} SAR)")
    if criteria.get("parking") and prop["parking"]:
        reasons.append("Has parking")
    if criteria.get("furnished") and criteria["furnished"].lower() in prop["furnishing"].lower():
        reasons.append(f"{prop['furnishing']}")
    if criteria.get("bedrooms") and prop["bedrooms"] == criteria["bedrooms"]:
        reasons.append(f"{prop['bedrooms']} bedrooms as requested")
    if not reasons:
        reasons.append("Closest match to your requirements")
    return reasons[:3]

# ──────────────────────────────────────────────
# SECTION 4: MATCHING LOGIC
# ──────────────────────────────────────────────

def find_top_matches(criteria: dict, top_n: int = 5) -> list[dict]:
    """
    Strict two-phase matching pipeline.

    Phase 1 — Hard filters (city / purpose / property_type).
      Each specified criterion eliminates non-matching properties outright.
      A property is only allowed to proceed if it passes EVERY active filter.
      These three dimensions are NEVER used in scoring — they are gates.

    Phase 2 — Soft scoring (budget / bedrooms / parking / furnishing).
      Only properties that cleared all hard filters are scored.
      Score range 0-100; higher is a better soft match.

    Returns up to top_n results sorted by score desc, then price asc.
    Returns [] when no property clears all hard filters.
    """

    # ── Phase 1: Hard filters ──────────────────────────────────────────
    # Extract once; use .lower() for all comparisons so casing never matters.
    want_city    = (criteria.get("city")          or "").strip().lower()
    want_purpose = (criteria.get("purpose")       or "").strip().lower()
    want_type    = (criteria.get("property_type") or "").strip().lower()

    candidates = []
    for prop in PROPERTIES:
        # Gate 1 — city must match exactly when specified
        if want_city and prop["city"].lower() != want_city:
            continue

        # Gate 2 — purpose (Rent/Sale) must match exactly when specified
        if want_purpose and prop["purpose"].lower() != want_purpose:
            continue

        # Gate 3 — property type must match exactly when specified
        if want_type and prop["type"].lower() != want_type:
            continue
        if criteria.get("budget"):
         if prop["price"] > criteria["budget"]:
          continue

        candidates.append(prop)

    # Nothing survived the hard filters → caller shows no-match message
    if not candidates:
        return []

    # ── Phase 2: Soft scoring on filtered candidates only ─────────────
    def score(prop: dict) -> int:
        pts = 0

        # Budget (40 pts) — largest weight because it is the most decisive soft factor
        budget = criteria.get("budget")
        if budget:
            try:
                budget = float(budget)
            except (TypeError, ValueError):
                budget = None
        if budget:
            if prop["price"] <= budget:
                pts += 40                          # within budget  → full
            elif prop["price"] <= budget * 1.10:
                pts += 20                          # up to 10% over → half
            # more than 10% over → 0
        else:
            pts += 40                              # no budget preference → full

        # Bedrooms (30 pts)
        want_bd = criteria.get("bedrooms")
        if want_bd is not None:
            try:
                want_bd = int(want_bd)
            except (TypeError, ValueError):
                want_bd = None
        if want_bd is not None:
            diff = abs(prop["bedrooms"] - want_bd)
            if diff == 0:
                pts += 30
            elif diff == 1:
                pts += 15
        else:
            pts += 30

        # Parking (15 pts)
        want_parking = criteria.get("parking")
        if want_parking is not None:
            if prop["parking"] == want_parking:
                pts += 15
        else:
            pts += 15

        # Furnishing (15 pts)
        want_furnish = (criteria.get("furnished") or "").strip().lower()
        if want_furnish:
            prop_furnish = prop["furnishing"].lower()
            if prop_furnish == want_furnish:
                pts += 15
            elif "semi" in prop_furnish or "semi" in want_furnish:
                pts += 7
        else:
            pts += 15

        return min(pts, 100)

    scored = [{**prop, "_score": score(prop)} for prop in candidates]
    scored.sort(key=lambda x: (-x["_score"], x["price"]))
    return scored[:top_n]

# ──────────────────────────────────────────────
# SECTION 5: UI COMPONENTS
# ──────────────────────────────────────────────

def render_styles():
    """Inject custom CSS for the dark-themed property cards and chat UI."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.prop-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    border-top: 3px solid #00d4aa;
}
.prop-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    color: #e2e8f0;
    margin: 6px 0 2px;
}
.prop-price {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f59e0b;
    margin: 4px 0 8px;
}
.prop-location { color: #64748b; font-size: 0.82rem; margin-bottom: 6px; }
.prop-desc     { color: #94a3b8; font-size: 0.81rem; line-height: 1.5; margin-top: 8px; }
.match-badge {
    display: inline-block;
    background: rgba(0,212,170,0.12);
    color: #00d4aa;
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.74rem;
    font-weight: 600;
}
.purpose-rent { background:#10b981; color:#fff; padding:2px 8px; border-radius:20px; font-size:0.72rem; font-weight:600; }
.purpose-sale { background:#f59e0b; color:#fff; padding:2px 8px; border-radius:20px; font-size:0.72rem; font-weight:600; }
.spec-pill {
    display: inline-block;
    background: #1a2236;
    color: #64748b;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.74rem;
    margin: 2px;
}
.reason-item  { color: #00d4aa; font-size: 0.8rem; margin: 1px 0; }
.follow-up {
    background: rgba(0,212,170,0.07);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 10px;
    padding: 10px 14px;
    color: #00d4aa;
    font-size: 0.88rem;
    margin-top: 10px;
}
.no-match {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    color: #f59e0b;
    margin-top: 10px;
}
.disclaimer {
    background: #1a2236;
    border-left: 3px solid #f59e0b;
    border-radius: 6px;
    padding: 8px 14px;
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


def render_property_card(prop: dict, score: int, reasons: list[str]):
    """Render a single property card as a fully self-contained HTML block."""
    purpose_class = "purpose-rent" if prop["purpose"] == "Rent" else "purpose-sale"
    period_str    = f"/{prop['period']}" if prop.get("period") else ""

    # Build spec pills as raw HTML strings
    bed_pill      = (
        f"<span class='spec-pill'>🛏️ {prop['bedrooms']} bd</span>"
        if prop["bedrooms"] > 0
        else "<span class='spec-pill'>Studio</span>"
    )
    bath_pill     = f"<span class='spec-pill'>🚿 {prop['bathrooms']} ba</span>"
    area_pill     = f"<span class='spec-pill'>📐 {prop['area']} m²</span>"
    parking_pill  = "<span class='spec-pill'>🚗 Parking</span>" if prop["parking"] else ""
    furnish_pill  = f"<span class='spec-pill'>{prop['furnishing']}</span>"

    reasons_html  = "".join(
        f"<div class='reason-item'>✓ {r}</div>" for r in reasons
    )

    card_html = f"""
<div class="prop-card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span class="match-badge">{score}% Match</span>
    <span class="{purpose_class}">{prop["purpose"]}</span>
  </div>
  <div class="prop-title">{prop["title"]}</div>
  <div class="prop-location">📍 {prop["district"]}, {prop["city"]}</div>
  <div class="prop-price">{prop["price"]:,} SAR{period_str}</div>
  <div style="margin:6px 0">
    {bed_pill}{bath_pill}{area_pill}{parking_pill}{furnish_pill}
  </div>
  <div class="prop-desc">{prop["description"]}</div>
  <div style="margin-top:10px">{reasons_html}</div>
  <div style="color:#1e2d45;font-size:0.68rem;margin-top:8px">ID: {prop["id"]}</div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)


def render_criteria_expander(criteria: dict):
    """Show the AI-extracted search criteria in a collapsible expander."""
    with st.expander("🔍 Extracted Search Criteria", expanded=False):
        cols = st.columns(4)
        fields = [
            ("🏙️ City",          criteria.get("city")),
            ("🏠 Type",          criteria.get("property_type")),
            ("🔑 Purpose",       criteria.get("purpose")),
            ("💰 Max Budget",    f"{criteria['budget']:,} SAR" if criteria.get("budget") else None),
            ("🛏️ Bedrooms",     criteria.get("bedrooms")),
            ("🚿 Bathrooms",    criteria.get("bathrooms")),
            ("🚗 Parking",       "Yes" if criteria.get("parking") is True else ("No" if criteria.get("parking") is False else None)),
            ("🛋️ Furnishing",   criteria.get("furnished")),
        ]
        for i, (label, value) in enumerate(fields):
            with cols[i % 4]:
                st.metric(label, value if value is not None else "Any")


def render_no_match(criteria: dict):
    """Render a helpful 'no results' message with suggestions."""
    city    = criteria.get("city", "your chosen area")
    budget  = criteria.get("budget")
    budget_str = f"{budget:,} SAR" if budget else "your budget"

    st.markdown(f"""
<div class="no-match">
  <strong>😔 No exact matches found</strong><br><br>
  We couldn't find properties matching all your criteria in <strong>{city}</strong>
  within <strong>{budget_str}</strong>.<br><br>
  <strong>Try:</strong><br>
  • Increasing your budget by 10–20%<br>
  • Searching in a nearby city<br>
  • Removing furnishing or parking requirements<br>
  • Changing the property type
</div>
""", unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with dataset coverage and example queries."""
    with st.sidebar:
        st.markdown("## 🏙️ PropFind AI")
        st.markdown("*Saudi Real Estate Agent*")
        st.divider()

        st.markdown("**Dataset Coverage**")
        cities = sorted({p["city"] for p in PROPERTIES})
        for city in cities:
            count = sum(1 for p in PROPERTIES if p["city"] == city)
            st.markdown(f"📍 {city} — {count} listings")

        st.divider()
        st.markdown("**Try asking:**")
        examples = [
            "Furnished 2BR in Riyadh under 7,000 SAR/month with parking",
            "Villa for sale in Jeddah, 4+ bedrooms",
            "Affordable studio in Al Khobar",
            "3BR apartment in Dammam for rent",
            "Chalet or unique property anywhere in Saudi Arabia",
        ]
        for ex in examples:
            st.markdown(f"• *{ex}*")

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("""
<div class="disclaimer">
  ⚠️ This application uses synthetic/demo property data for demonstration purposes only.
  All listings are fictional and do not represent real properties.
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SECTION 6: MAIN APP
# ──────────────────────────────────────────────

def main():
    render_styles()
    render_sidebar()

    # Guard: require API key from .env
    if not API_KEY:
        st.error(
            "⚠️ OpenAI API key not found.\n\n"
            "Create a `.env` file in the project root with:\n"
            "```\nOPENAI_API_KEY=your_key_here\n```"
        )
        st.stop()

    st.markdown("### 🏡 Find Your Perfect Property in Saudi Arabia")
    st.markdown(
        "Describe what you're looking for in plain English — "
        "city, budget, bedrooms, furnishing, and more."
    )

    # Disclaimer banner
    st.info(
        "⚠️ **Demo Notice:** This application uses synthetic/demo property data "
        "for demonstration purposes only. All listings are fictional.",
        icon="ℹ️"
    )

    # Initialize chat history and pending criteria accumulator
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_criteria" not in st.session_state:
        # Stores partial criteria while the AI is asking follow-up questions.
        # Keys with None mean "not yet known"; filled in as the user answers.
        st.session_state.pending_criteria = {
            "city": None, "property_type": None, "purpose": None,
            "budget": None, "bedrooms": None, "bathrooms": None,
            "parking": None, "furnished": None,
        }

    # Render existing chat turns
    for turn in st.session_state.messages:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.write(turn["content"])

            elif turn["role"] == "assistant":
                data           = turn["data"]
                criteria       = data.get("criteria", {})
                matches        = data.get("matches", [])
                follow_up      = data.get("followUpQuestion")
                waiting        = bool(follow_up)

                st.write(data.get("message", ""))

                # Show criteria expander only when we have results, not while waiting
                if not waiting and any(v is not None for v in criteria.values()):
                    render_criteria_expander(criteria)

                if waiting:
                    # Show follow-up question — do NOT show cards
                    st.markdown(
                        f'<div class="follow-up">💬 {follow_up}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # No pending follow-up — show property cards or no-match
                    if matches:
                        st.markdown(f"**Top {len(matches)} matching properties:**")
                        cols = st.columns(min(len(matches), 3))
                        for i, match in enumerate(matches):
                            with cols[i % 3]:
                                render_property_card(
                                    match["property"],
                                    match["score"],
                                    match["reasons"],
                                )
                    elif any(v is not None for v in criteria.values()):
                        render_no_match(criteria)

    # Chat input
    user_input = st.chat_input("Describe your ideal property...")

    EMPTY_CRITERIA = {
        "city": None, "property_type": None, "purpose": None,
        "budget": None, "bedrooms": None, "bathrooms": None,
        "parking": None, "furnished": None,
    }

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):

                # Step 1: LLM extracts criteria from the latest user message
                ai_result    = extract_criteria(
                    user_input,
                    st.session_state.messages[:-1],
                    API_KEY,
                )
                new_criteria = normalize_criteria(ai_result.get("criteria", {}))
                follow_up    = ai_result.get("followUpQuestion")

                # New search: clear old state unless we're in a follow-up flow
                if not st.session_state.get("awaiting_followup", False):
                    st.session_state.pending_criteria = dict(EMPTY_CRITERIA)

                # Step 2: Merge non-None values into pending_criteria so that
                # answers to follow-up questions accumulate correctly.
                for key, value in new_criteria.items():
                    if value is not None:
                        st.session_state.pending_criteria[key] = value

                # Step 3: Build the final criteria dict from the merged state
                criteria = dict(st.session_state.pending_criteria)
                waiting  = bool(follow_up)
                st.session_state.awaiting_followup = waiting

                # Step 4: Run matching only when the AI has no pending question.
                # find_top_matches() applies hard filters internally — city,
                # purpose, and property_type are enforced as strict gates before
                # any scoring takes place.
                matches = []
                if not waiting:
                    top_props = find_top_matches(criteria, top_n=5)
                    for prop in top_props:
                        score   = prop.pop("_score")
                        reasons = generate_match_explanation(prop, criteria, API_KEY)
                        matches.append({"property": prop, "score": score, "reasons": reasons})

                    # Reset pending criteria after a completed search so the
                    # next query always starts from a clean slate
                    st.session_state.pending_criteria = dict(EMPTY_CRITERIA)



            # Render AI message
            st.write(ai_result.get("message", ""))

            if waiting:
                # Still collecting info — show follow-up question only, no cards
                st.markdown(
                    f'<div class="follow-up">💬 {follow_up}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # All criteria gathered — show expander + cards
                if any(v is not None for v in criteria.values()):
                    render_criteria_expander(criteria)

                if matches:
                    st.markdown(f"**Top {len(matches)} matching properties:**")
                    cols = st.columns(min(len(matches), 3))
                    for i, match in enumerate(matches):
                        with cols[i % 3]:
                            render_property_card(
                                match["property"],
                                match["score"],
                                match["reasons"],
                            )
                elif any(v is not None for v in criteria.values()):
                    render_no_match(criteria)

        # Save assistant turn to history (store merged criteria, not just latest)
        st.session_state.messages.append({
            "role": "assistant",
            "data": {
                "message":          ai_result.get("message", ""),
                "criteria":         criteria,
                "followUpQuestion": follow_up,
                "matches":          matches,
            }
        })


if __name__ == "__main__":
    main()
