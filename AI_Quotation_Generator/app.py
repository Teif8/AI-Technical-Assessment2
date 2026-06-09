"""
AI Quotation Generator Agent
==============================
A professional quotation generator powered by GPT-4o-mini.
Converts natural language project requests into structured, itemised quotations.

Architecture:
    1. Configuration       – constants, env loading, page setup
    2. Pricing Catalog     – service definitions and pricing
    3. AI Functions        – GPT-4o-mini extraction + JSON validation
    4. Pricing Engine      – line-item calculation and add-on logic
    5. Quotation Builder   – assembles the full quotation document
    6. UI Components       – Streamlit rendering helpers
    7. Main App            – page routing and state management
"""

import os
import re
import json
import uuid
import datetime
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

PAGE_TITLE   = "QuoteForge AI"
PAGE_ICON    = "⚡"
APP_VERSION  = "1.0.0"
COMPANY_NAME = "QuoteForge Digital Agency"
COMPANY_EMAIL = "hello@quoteforge.io"
VAT_RATE     = 0.15          # 15 %
MODEL_NAME   = "gpt-4o-mini"
MAX_TOKENS   = 900

# ─────────────────────────────────────────────────────────────────────────────
# 2. PRICING CATALOG
# ─────────────────────────────────────────────────────────────────────────────

PRICING_CATALOG: dict[str, dict] = {
    # key: {label, unit_price, unit, description, category}
    # All prices in Saudi Riyal (SAR)
    "website_page": {
        "label":       "Website Page",
        "unit_price":  500,
        "unit":        "page",
        "description": "Custom-designed, responsive web page (design + development)",
        "category":    "Web Development",
    },
    "contact_form": {
        "label":       "Contact Form",
        "unit_price":  250,
        "unit":        "form",
        "description": "Secure contact / inquiry form with email notification",
        "category":    "Web Development",
    },
    "basic_seo": {
        "label":       "Basic SEO Package",
        "unit_price":  750,
        "unit":        "project",
        "description": "On-page SEO: meta tags, sitemap, robots.txt, keyword optimisation",
        "category":    "Marketing",
    },
    "advanced_seo": {
        "label":       "Advanced SEO Package",
        "unit_price":  1500,
        "unit":        "project",
        "description": "Full SEO audit, backlink strategy, monthly reporting (3 months)",
        "category":    "Marketing",
    },
    "hosting": {
        "label":       "Web Hosting",
        "unit_price":  500,
        "unit":        "year",
        "description": "Managed cloud hosting with SSL certificate, daily backups, CDN",
        "category":    "Infrastructure",
    },
    "domain_setup": {
        "label":       "Domain Registration & Setup",
        "unit_price":  100,
        "unit":        "year",
        "description": "Domain name registration, DNS configuration, email forwarding",
        "category":    "Infrastructure",
    },
    "logo_design": {
        "label":       "Logo Design",
        "unit_price":  800,
        "unit":        "project",
        "description": "Professional logo with 3 concepts, 2 revision rounds, all source files",
        "category":    "Design",
    },
    "ecommerce": {
        "label":       "E-commerce Functionality",
        "unit_price":  3000,
        "unit":        "project",
        "description": "Product catalogue, shopping cart, checkout flow, order management",
        "category":    "Web Development",
    },
    "payment_gateway": {
        "label":       "Payment Gateway Integration",
        "unit_price":  1200,
        "unit":        "gateway",
        "description": "Mada / STC Pay / HyperPay integration with webhooks and receipts",
        "category":    "Web Development",
    },
    "monthly_maintenance": {
        "label":       "Monthly Maintenance",
        "unit_price":  400,
        "unit":        "month",
        "description": "CMS updates, security patches, uptime monitoring, monthly report",
        "category":    "Support",
    },
    "ai_chatbot": {
        "label":       "AI Chatbot Integration",
        "unit_price":  5000,
        "unit":        "project",
        "description": "Custom-trained AI chatbot (GPT-powered), FAQ automation, live handoff",
        "category":    "AI & Automation",
    },
    "content_writing": {
        "label":       "Content Writing",
        "unit_price":  150,
        "unit":        "page",
        "description": "SEO-optimised copywriting per page (up to 600 words, Arabic or English)",
        "category":    "Content",
    },
    "uiux_design": {
        "label":       "UI/UX Design",
        "unit_price":  2000,
        "unit":        "project",
        "description": "Wireframes, interactive Figma prototype, design system, handoff docs",
        "category":    "Design",
    },
    "social_media_setup": {
        "label":       "Social Media Profile Setup",
        "unit_price":  350,
        "unit":        "project",
        "description": "Branded profiles on up to 4 platforms with cover art and bio copy",
        "category":    "Marketing",
    },
    "google_analytics": {
        "label":       "Analytics & Tracking Setup",
        "unit_price":  300,
        "unit":        "project",
        "description": "Google Analytics 4, Search Console, conversion events, dashboard",
        "category":    "Marketing",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. AI FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a project requirements extraction engine for a digital agency.

Your ONLY job is to parse a natural-language project request and return a
single, valid JSON object — nothing else, no markdown, no commentary.

JSON schema (all fields required):
{
  "client_name":          string | null,
  "project_type":         string,
  "timeline_weeks":       integer,
  "special_requirements": string | null,
  "services": [
    {
      "service_key":  string,   // must be one of the known keys listed below
      "quantity":     number    // positive number, decimals allowed for fractional units
    }
  ]
}

Known service_key values (use ONLY these exact strings):
  website_page, contact_form, basic_seo, advanced_seo, hosting, domain_setup,
  logo_design, ecommerce, payment_gateway, monthly_maintenance, ai_chatbot,
  content_writing, uiux_design, social_media_setup, google_analytics

Extraction rules:
- If the request mentions "X pages", set service_key="website_page", quantity=X.
- If "basic seo" or "seo" (no qualifier), use basic_seo.
- If "advanced seo", use advanced_seo.
- If "hosting for N year(s)", set service_key="hosting", quantity=N.
- If "domain" or "domain setup", include domain_setup with quantity=1.
- If "maintenance for N months", set service_key="monthly_maintenance", quantity=N.
- If "e-commerce" or "online shop", include ecommerce with quantity=1.
- If "payment gateway" or "payment integration", include payment_gateway with quantity=1.
- If "chatbot" or "ai chatbot", include ai_chatbot with quantity=1.
- If "content" or "content writing", include content_writing — infer quantity from pages
  mentioned, or default to 1 if unclear.
- If "logo", include logo_design with quantity=1.
- If "ui/ux" or "design mockup", include uiux_design with quantity=1.
- If "analytics" or "tracking", include google_analytics with quantity=1.
- If "social media setup", include social_media_setup with quantity=1.
- If "contact form", include contact_form with quantity=1.
- If timeline is not stated, default to 4 weeks.
- Do NOT invent services not mentioned.
- client_name: extract if mentioned, otherwise null.
- project_type: a short label like "Corporate Website", "E-commerce Store", "Portfolio Site".
- Return ONLY the JSON object. No preamble, no trailing text.
"""


def get_openai_client() -> OpenAI | None:
    """Return an authenticated OpenAI client, or None if the key is missing."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def extract_requirements(user_request: str) -> dict | None:
    """
    Send the user request to GPT-4o-mini and extract structured requirements.

    Returns a validated dict or None on failure.
    """
    client = get_openai_client()
    if client is None:
        st.error("⚠️ OPENAI_API_KEY not found. Please add it to your .env file.")
        return None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=0.1,        # Low temperature for deterministic JSON output
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_request},
            ],
        )
        raw_text = response.choices[0].message.content or ""
        return validate_and_normalise(raw_text)

    except Exception as exc:
        st.error(f"OpenAI API error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Validation & normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_KEYS = set(PRICING_CATALOG.keys())

def validate_and_normalise(raw: str) -> dict | None:
    """
    Parse the LLM response into a clean, validated dict.

    Handles:
    - Markdown code-fence stripping
    - JSON extraction via regex fallback
    - Field-level type coercion
    - Unknown service_key filtering
    - Quantity clamping (min 0.5, max 60)
    """
    # Strip markdown fences if present
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse
    data = _try_json(text)
    if data is None:
        # Regex fallback: find the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = _try_json(match.group())
    if data is None:
        st.error("Failed to parse AI response as JSON. Please rephrase your request.")
        return None

    # ── Required fields with sane defaults ──────────────────────────────────
    data.setdefault("client_name", None)
    data.setdefault("project_type", "Custom Project")
    data.setdefault("timeline_weeks", 4)
    data.setdefault("special_requirements", None)
    data.setdefault("services", [])

    # ── Type coercion ────────────────────────────────────────────────────────
    if not isinstance(data["timeline_weeks"], (int, float)):
        data["timeline_weeks"] = _safe_int(data["timeline_weeks"], 4)
    data["timeline_weeks"] = max(1, int(data["timeline_weeks"]))

    if not isinstance(data["services"], list):
        data["services"] = []

    # ── Normalise each service entry ─────────────────────────────────────────
    clean_services: list[dict] = []
    seen_keys: set[str] = set()

    for svc in data["services"]:
        if not isinstance(svc, dict):
            continue
        key = str(svc.get("service_key", "")).strip().lower()
        if key not in KNOWN_KEYS:
            continue                           # Silently drop unknown keys
        if key in seen_keys:
            continue                           # Deduplicate
        qty = _safe_float(svc.get("quantity", 1), 1.0)
        qty = max(0.5, min(qty, 60.0))         # Guard against wild values
        clean_services.append({"service_key": key, "quantity": round(qty, 1)})
        seen_keys.add(key)

    data["services"] = clean_services
    return data


def _try_json(text: str) -> dict | None:
    """Attempt JSON parse; return None on failure."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRICING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_line_items(services: list[dict]) -> list[dict]:
    """
    Convert extracted service entries into priced line items.

    Returns a list of dicts with keys:
        label, description, unit, quantity, unit_price, line_total
    """
    items: list[dict] = []
    for svc in services:
        key      = svc["service_key"]
        qty      = svc["quantity"]
        catalog  = PRICING_CATALOG[key]
        total    = round(catalog["unit_price"] * qty, 2)
        items.append({
            "key":         key,
            "category":    catalog["category"],
            "label":       catalog["label"],
            "description": catalog["description"],
            "unit":        catalog["unit"],
            "quantity":    qty,
            "unit_price":  catalog["unit_price"],
            "line_total":  total,
        })
    return items


def calculate_totals(line_items: list[dict]) -> dict:
    """Compute subtotal, VAT, and grand total."""
    subtotal  = sum(item["line_total"] for item in line_items)
    vat       = round(subtotal * VAT_RATE, 2)
    grand     = round(subtotal + vat, 2)
    return {"subtotal": subtotal, "vat": vat, "grand_total": grand}


ADD_ON_RULES: dict[str, list[tuple[str, str]]] = {
    # trigger_key: [(addon_key, reason), ...]
    "website_page": [
        ("basic_seo",        "Boost organic visibility for your new website"),
        ("google_analytics", "Track visitor behaviour and measure performance"),
        ("content_writing",  "Professional copy makes your pages convert better"),
    ],
    "ecommerce": [
        ("ai_chatbot",       "Automate product queries and reduce support tickets"),
        ("payment_gateway",  "Accept card payments seamlessly in your store"),
        ("monthly_maintenance", "Keep your store secure and running smoothly"),
    ],
    "logo_design": [
        ("uiux_design",      "Extend your brand identity into a cohesive UI system"),
        ("social_media_setup","Launch branded profiles alongside your new logo"),
    ],
    "basic_seo": [
        ("google_analytics", "Pair SEO with analytics to measure ranking improvements"),
        ("content_writing",  "Fresh content accelerates SEO gains significantly"),
    ],
    "ai_chatbot": [
        ("monthly_maintenance","Keep your chatbot trained and performing over time"),
    ],
    "hosting": [
        ("domain_setup",     "Bundle your domain registration for a complete setup"),
    ],
}

# Startup / small-business signals → recommend branding bundle
STARTUP_SIGNALS = {"website_page", "hosting", "domain_setup"}


def recommend_addons(services: list[dict]) -> list[dict]:
    """
    Return a deduplicated list of add-on recommendations with reasons.

    Rules:
    - Map each selected service against ADD_ON_RULES.
    - If selected services match STARTUP_SIGNALS, push logo + content.
    - Never recommend a service the user already has.
    """
    selected_keys = {s["service_key"] for s in services}
    recommendations: dict[str, str] = {}   # key → reason

    for svc in services:
        for addon_key, reason in ADD_ON_RULES.get(svc["service_key"], []):
            if addon_key not in selected_keys and addon_key not in recommendations:
                recommendations[addon_key] = reason

    # Startup bundle heuristic
    if STARTUP_SIGNALS.issubset(selected_keys) and "ecommerce" not in selected_keys:
        for key, reason in [
            ("logo_design",    "A professional logo makes your new business memorable"),
            ("content_writing","Polished copy builds trust from day one"),
        ]:
            if key not in selected_keys and key not in recommendations:
                recommendations[key] = reason

    addons: list[dict] = []
    for key, reason in recommendations.items():
        catalog = PRICING_CATALOG[key]
        addons.append({
            "key":        key,
            "label":      catalog["label"],
            "unit_price": catalog["unit_price"],
            "unit":       catalog["unit"],
            "reason":     reason,
        })
    return addons[:5]   # Cap at 5 recommendations


# ─────────────────────────────────────────────────────────────────────────────
# 5. QUOTATION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_quotation(requirements: dict) -> dict:
    """
    Assemble the complete quotation document from extracted requirements.

    Returns a nested dict that the UI renders.
    """
    line_items = calculate_line_items(requirements["services"])
    totals     = calculate_totals(line_items)
    addons     = recommend_addons(requirements["services"])

    # Group line items by category for display
    grouped: dict[str, list] = {}
    for item in line_items:
        grouped.setdefault(item["category"], []).append(item)

    # Derive timeline label
    weeks = requirements.get("timeline_weeks", 4)
    if weeks <= 2:
        timeline_label = f"{weeks} week{'s' if weeks > 1 else ''} (Rush delivery)"
    elif weeks <= 6:
        timeline_label = f"{weeks} weeks"
    else:
        months = round(weeks / 4.33, 1)
        timeline_label = f"{weeks} weeks (~{months} months)"

    quote_number = f"QF-{datetime.date.today().strftime('%Y%m')}-{uuid.uuid4().hex[:5].upper()}"

    return {
        # ── Header ────────────────────────────────────────────────────────────
        "quote_number":      quote_number,
        "date":              datetime.date.today().strftime("%d %B %Y"),
        "client_name":       requirements.get("client_name") or "Valued Client",
        "prepared_by":       COMPANY_NAME,
        # ── Scope ─────────────────────────────────────────────────────────────
        "project_type":      requirements.get("project_type", "Custom Digital Project"),
        "special_requirements": requirements.get("special_requirements"),
        "timeline_label":    timeline_label,
        "timeline_weeks":    weeks,
        # ── Line items ────────────────────────────────────────────────────────
        "grouped_items":     grouped,
        "line_items":        line_items,
        # ── Financials ────────────────────────────────────────────────────────
        "subtotal":          totals["subtotal"],
        "vat":               totals["vat"],
        "grand_total":       totals["grand_total"],
        "vat_rate":          VAT_RATE,
        # ── Recommendations ───────────────────────────────────────────────────
        "addons":            addons,
        # ── Terms ─────────────────────────────────────────────────────────────
        "payment_terms":     _payment_terms(totals["grand_total"]),
        "scope_summary":     _scope_summary(requirements),
        "terms_conditions":  _terms_and_conditions(),
    }


def _payment_terms(total: float) -> list[str]:
    """Return milestone-based payment terms scaled to project value."""
    if total < 3000:
        return [
            "50% deposit required before work commences.",
            "50% balance due upon project completion.",
        ]
    return [
        "40% deposit required before work commences.",
        "30% upon delivery of design mockups / mid-project milestone.",
        "30% balance due upon final delivery and sign-off.",
    ]


def _scope_summary(req: dict) -> list[str]:
    """Generate bullet-point scope of work from the requirements."""
    bullets: list[str] = []
    service_map = {s["service_key"]: s["quantity"] for s in req["services"]}

    if "website_page" in service_map:
        n = int(service_map["website_page"])
        bullets.append(f"Design and develop a {n}-page responsive website.")
    if "ecommerce" in service_map:
        bullets.append("Build a full e-commerce storefront with product management.")
    if "contact_form" in service_map:
        bullets.append("Implement a secure contact / inquiry form with notifications.")
    if "basic_seo" in service_map or "advanced_seo" in service_map:
        tier = "advanced" if "advanced_seo" in service_map else "basic"
        bullets.append(f"Deliver {tier} SEO optimisation and search console setup.")
    if "hosting" in service_map:
        yr = int(service_map["hosting"])
        bullets.append(f"Configure managed cloud hosting for {yr} year(s) with SSL.")
    if "domain_setup" in service_map:
        bullets.append("Register domain name and configure DNS records.")
    if "logo_design" in service_map:
        bullets.append("Create professional logo with full brand guidelines.")
    if "payment_gateway" in service_map:
        bullets.append("Integrate online payment processing with automated receipts.")
    if "monthly_maintenance" in service_map:
        mo = int(service_map["monthly_maintenance"])
        bullets.append(f"Provide {mo}-month website maintenance and support retainer.")
    if "ai_chatbot" in service_map:
        bullets.append("Deploy AI-powered chatbot trained on client content.")
    if "content_writing" in service_map:
        pg = int(service_map["content_writing"])
        bullets.append(f"Write SEO-optimised copy for {pg} page(s).")
    if "uiux_design" in service_map:
        bullets.append("Produce full UI/UX wireframes and interactive prototype.")
    if "google_analytics" in service_map:
        bullets.append("Set up Google Analytics 4 and conversion tracking.")
    if "social_media_setup" in service_map:
        bullets.append("Create and brand social media profiles across key platforms.")

    if not bullets:
        bullets.append("Deliver custom digital project per client specifications.")
    return bullets


def _terms_and_conditions() -> list[str]:
    return [
        "This quotation is valid for 30 days from the date of issue.",
        "All prices are in Saudi Riyal (SAR) and exclude VAT unless stated otherwise.",
        "VAT is charged at the prevailing rate (currently 15%) where applicable.",
        "Scope changes after project kick-off may be subject to a change-order fee.",
        f"Intellectual property transfers to the client upon receipt of final payment.",
        "Hosting and domain renewals are billed annually and subject to price changes.",
        f"For queries, contact us at {COMPANY_EMAIL}.",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    """Inject all custom CSS for the dark, modern theme."""
    st.markdown("""
    <style>
    /* ── Google Fonts ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Root tokens ──────────────────────────────────────────────────────── */
    :root {
        --bg-base:        #0d0f14;
        --bg-card:        #141820;
        --bg-elevated:    #1c2230;
        --bg-input:       #1a1f2b;
        --accent:         #6c63ff;
        --accent-soft:    #8b84ff;
        --accent-glow:    rgba(108,99,255,0.18);
        --success:        #22c55e;
        --warning:        #f59e0b;
        --danger:         #ef4444;
        --text-primary:   #f0f2f8;
        --text-secondary: #8b93a8;
        --text-muted:     #545d73;
        --border:         rgba(255,255,255,0.07);
        --border-accent:  rgba(108,99,255,0.35);
        --font-display:   'Space Grotesk', sans-serif;
        --font-body:      'Inter', sans-serif;
        --radius-sm:      8px;
        --radius-md:      14px;
        --radius-lg:      20px;
        --shadow-card:    0 4px 24px rgba(0,0,0,0.35);
    }

    /* ── Global overrides ─────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: var(--font-body) !important;
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
    }
    .stApp { background: var(--bg-base); }
    .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1100px; }

    /* ── Hide Streamlit chrome ────────────────────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero header ──────────────────────────────────────────────────────── */
    .qf-hero {
        background: linear-gradient(135deg, #1a1635 0%, #0d0f14 60%);
        border: 1px solid var(--border-accent);
        border-radius: var(--radius-lg);
        padding: 2.8rem 3rem 2.4rem;
        margin-bottom: 2.4rem;
        position: relative;
        overflow: hidden;
    }
    .qf-hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(108,99,255,0.2) 0%, transparent 70%);
        pointer-events: none;
    }
    .qf-hero-badge {
        display: inline-block;
        background: var(--accent-glow);
        border: 1px solid var(--border-accent);
        color: var(--accent-soft);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.28rem 0.85rem;
        border-radius: 100px;
        margin-bottom: 1rem;
    }
    .qf-hero h1 {
        font-family: var(--font-display) !important;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        line-height: 1.15 !important;
        margin: 0 0 0.7rem !important;
        background: linear-gradient(135deg, #ffffff 30%, #a099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .qf-hero p {
        color: var(--text-secondary);
        font-size: 1.05rem;
        max-width: 560px;
        line-height: 1.65;
        margin: 0;
    }

    /* ── Cards ────────────────────────────────────────────────────────────── */
    .qf-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow: var(--shadow-card);
    }
    .qf-card-accent {
        border-color: var(--border-accent);
        background: linear-gradient(135deg, #141820 80%, #1a1635);
    }

    /* ── Section labels ───────────────────────────────────────────────────── */
    .qf-section-label {
        font-family: var(--font-display);
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--accent-soft);
        margin-bottom: 0.5rem;
    }
    .qf-section-title {
        font-family: var(--font-display);
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.2rem;
    }

    /* ── Quote header block ───────────────────────────────────────────────── */
    .qf-quote-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 1rem;
        padding-bottom: 1.4rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.8rem;
    }
    .qf-quote-number {
        font-family: var(--font-display);
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .qf-quote-meta span {
        display: block;
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-bottom: 0.22rem;
    }
    .qf-quote-meta strong { color: var(--text-primary); }
    .qf-client-pill {
        background: var(--accent-glow);
        border: 1px solid var(--border-accent);
        border-radius: 100px;
        padding: 0.35rem 1.1rem;
        font-size: 0.88rem;
        color: var(--accent-soft);
        font-weight: 500;
    }

    /* ── Line items table ─────────────────────────────────────────────────── */
    .qf-table-wrap { overflow-x: auto; }
    .qf-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .qf-table th {
        text-align: left;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        padding: 0.55rem 0.9rem;
        border-bottom: 1px solid var(--border);
    }
    .qf-table th.right, .qf-table td.right { text-align: right; }
    .qf-table td {
        padding: 0.75rem 0.9rem;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        color: var(--text-primary);
        vertical-align: top;
    }
    .qf-table tr:last-child td { border-bottom: none; }
    .qf-table .cat-row td {
        background: rgba(108,99,255,0.06);
        color: var(--accent-soft);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.4rem 0.9rem;
    }
    .qf-table .item-desc {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.18rem;
    }
    .qf-table .line-total { font-weight: 600; }

    /* ── Totals block ─────────────────────────────────────────────────────── */
    .qf-totals {
        margin-top: 1.2rem;
        border-top: 1px solid var(--border);
        padding-top: 1rem;
    }
    .qf-totals-row {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 1.8rem;
        padding: 0.3rem 0.9rem;
        font-size: 0.88rem;
        color: var(--text-secondary);
    }
    .qf-totals-row .label { min-width: 120px; text-align: right; }
    .qf-totals-row .value { min-width: 90px; text-align: right; }
    .qf-totals-grand {
        background: var(--accent-glow);
        border-radius: var(--radius-sm);
        padding: 0.7rem 0.9rem;
        margin-top: 0.4rem;
    }
    .qf-totals-grand .label,
    .qf-totals-grand .value {
        font-family: var(--font-display);
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary) !important;
    }

    /* ── Scope bullets ────────────────────────────────────────────────────── */
    .qf-scope-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }
    .qf-scope-item:last-child { border-bottom: none; }
    .qf-scope-dot {
        width: 7px; height: 7px;
        background: var(--accent);
        border-radius: 50%;
        margin-top: 0.48rem;
        flex-shrink: 0;
    }

    /* ── Add-on cards ─────────────────────────────────────────────────────── */
    .qf-addon-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 1rem;
    }
    .qf-addon-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.3rem;
        transition: border-color 0.2s;
    }
    .qf-addon-card:hover { border-color: var(--border-accent); }
    .qf-addon-name {
        font-family: var(--font-display);
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }
    .qf-addon-price {
        font-size: 0.82rem;
        color: var(--accent-soft);
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .qf-addon-reason {
        font-size: 0.78rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* ── Timeline pill ────────────────────────────────────────────────────── */
    .qf-timeline-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.25);
        color: #22c55e;
        padding: 0.35rem 1rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* ── Terms list ───────────────────────────────────────────────────────── */
    .qf-terms-item {
        font-size: 0.82rem;
        color: var(--text-muted);
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        line-height: 1.55;
    }
    .qf-terms-item:last-child { border-bottom: none; }

    /* ── Payment term ─────────────────────────────────────────────────────── */
    .qf-payment-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.6rem 0;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    .qf-payment-icon {
        color: var(--warning);
        font-size: 1rem;
    }

    /* ── Stat chips ───────────────────────────────────────────────────────── */
    .qf-stat-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.6rem;
    }
    .qf-stat-chip {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.7rem 1.2rem;
        min-width: 120px;
    }
    .qf-stat-chip .chip-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .qf-stat-chip .chip-value {
        font-family: var(--font-display);
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 0.15rem;
    }

    /* ── Input area ───────────────────────────────────────────────────────── */
    .stTextArea textarea {
        background: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-size: 0.93rem !important;
        padding: 1rem !important;
        transition: border-color 0.2s !important;
        resize: vertical !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--border-accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    /* ── Buttons ──────────────────────────────────────────────────────────── */
    .stButton > button {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-display) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 2rem !important;
        cursor: pointer !important;
        transition: opacity 0.2s, transform 0.15s !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Select boxes ─────────────────────────────────────────────────────── */
    .stSelectbox > div > div {
        background: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-secondary) !important; }

    /* ── Divider ──────────────────────────────────────────────────────────── */
    hr { border-color: var(--border) !important; margin: 1.8rem 0 !important; }

    /* ── Spinner ──────────────────────────────────────────────────────────── */
    .stSpinner > div { border-top-color: var(--accent) !important; }
    </style>
    """, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown("""
    <div class="qf-hero">
        <div class="qf-hero-badge">⚡ AI-Powered &nbsp;·&nbsp; GPT-4o-mini</div>
        <h1>QuoteForge AI</h1>
        <p>Describe your project in plain English and receive a detailed,
        professionally structured quotation in seconds — complete with
        itemised pricing, scope of work, and smart add-on recommendations.</p>
    </div>
    """, unsafe_allow_html=True)


def render_quotation(q: dict) -> None:
    """Render the full quotation document using the q dict from build_quotation()."""

    # ── Stat chips ────────────────────────────────────────────────────────────
    item_count = len(q["line_items"])
    st.markdown(f"""
    <div class="qf-stat-row">
        <div class="qf-stat-chip">
            <div class="chip-label">Grand Total</div>
            <div class="chip-value">SAR&nbsp;{q['grand_total']:,.2f}</div>
        </div>
        <div class="qf-stat-chip">
            <div class="chip-label">Services</div>
            <div class="chip-value">{item_count}</div>
        </div>
        <div class="qf-stat-chip">
            <div class="chip-label">Timeline</div>
            <div class="chip-value">{q['timeline_weeks']}w</div>
        </div>
        <div class="qf-stat-chip">
            <div class="chip-label">VAT ({int(q['vat_rate']*100)}%)</div>
            <div class="chip-value">SAR&nbsp;{q['vat']:,.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quote header ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="qf-card qf-card-accent">
        <div class="qf-quote-header">
            <div>
                <div class="qf-section-label">Quotation</div>
                <div class="qf-quote-number">{q['quote_number']}</div>
            </div>
            <div class="qf-quote-meta">
                <span>Date: <strong>{q['date']}</strong></span>
                <span>Prepared by: <strong>{q['prepared_by']}</strong></span>
                <span>Project: <strong>{q['project_type']}</strong></span>
            </div>
        </div>
        <div>
            <div class="qf-section-label">Prepared for</div>
            <span class="qf-client-pill">👤 {q['client_name']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scope of Work ─────────────────────────────────────────────────────────
    scope_html = "".join(
        f'<div class="qf-scope-item"><div class="qf-scope-dot"></div><span>{b}</span></div>'
        for b in q["scope_summary"]
    )
    st.markdown(f"""
    <div class="qf-card">
        <div class="qf-section-label">Scope of Work</div>
        <div class="qf-section-title">What's included</div>
        {scope_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Line Items ────────────────────────────────────────────────────────────
    rows_html = ""
    for category, items in q["grouped_items"].items():
        rows_html += f'<tr class="cat-row"><td colspan="5">{category}</td></tr>'
        for item in items:
            qty_display = int(item["quantity"]) if item["quantity"] == int(item["quantity"]) else item["quantity"]
            rows_html += f"""
            <tr>
                <td>
                    <strong>{item['label']}</strong>
                    <div class="item-desc">{item['description']}</div>
                </td>
                <td class="right">{qty_display}</td>
                <td>{item['unit']}</td>
                <td class="right">SAR {item['unit_price']:,.2f}</td>
                <td class="right line-total">SAR {item['line_total']:,.2f}</td>
            </tr>"""

    totals_html = f"""
    <div class="qf-totals">
        <div class="qf-totals-row">
            <span class="label">Subtotal</span>
            <span class="value">SAR {q['subtotal']:,.2f}</span>
        </div>
        <div class="qf-totals-row">
            <span class="label">VAT ({int(q['vat_rate']*100)}%)</span>
            <span class="value">SAR {q['vat']:,.2f}</span>
        </div>
        <div class="qf-totals-row qf-totals-grand">
            <span class="label">Total Due</span>
            <span class="value">SAR {q['grand_total']:,.2f}</span>
        </div>
    </div>"""

    st.markdown(f"""
    <div class="qf-card">
        <div class="qf-section-label">Pricing Breakdown</div>
        <div class="qf-section-title">Line Items</div>
        <div class="qf-table-wrap">
        <table class="qf-table">
            <thead>
                <tr>
                    <th>Service</th>
                    <th class="right">Qty</th>
                    <th>Unit</th>
                    <th class="right">Unit Price</th>
                    <th class="right">Total</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        {totals_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="qf-card">
        <div class="qf-section-label">Project Timeline</div>
        <div class="qf-section-title">Estimated Delivery</div>
        <span class="qf-timeline-pill">🗓 {q['timeline_label']}</span>
        <p style="color:var(--text-muted);font-size:0.83rem;margin-top:0.8rem;line-height:1.6;">
        Timeline is an estimate based on standard production schedules.
        Exact delivery will be confirmed upon project kick-off and sign-off of scope.
        Rush delivery may incur an additional surcharge.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Payment Terms ─────────────────────────────────────────────────────────
    payment_html = "".join(
        f'<div class="qf-payment-item"><span class="qf-payment-icon">💳</span>{term}</div>'
        for term in q["payment_terms"]
    )
    st.markdown(f"""
    <div class="qf-card">
        <div class="qf-section-label">Payment Terms</div>
        <div class="qf-section-title">Billing Milestones</div>
        {payment_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Recommended Add-ons ───────────────────────────────────────────────────
    if q["addons"]:
        addon_cards = "".join(f"""
        <div class="qf-addon-card">
            <div class="qf-addon-name">{a['label']}</div>
            <div class="qf-addon-price">From SAR {a['unit_price']:,} / {a['unit']}</div>
            <div class="qf-addon-reason">{a['reason']}</div>
        </div>""" for a in q["addons"])

        st.markdown(f"""
        <div class="qf-card">
            <div class="qf-section-label">Smart Recommendations</div>
            <div class="qf-section-title">Suggested Add-ons</div>
            <div class="qf-addon-grid">{addon_cards}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Special Requirements ──────────────────────────────────────────────────
    if q.get("special_requirements"):
        st.markdown(f"""
        <div class="qf-card">
            <div class="qf-section-label">Client Notes</div>
            <div class="qf-section-title">Special Requirements</div>
            <p style="color:var(--text-secondary);font-size:0.9rem;line-height:1.65;margin:0;">
            {q['special_requirements']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Terms & Conditions ────────────────────────────────────────────────────
    terms_html = "".join(
        f'<div class="qf-terms-item">• {t}</div>'
        for t in q["terms_conditions"]
    )
    st.markdown(f"""
    <div class="qf-card">
        <div class="qf-section-label">Legal</div>
        <div class="qf-section-title">Terms & Conditions</div>
        {terms_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SAMPLE QUOTATIONS
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_REQUESTS: dict[str, str] = {
    "Corporate Website": (
        "We need a professional corporate website for TechVentures Ltd. "
        "We want 8 pages, a contact form, basic SEO, hosting for 2 years, "
        "domain setup, and 6 months of monthly maintenance. "
        "Timeline: please complete within 6 weeks."
    ),
    "E-commerce Store": (
        "Build an online store for StyleHouse Fashion. "
        "We need 5 product pages, full e-commerce functionality, "
        "payment gateway integration, basic SEO, hosting for 1 year, "
        "logo design, and an AI chatbot for customer support. "
        "Timeline: 10 weeks."
    ),
    "Startup Launch Package": (
        "We are a startup called GreenLeaf Organics. "
        "We need a 4-page website, logo design, UI/UX design, "
        "content writing for all pages, domain setup, hosting, "
        "Google Analytics setup, and social media profile setup. "
        "Timeline: 4 weeks."
    ),
    "Portfolio Site": (
        "Create a personal portfolio website for photographer Alex Moyo. "
        "I need 3 pages, a contact form, basic SEO, and hosting for 1 year. "
        "Timeline: 2 weeks."
    ),
}


def render_sidebar(on_sample_click) -> None:
    """Render the sidebar with sample requests and pricing catalog."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1rem 0 0.5rem;">
            <div style="font-family:var(--font-display,sans-serif);font-size:1.1rem;
                        font-weight:700;color:#f0f2f8;">⚡ QuoteForge</div>
            <div style="font-size:0.75rem;color:#545d73;margin-top:0.2rem;">v{APP_VERSION}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div style="font-size:0.72rem;color:#8b93a8;text-transform:uppercase;'
                    'letter-spacing:0.1em;font-weight:600;margin-bottom:0.7rem;">Sample Requests</div>',
                    unsafe_allow_html=True)

        for label in SAMPLE_REQUESTS:
            if st.button(f"📋 {label}", key=f"sample_{label}", use_container_width=True):
                on_sample_click(label)

        st.markdown("---")
        st.markdown('<div style="font-size:0.72rem;color:#8b93a8;text-transform:uppercase;'
                    'letter-spacing:0.1em;font-weight:600;margin-bottom:0.7rem;">Pricing Catalog</div>',
                    unsafe_allow_html=True)

        for key, svc in PRICING_CATALOG.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<span style="font-size:0.78rem;color:#8b93a8;">{svc["label"]}</span>'
                f'<span style="font-size:0.78rem;color:#6c63ff;font-weight:500;">SAR {svc["unit_price"]:,}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.72rem;color:#545d73;line-height:1.6;">'
            f'All prices in SAR · VAT ({int(VAT_RATE*100)}%) added at checkout<br>'
            f'Powered by GPT-4o-mini</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # ── Session state ─────────────────────────────────────────────────────────
    if "request_text" not in st.session_state:
        st.session_state.request_text = ""
    if "quotation" not in st.session_state:
        st.session_state.quotation = None
    if "error" not in st.session_state:
        st.session_state.error = None

    def load_sample(label: str) -> None:
        st.session_state.request_text = SAMPLE_REQUESTS[label]
        st.session_state.quotation    = None
        st.session_state.error        = None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    render_sidebar(load_sample)

    # ── Hero ──────────────────────────────────────────────────────────────────
    render_hero()

    # ── Input card ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="qf-section-label">Step 1</div>
    <div class="qf-section-title" style="font-size:1.1rem;margin-bottom:0.5rem;">
        Describe your project
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_area(
        label="project_request",
        label_visibility="collapsed",
        value=st.session_state.request_text,
        height=130,
        placeholder=(
            'e.g. "Create a quotation for TechCo — 6-page website, contact form, '
            'basic SEO, hosting for 1 year, and 3 months of monthly maintenance."'
        ),
        key="input_area",
    )
    st.session_state.request_text = user_input

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        generate_clicked = st.button("⚡ Generate Quotation", use_container_width=True)
    with col_hint:
        st.markdown(
            '<p style="color:var(--text-muted,#545d73);font-size:0.8rem;'
            'margin-top:0.7rem;">Include client name, services needed, quantities, and timeline for best results.</p>',
            unsafe_allow_html=True,
        )

    # ── Generate ──────────────────────────────────────────────────────────────
    if generate_clicked:
        st.session_state.quotation = None
        st.session_state.error     = None

        if not user_input.strip():
            st.warning("Please describe your project before generating a quotation.")
        else:
            with st.spinner("Analysing your request with AI…"):
                requirements = extract_requirements(user_input.strip())

            if requirements is None:
                st.session_state.error = "Could not extract requirements. Please rephrase and try again."
            elif not requirements.get("services"):
                st.session_state.error = (
                    "No recognisable services were detected in your request. "
                    "Try mentioning specific services like 'website pages', 'SEO', 'hosting', or 'logo design'."
                )
            else:
                st.session_state.quotation = build_quotation(requirements)

    # ── Output ────────────────────────────────────────────────────────────────
    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.quotation:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div class="qf-section-label">Step 2</div>
        <div class="qf-section-title" style="font-size:1.1rem;margin-bottom:1rem;">
            Your Quotation
        </div>
        """, unsafe_allow_html=True)
        render_quotation(st.session_state.quotation)

        # ── JSON debug toggle ─────────────────────────────────────────────────
        with st.expander("🔍 View raw quotation data (JSON)"):
            st.json(st.session_state.quotation)


if __name__ == "__main__":
    main()
