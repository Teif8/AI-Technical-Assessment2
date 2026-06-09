# ⚡ QuoteForge AI — AI Quotation Generator Agent

A production-quality Streamlit application that converts natural-language project
descriptions into fully itemised, professional digital-agency quotations — powered
by GPT-4o-mini.

-----

## Quick Start

```bash
# 1. Clone / download the project
cd quotation_app

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Open .env and add your OpenAI key:  OPENAI_API_KEY=sk-...

# 5. Run the app
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

-----

##  Project Structure

```
quotation_app/
├── app.py               # Main application (all logic + UI)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── README.md            # This file
```

`app.py` is organised into eight clearly labelled sections:

|#|Section          |Responsibility                              |
|-|-----------------|--------------------------------------------|
|1|Configuration    |Constants, env loading, page config         |
|2|Pricing Catalog  |15 services with prices, units, descriptions|
|3|AI Functions     |GPT-4o-mini extraction + JSON validation    |
|4|Pricing Engine   |Line-item calculation + add-on logic        |
|5|Quotation Builder|Assembles the full quotation document       |
|6|UI Components    |CSS injection + Streamlit rendering helpers |
|7|Sample Quotations|4 pre-built demo requests                   |
|8|Main App         |Page routing and session state              |

-----

##  Pricing Catalog

All prices are in South African Rand (ZAR).

|Service                    |Unit Price|Unit   |
|---------------------------|----------|-------|
|Website Page               |R 350     |page   |
|Contact Form               |R 180     |form   |
|Basic SEO Package          |R 450     |project|
|Advanced SEO Package       |R 950     |project|
|Web Hosting                |R 120     |year   |
|Domain Registration & Setup|R 85      |year   |
|Logo Design                |R 650     |project|
|E-commerce Functionality   |R 1,800   |project|
|Payment Gateway Integration|R 550     |gateway|
|Monthly Maintenance        |R 250     |month  |
|AI Chatbot Integration     |R 1,200   |project|
|Content Writing            |R 95      |page   |
|UI/UX Design               |R 1,100   |project|
|Social Media Profile Setup |R 220     |project|
|Analytics & Tracking Setup |R 180     |project|

### Pricing Assumptions

1. **Website Pages (R 350/page)**: Covers design and front-end development of a single
   responsive page. Does not include custom back-end functionality (quoted separately).
1. **Hosting (R 120/year)**: Based on shared managed hosting for a standard brochure or
   small e-commerce site. High-traffic or enterprise hosting is quoted on request.
1. **Basic SEO (R 450)**: One-time on-page optimisation only (meta tags, sitemap,
   structured data). Ongoing SEO strategy is quoted as Advanced SEO (R 950).
1. **Monthly Maintenance (R 250/month)**: CMS/plugin updates, security monitoring,
   uptime checks, and one minor content edit per month. Does not cover feature additions.
1. **E-commerce (R 1,800)**: Standard WooCommerce/Shopify-level store (up to 50 products).
   Custom product configurators or marketplace features are quoted separately.
1. **AI Chatbot (R 1,200)**: GPT-powered chatbot trained on up to 10 FAQ documents,
   with a simple live-handoff mechanism. Excludes CRM integration.
1. **Logo Design (R 650)**: Three initial concepts, two revision rounds. All source files
   (SVG, EPS, PNG) delivered. Brand guidelines document not included.
1. **Content Writing (R 95/page)**: Up to 600 SEO-optimised words per page in English.
   Specialist technical or legal copy is quoted at R 140/page.
1. **VAT**: Charged at 15% (South African standard rate) on all taxable services.
   International clients may be zero-rated — confirm with your accountant.
1. **Quotation validity**: All quotations are valid for 30 days from the date of issue.

-----

##  AI Usage

### Model

**GPT-4o-mini** via the OpenAI Chat Completions API.

### What the AI does

The LLM’s sole job is **structured information extraction**. Given a natural-language
project description, it returns a JSON object containing:

- `client_name` — extracted if mentioned, else null
- `project_type` — short label (e.g. “Corporate Website”)
- `timeline_weeks` — integer, defaults to 4 if not stated
- `special_requirements` — free-text notes, or null
- `services` — array of `{service_key, quantity}` pairs

### What the AI does NOT do

- **No pricing decisions** — all arithmetic is done in Python.
- **No quotation copy** — scope bullets, terms, and payment milestones are
  generated deterministically by Python functions.
- **No hallucinated services** — the system prompt restricts output to a fixed
  enumeration of known service keys.

### Temperature

Set to `0.1` for near-deterministic, consistent JSON output.

### Prompt engineering

The `SYSTEM_PROMPT` instructs the model to:

1. Return **only** a JSON object (no markdown, no preamble).
1. Map natural-language phrases to exact `service_key` values.
1. Apply default rules (e.g. `timeline_weeks = 4` if unstated).

### Validation pipeline

1. Strip markdown code fences if present.
1. Attempt `json.loads()` — fall back to regex extraction if it fails.
1. Coerce every field to the expected type.
1. Filter out any `service_key` values not in the known catalog.
1. Clamp quantities to `[0.5, 60]`.

-----

##  Security

- API keys are loaded exclusively from environment variables (`python-dotenv`).
- No secrets are hardcoded anywhere in the source.
- `.env` is listed in `.gitignore` (add it if you use Git).

-----

##  Sample Quotation Examples

Four sample requests are built into the sidebar:

### 1. Corporate Website — TechVentures Ltd

> “8 pages, contact form, basic SEO, hosting for 2 years, domain setup,
> 6 months monthly maintenance. Timeline: 6 weeks.”

**Expected total**: ~R 6,665 excl. VAT / ~R 7,664.75 incl. VAT

### 2. E-commerce Store — StyleHouse Fashion

> “5 product pages, e-commerce, payment gateway, basic SEO, hosting 1 year,
> logo design, AI chatbot. Timeline: 10 weeks.”

**Expected total**: ~R 6,820 excl. VAT / ~R 7,843 incl. VAT

### 3. Startup Launch Package — GreenLeaf Organics

> “4-page website, logo, UI/UX, content writing, domain, hosting,
> Google Analytics, social media setup. Timeline: 4 weeks.”

**Expected total**: ~R 4,755 excl. VAT / ~R 5,468.25 incl. VAT

### 4. Portfolio Site — Alex Moyo

> “3 pages, contact form, basic SEO, hosting 1 year. Timeline: 2 weeks.”

**Expected total**: ~R 1,665 excl. VAT / ~R 1,914.75 incl. VAT

-----

##  Limitations

1. **English only** — the extraction prompt and UI are English-only.
1. **Single currency** — ZAR only. Multi-currency support would require a
   currency-conversion step.
1. **No PDF export** — quotations are rendered in-browser only.
1. **LLM accuracy** — very unusual phrasings may confuse the extractor;
   the validation layer will catch most issues, but some edge cases may
   return zero services.
1. **Static pricing** — prices are hard-coded in `PRICING_CATALOG`.
   A database or CMS integration would be needed for dynamic pricing.
1. **No user accounts / history** — each session is stateless.
1. **Rate limits** — governed by your OpenAI tier; high concurrency may
   hit rate limits.

-----

##  Future Improvements

|Priority|Feature                                                   |
|--------|----------------------------------------------------------|
|High    |PDF export with branded letterhead                        |
|High    |Editable line items in-browser before finalising          |
|High    |Email quotation directly to client                        |
|Medium  |Database storage for quotation history                    |
|Medium  |Admin panel for managing the pricing catalog              |
|Medium  |Multi-currency and locale support                         |
|Medium  |Client portal (view + accept quotation online)            |
|Low     |Integration with accounting software (Xero, QuickBooks)   |
|Low     |Proposal builder (full project proposal, not just pricing)|
|Low     |Discount / promo code support                             |

-----

##  Tech Stack

|Component   |Technology                                    |
|------------|----------------------------------------------|
|UI framework|Streamlit                                     |
|AI model    |GPT-4o-mini (OpenAI)                          |
|Language    |Python 3.11+                                  |
|Styling     |Custom CSS (dark theme, Space Grotesk + Inter)|
|Config      |python-dotenv                                 |

-----

##  License

MIT — free to use, modify, and distribute with attribution.