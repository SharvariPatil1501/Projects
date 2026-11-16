"""All settings in one place. Override with environment variables where shown."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- LLM (used only to plan searches and to write the short explanation) ----
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = [m.strip() for m in os.getenv("FALLBACK_MODELS", "gemini-3.1-flash-lite").split(",") if m.strip()]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_THINKING_BUDGET = os.getenv("GEMINI_THINKING_BUDGET")
LLM_ATTEMPTS_PER_MODEL = 2
LLM_TIMEOUT_S = 40

# ---- Free sources ----
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")                  # optional, polite-pool courtesy for OpenAlex/PubMed
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")  # optional, free: Google Fact Check Tools API
RESULTS_PER_SEARCH = 5
MAX_SOURCES = 30          # sources kept after ranking
FETCH_FULL_TOP = 5        # web results upgraded from snippet to full page
FETCH_TIMEOUT = 6
MAX_TEXT_CHARS = 6000

# ---- Verification ----
NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")  # lighter: cross-encoder/nli-MiniLM2-L6-H768
MAX_PARTS = int(os.getenv("MAX_PARTS", "6"))  # safety cap on statement parts (each part adds searches + NLI time)
CLASSIFY_PER_PART = 20    # sources checked per statement part
PASSAGES_PER_SOURCE = 3
PASSAGE_CHARS = 500
MIN_OVERLAP = 0.25        # a passage must share this fraction of the statement's key words
CONF_THRESHOLD = 0.7      # minimum confidence for 'supports' / 'contradicts'
MIN_WEIGHT = 3            # evidence weight needed for a verdict (tier1=3, tier2=2, tier3=1)

# ---- Source trust tiers ----
TIER1_DOMAINS = {
    "arxiv.org", "openalex.org", "doi.org", "ncbi.nlm.nih.gov", "nih.gov", "who.int", "cdc.gov",
    "nature.com", "science.org", "thelancet.com", "nejm.org", "ieee.org", "acm.org", "openreview.net",
    "aclanthology.org", "huggingface.co", "un.org", "worldbank.org", "europa.eu",
}
TIER1_SUFFIXES = (".gov", ".edu", ".gov.uk", ".ac.uk", ".gov.au", ".gc.ca")
TIER2_DOMAINS = {
    "wikipedia.org", "britannica.com", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "nytimes.com", "theguardian.com", "washingtonpost.com", "snopes.com", "politifact.com",
    "factcheck.org", "fullfact.org", "arstechnica.com", "scientificamerican.com",
}
