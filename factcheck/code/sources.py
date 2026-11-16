"""Free source connectors. Plain functions (no LLM). Wrapped as MCP tools by search_server.py.
Each returns a list of source dicts and raises on real errors (the MCP layer reports them)."""
import hashlib
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx

from config import (CONTACT_EMAIL, FETCH_TIMEOUT, GOOGLE_FACTCHECK_API_KEY, MAX_TEXT_CHARS,
                    RESULTS_PER_SEARCH)
from textutil import keywords
from trust import domain_of, tier_of

UA = {"User-Agent": f"fact-checker-student-project/1.0 ({CONTACT_EMAIL or 'local use'})"}
_arxiv_lock = threading.Lock()


def make_source(url, title, text, source_type, evidence):
    d = domain_of(url)
    return {"id": hashlib.sha1(url.encode()).hexdigest()[:12], "uri": url, "url": url,
            "title": " ".join((title or url).split()), "text": (text or "")[:MAX_TEXT_CHARS],
            "source_type": source_type, "evidence": evidence, "domain": d, "tier": tier_of(d)}


def _get(url, **kw):
    kw.setdefault("timeout", 12)
    return httpx.get(url, headers=UA, follow_redirects=True, **kw)


# ---------------- general web ----------------
def search_web(query):
    from ddgs import DDGS
    try:
        hits = DDGS().text(query, max_results=RESULTS_PER_SEARCH) or []
    except Exception as e:
        if "no results" in str(e).lower():
            return []
        raise
    return [make_source(h["href"], h.get("title"), h.get("body", ""), "web", "snippet_only")
            for h in hits if h.get("href")]


# ---------------- Wikipedia (free, no key) ----------------
def search_wikipedia(query):
    api = "https://en.wikipedia.org/w/api.php"
    r = _get(api, params={"action": "query", "list": "search", "srsearch": query, "srlimit": 3, "format": "json"})
    out = []
    for h in r.json().get("query", {}).get("search", []):
        pid, title = h["pageid"], h["title"]
        r2 = _get(api, params={"action": "query", "prop": "extracts", "explaintext": 1,
                               "pageids": pid, "format": "json"})
        extract = r2.json()["query"]["pages"][str(pid)].get("extract", "")
        if extract:
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            out.append(make_source(url, title, extract, "wikipedia", "full_page"))
    return out


# ---------------- scholarly: arXiv + OpenAlex (free, no key) ----------------
def _arxiv_get(params):
    with _arxiv_lock:  # arXiv asks for one request at a time
        for _ in range(2):
            r = _get("https://export.arxiv.org/api/query", params=params, timeout=15)
            if r.status_code == 200:
                time.sleep(1)
                return r.text
            if r.status_code != 429:
                raise RuntimeError(f"arXiv HTTP {r.status_code}")
            time.sleep(3)
    raise RuntimeError("arXiv rate-limited (HTTP 429)")


def search_arxiv(query):
    words = [w for w in keywords(query, 8) if re.fullmatch(r"[a-z][a-z0-9\-]+", w)]
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for n, op in ((5, " AND "), (3, " AND "), (3, " OR ")):
        terms = words[:n]
        if not terms:
            return []
        xml = _arxiv_get({"search_query": op.join(f"all:{t}" for t in terms),
                          "max_results": RESULTS_PER_SEARCH, "sortBy": "relevance"})
        out = []
        for e in ET.fromstring(xml).findall("a:entry", ns):
            url = (e.findtext("a:id", namespaces=ns) or "").strip()
            abstract = " ".join((e.findtext("a:summary", namespaces=ns) or "").split())
            if url and abstract:
                out.append(make_source(url, e.findtext("a:title", namespaces=ns), abstract, "arxiv", "abstract"))
        if out:
            return out
    return []


def _abstract(inv):
    if not inv:
        return ""
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))


def search_openalex(query):
    params = {"search": query, "per-page": RESULTS_PER_SEARCH}
    if CONTACT_EMAIL:
        params["mailto"] = CONTACT_EMAIL
    out = []
    for w in _get("https://api.openalex.org/works", params=params).json().get("results", []):
        text = _abstract(w.get("abstract_inverted_index"))
        url = w.get("doi") or w.get("id")
        if text and url:
            out.append(make_source(url, w.get("display_name"), text, "openalex", "abstract"))
    return out


# ---------------- PubMed (free, no key) ----------------
def search_pubmed(query):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    common = {"tool": "fact-checker", **({"email": CONTACT_EMAIL} if CONTACT_EMAIL else {})}
    ids = _get(base + "esearch.fcgi", params={**common, "db": "pubmed", "term": query,
               "retmax": RESULTS_PER_SEARCH, "retmode": "json"}).json()["esearchresult"]["idlist"]
    if not ids:
        return []
    root = ET.fromstring(_get(base + "efetch.fcgi", params={**common, "db": "pubmed", "id": ",".join(ids),
                                                            "retmode": "xml"}).text)
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abstract = " ".join("".join(a.itertext()) for a in art.findall(".//AbstractText"))
        if pmid and abstract:
            out.append(make_source(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", title, abstract, "pubmed", "abstract"))
    return out


# ---------------- Hugging Face model cards (free, no key) ----------------
def search_huggingface(query):
    models = _get("https://huggingface.co/api/models",
                  params={"search": query, "limit": 3, "sort": "downloads", "direction": -1}).json()
    out = []
    for m in models:
        mid = m.get("id") or m.get("modelId")
        if not mid:
            continue
        r = _get(f"https://huggingface.co/{mid}/raw/main/README.md", timeout=FETCH_TIMEOUT)
        if r.status_code == 200 and r.text.strip():
            out.append(make_source(f"https://huggingface.co/{mid}", f"{mid} (model card)", r.text, "huggingface", "full_page"))
    return out


# ---------------- Existing fact-checks (free key, optional) ----------------
def search_factchecks(query):
    if not GOOGLE_FACTCHECK_API_KEY:
        return []
    r = _get("https://factchecktools.googleapis.com/v1alpha1/claims:search",
             params={"query": query, "key": GOOGLE_FACTCHECK_API_KEY, "pageSize": 5})
    out = []
    for c in r.json().get("claims", []):
        for rev in c.get("claimReview", []):
            url = rev.get("url")
            if not url:
                continue
            text = (f'Claim: {c.get("text", "")} | Claimant: {c.get("claimant", "unknown")} | '
                    f'Rating by {rev.get("publisher", {}).get("name", "?")}: {rev.get("textualRating", "?")}')
            out.append(make_source(url, rev.get("title") or "Fact-check", text, "factcheck", "rating"))
    return out


# ---------------- full page ----------------
def fetch_page(url):
    import trafilatura
    r = _get(url, timeout=FETCH_TIMEOUT)
    if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
        return ""
    return (trafilatura.extract(r.text) or "")[:MAX_TEXT_CHARS]
