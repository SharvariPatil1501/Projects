"""Evidence classification. For one statement and one source text, decide: supports / contradicts / unrelated.
Method: pick the source passages that share the most key words with the statement, then ask a local NLI model.
A figure-mismatch heuristic catches cases like '$0.05' vs '$0.50' that NLI models often call 'neutral'."""
import re
import threading

import numpy as np

from config import (CONF_THRESHOLD, MIN_OVERLAP, NLI_MODEL, PASSAGE_CHARS, PASSAGES_PER_SOURCE)
from textutil import tokens

_lock = threading.RLock()
_nli = None
_idx = None

_NUM = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(thousand|million|billion|k|m|b)?\b", re.I)
_MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9}


def load_nli():
    global _nli, _idx
    with _lock:
        if _nli is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(NLI_MODEL)
            id2 = model.model.config.id2label
            _idx = {k: next(int(i) for i, l in id2.items() if key in l.lower())
                    for k, key in (("ent", "entail"), ("con", "contra"), ("neu", "neutral"))}
            _nli = model
    return _nli, _idx


def nli_probs(pairs):
    """pairs: [(premise, hypothesis)] -> [{ent, con, neu}]"""
    model, idx = load_nli()
    with _lock:
        out = np.asarray(model.predict(pairs), dtype=float)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if not np.allclose(out.sum(axis=1), 1.0, atol=1e-3):  # logits -> probabilities
        e = np.exp(out - out.max(axis=1, keepdims=True))
        out = e / e.sum(axis=1, keepdims=True)
    return [{k: float(r[i]) for k, i in idx.items()} for r in out]


def numbers(s: str) -> set:
    """Figures in text, normalised ('1M' == '1,000,000'). Bare years are ignored."""
    out = set()
    for n, suf in _NUM.findall(s):
        v = float(n.replace(",", ""))
        if not suf and v.is_integer() and 1900 <= v <= 2100:
            continue
        out.add(round(v * _MULT.get((suf or "").lower(), 1), 6))
    return out


def split_passages(text: str) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]
    return [" ".join(sents[i:i + 2])[:PASSAGE_CHARS] for i in range(len(sents))]


def top_passages(claim: str, text: str, k: int = PASSAGES_PER_SOURCE):
    ctoks = set(tokens(claim))
    if not ctoks:
        return []
    scored = []
    for p in split_passages(text):
        ov = len(ctoks & set(tokens(p))) / len(ctoks)
        if ov >= MIN_OVERLAP:
            scored.append((ov, p))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def classify_source(claim: str, text: str, nli=nli_probs) -> dict:
    passages = top_passages(claim, text)
    base = {"label": "unrelated", "confidence": 0.0, "quote": "", "reason": "no relevant passage", "method": "none"}
    if not passages:
        return base
    cnums = numbers(claim)
    probs = nli([(p, claim) for _, p in passages])
    best = None
    for (ov, p), pr in zip(passages, probs):
        pnums = numbers(p)
        cand = None
        if pr["ent"] >= CONF_THRESHOLD:
            if cnums - pnums:
                cand = ("unrelated", pr["ent"], "figures in the statement are not in the passage", "nli+numbers")
            else:
                cand = ("supports", pr["ent"], "passage entails the statement", "nli")
        elif pr["con"] >= CONF_THRESHOLD:
            cand = ("contradicts", pr["con"], "passage contradicts the statement", "nli")
        elif cnums and pnums and not (cnums & pnums) and ov >= 0.6:
            cand = ("contradicts", 0.7, "same topic but a different figure (heuristic)", "numeric")
        if cand and (best is None or (cand[0] != "unrelated" and
                                      (best[0][0] == "unrelated" or cand[1] > best[0][1]))):
            best = (cand, p)
    if best is None:
        return {**base, "reason": "passages are neutral to the statement", "quote": passages[0][1]}
    (label, conf, reason, method), quote = best
    return {"label": label, "confidence": round(conf, 3), "quote": quote, "reason": reason, "method": method}


def classify_sources(claim: str, sources: list[dict], nli=nli_probs) -> list[dict]:
    return [{"uri": s["uri"], **classify_source(claim, s.get("text", ""), nli)} for s in sources]
