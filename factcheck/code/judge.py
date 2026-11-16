"""The verdict is decided by plain code from the classified evidence. No LLM involved."""
from config import CONF_THRESHOLD, MIN_WEIGHT
from trust import TIER_NAMES

WEIGHT = {1: 3, 2: 2, 3: 1}
VERDICT_TEXT = {
    "SUPPORTED": "Supported by the sources",
    "CONTRADICTED": "Contradicted by the sources",
    "MIXED": "Sources disagree",
    "PARTIALLY_SUPPORTED": "Partly supported (other parts lack evidence)",
    "NOT_ENOUGH_EVIDENCE": "Not enough evidence",
    "ERROR": "Verification could not run",
}


def _dedupe(items):
    """Count repeated copies of the same passage, or many pages of one site, only once or twice."""
    seen, per_domain, out = set(), {}, []
    for e in sorted(items, key=lambda e: (e["tier"], -e["confidence"])):
        key = " ".join(e["quote"].lower().split())[:80]
        if key in seen or per_domain.get(e["domain"], 0) >= 2:
            continue
        seen.add(key)
        per_domain[e["domain"]] = per_domain.get(e["domain"], 0) + 1
        out.append(e)
    return out


def judge_part(evidence: list[dict]) -> dict:
    decisive = [e for e in evidence if e["label"] in ("supports", "contradicts") and e["confidence"] >= CONF_THRESHOLD]
    sup = _dedupe([e for e in decisive if e["label"] == "supports"])
    con = _dedupe([e for e in decisive if e["label"] == "contradicts"])
    s = sum(WEIGHT[e["tier"]] for e in sup)
    c = sum(WEIGHT[e["tier"]] for e in con)
    strong_s = s >= MIN_WEIGHT and any(e["tier"] <= 2 for e in sup)
    strong_c = c >= MIN_WEIGHT and any(e["tier"] <= 2 for e in con)
    note = ""
    if strong_s and strong_c:
        if s >= 2 * c:
            verdict, note = "SUPPORTED", "some sources disagree"
        elif c >= 2 * s:
            verdict, note = "CONTRADICTED", "some sources agree with the statement"
        else:
            verdict = "MIXED"
    elif strong_s:
        verdict = "SUPPORTED"
    elif strong_c:
        verdict = "CONTRADICTED"
    else:
        verdict = "NOT_ENOUGH_EVIDENCE"
        if sup or con:
            note = "only weak or low-trust evidence was found"
    return {"verdict": verdict, "note": note, "supports": sup, "contradicts": con,
            "support_weight": s, "contradict_weight": c,
            "unrelated": sum(1 for e in evidence if e["label"] == "unrelated")}


def overall_verdict(part_verdicts: list[str]) -> str:
    if "CONTRADICTED" in part_verdicts:
        return "CONTRADICTED"
    if "MIXED" in part_verdicts:
        return "MIXED"
    if all(v == "SUPPORTED" for v in part_verdicts):
        return "SUPPORTED"
    if "SUPPORTED" in part_verdicts:
        return "PARTIALLY_SUPPORTED"
    return "NOT_ENOUGH_EVIDENCE"


def to_markdown(state: dict) -> str:
    md = [f'# Fact-check: {state["claim"]}', "",
          f'**Verdict: {VERDICT_TEXT.get(state.get("verdict"), state.get("verdict"))}**', ""]
    if state.get("explanation"):
        md += [state["explanation"], ""]
    for i, p in enumerate(state.get("parts", []), 1):
        md += [f'## Part {i}: {p["text"]}', f'Verdict: {VERDICT_TEXT[p["verdict"]]}' + (f' ({p["note"]})' if p.get("note") else ""), ""]
        for title, key in (("Evidence for", "supports"), ("Evidence against", "contradicts")):
            if p[key]:
                md.append(f"### {title}")
                for e in p[key]:
                    md.append(f'- [{e["title"]}]({e["url"]}) ({TIER_NAMES[e["tier"]]}): "{e["quote"]}"')
                md.append("")
    md.append("_Verdicts describe what the retrieved sources say, not absolute truth._")
    return "\n".join(md)
