"""Run: python test_logic.py  -- tests verdict logic and evidence rules. No network, no models, no API key."""
from judge import judge_part, overall_verdict
from verifier import classify_source, numbers

NEUTRAL = lambda pairs: [{"ent": 0.1, "con": 0.1, "neu": 0.8} for _ in pairs]
ENTAIL = lambda pairs: [{"ent": 0.95, "con": 0.02, "neu": 0.03} for _ in pairs]
CONTRA = lambda pairs: [{"ent": 0.02, "con": 0.95, "neu": 0.03} for _ in pairs]

# --- figures ---
assert numbers("1M tokens") == numbers("1,000,000 tokens")
assert numbers("costs $0.05") != numbers("costs $0.50")
assert numbers("released in 2026") == set()

# --- classification ---
text = "Model X is priced at $0.50 per million tokens. It was released last year."
r = classify_source("Model X costs $0.05 per million tokens", text, NEUTRAL)
assert r["label"] == "contradicts" and r["method"] == "numeric", r   # the $0.05 vs $0.50 case
r = classify_source("Model X costs $0.50 per million tokens", text, ENTAIL)
assert r["label"] == "supports", r
r = classify_source("Model X costs $9 per million tokens", text, ENTAIL)
assert r["label"] == "unrelated", r                                   # NLI says yes but figure is not in passage
assert classify_source("Model X is priced low", text, CONTRA)["label"] == "contradicts"
assert classify_source("Penguins live in Antarctica", text, ENTAIL)["label"] == "unrelated"  # no relevant passage


def ev(label, tier, quote, domain, conf=0.9):
    return {"label": label, "tier": tier, "quote": quote, "domain": domain, "confidence": conf}


# --- verdicts ---
assert judge_part([ev("supports", 1, "a", "arxiv.org")])["verdict"] == "SUPPORTED"
assert judge_part([ev("supports", 3, "a", "blog1.com"), ev("supports", 3, "b", "blog2.com"),
                   ev("supports", 3, "c", "blog3.com")])["verdict"] == "NOT_ENOUGH_EVIDENCE"   # low-trust only
assert judge_part([ev("supports", 2, "a", "wikipedia.org")])["verdict"] == "NOT_ENOUGH_EVIDENCE"  # one tier-2 is not enough
assert judge_part([ev("contradicts", 1, "a", "nih.gov")])["verdict"] == "CONTRADICTED"
assert judge_part([ev("supports", 1, "a", "arxiv.org"), ev("contradicts", 1, "b", "nih.gov")])["verdict"] == "MIXED"
assert judge_part([ev("supports", 1, "a", "arxiv.org"), ev("supports", 1, "b", "nih.gov"),
                   ev("contradicts", 2, "c", "bbc.com")])["verdict"] == "SUPPORTED"
assert judge_part([ev("supports", 1, "same", "a.edu"), ev("supports", 1, "same", "b.edu")])["support_weight"] == 3  # duplicate quote counted once
assert judge_part([ev("supports", 1, "a", "arxiv.org", conf=0.5)])["verdict"] == "NOT_ENOUGH_EVIDENCE"
assert overall_verdict(["SUPPORTED", "SUPPORTED"]) == "SUPPORTED"
assert overall_verdict(["SUPPORTED", "NOT_ENOUGH_EVIDENCE"]) == "PARTIALLY_SUPPORTED"
assert overall_verdict(["SUPPORTED", "CONTRADICTED"]) == "CONTRADICTED"
assert overall_verdict(["NOT_ENOUGH_EVIDENCE"]) == "NOT_ENOUGH_EVIDENCE"
print("All logic tests passed.")
