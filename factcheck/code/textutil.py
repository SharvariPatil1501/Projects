import re

STOPWORDS = set("""a an the and or but if then of to in on at by for with from as is are was were be been being it its this that
these those per than also not no do does did has have had will would can could should may might about into over under
after before between during which who whom what when where why how i you he she we they them his her their our your""".split())


def tokens(s: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9][a-z0-9\.\-']*", s.lower()):
        t = t.strip(".-'")
        if t and t not in STOPWORDS:
            out.append(t)
    return out


def keywords(s: str, n: int) -> list[str]:
    seen, out = set(), []
    for t in tokens(s):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:n]


def overlap(claim: str, text: str) -> float:
    """Fraction of the claim's key words that also appear in the text."""
    c = set(tokens(claim))
    return len(c & set(tokens(text))) / len(c) if c else 0.0
