from urllib.parse import urlparse

from config import TIER1_DOMAINS, TIER1_SUFFIXES, TIER2_DOMAINS

TIER_NAMES = {1: "Primary / official", 2: "Reputable", 3: "General web"}


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().split(":")[0].removeprefix("www.")


def _match(domain: str, names) -> bool:
    return any(domain == n or domain.endswith("." + n) for n in names)


def tier_of(domain: str) -> int:
    if _match(domain, TIER1_DOMAINS) or domain.endswith(TIER1_SUFFIXES):
        return 1
    if _match(domain, TIER2_DOMAINS):
        return 2
    return 3
