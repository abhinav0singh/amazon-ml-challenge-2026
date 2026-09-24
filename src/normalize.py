"""
normalize.py: country-agnostic text normalisation for names and addresses.

Design rules (important for the unseen FRANCE test country):
  * Nothing here branches on the country label. Every rule applies to all rows.
  * Accents are stripped with the stdlib (unicodedata NFKD), so "Société" and
    "Societe" compare equal.
  * Dictionaries below are hand-written abbreviation lists. They are NOT
    external data lookups; they only rewrite tokens already in the record.
"""
from __future__ import annotations

import re
import unicodedata

# Legal / company-form tokens, removed to form the "core" business name.
LEGAL_TOKENS = {
    "pvt", "private", "ltd", "limited", "llc", "llp", "lp", "inc", "incorporated",
    "corp", "corporation", "co", "company", "plc", "pllc", "pc", "gmbh", "ag",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "the", "dba",
    "opc", "pty", "bv", "nv", "srl", "spa", "cie",
}

# Token rewrites applied to names (abbreviation -> canonical form).
NAME_MAP = {
    "&": "and", "intl": "international", "mfg": "manufacturing",
    "svcs": "services", "svc": "services", "ent": "enterprises", "bros": "brothers",
    "assoc": "associates", "tech": "technologies", "techs": "technologies",
    "sys": "systems", "mgmt": "management", "dept": "department", "ctr": "center",
    "centre": "center", "st": "saint", "ste": "sainte",
}

# Token rewrites applied to addresses.
ADDR_MAP = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "building", "bldg": "building",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "sq": "square",
    "hwy": "highway", "pkwy": "parkway", "fwy": "freeway", "cir": "circle",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor",
    "nr": "near", "opp": "opposite", "mkt": "market", "ngr": "nagar", "clny": "colony",
    "sec": "sector", "ph": "phase", "no": "number", "chs": "chemin", "rte": "route",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_POSTAL = re.compile(r"\b(\d{5,6})\b")
_NUM = re.compile(r"\d+")


def strip_accents(s: str) -> str:
    """Remove diacritics: 'Société Générale' -> 'Societe Generale'."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def basic_clean(s: str) -> str:
    """Lowercase, strip accents, '&'->' and ', drop punctuation, collapse spaces.
    Dots between letters are removed first so 'p.v.t.' -> 'pvt'."""
    s = strip_accents(s or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", s)
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def _rewrite(tokens, mapping):
    return [mapping.get(t, t) for t in tokens]


def norm_name(s: str) -> str:
    """Full normalised name (legal suffixes kept, abbreviations expanded)."""
    return " ".join(_rewrite(basic_clean(s).split(), NAME_MAP))


def core_name(s: str) -> str:
    """Name without legal-form tokens: 'ABC Pvt. Ltd.' -> 'abc'.
    Falls back to the full normalised name if everything was a legal token."""
    toks = norm_name(s).split()
    core = [t for t in toks if t not in LEGAL_TOKENS]
    return " ".join(core) if core else " ".join(toks)


def norm_addr(s: str) -> str:
    """Normalised address with street/landmark abbreviations expanded."""
    return " ".join(_rewrite(basic_clean(s).split(), ADDR_MAP))


def postal_codes(s: str) -> set:
    """5-6 digit tokens (US ZIP, FR code postal, IN PIN) found in the raw address."""
    return set(_POSTAL.findall(s or ""))


def numbers(s: str) -> set:
    """All digit runs in the address (house numbers, plots, postal codes)."""
    return set(_NUM.findall(s or ""))


def add_normalized_columns(df):
    """Add the normalised columns used by blocking and features (returns a copy)."""
    df = df.copy()
    df["name_n"] = df["business_name"].map(norm_name)
    df["name_core"] = df["business_name"].map(core_name)
    df["addr_n"] = df["business_address"].map(norm_addr)
    df["full_n"] = df["name_core"] + " | " + df["addr_n"]
    df["postal"] = df["business_address"].map(postal_codes)
    df["nums"] = df["business_address"].map(numbers)
    df["country_n"] = df["country"].map(lambda c: basic_clean(c))
    return df
