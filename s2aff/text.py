import re
import numpy as np
from text_unidecode import unidecode
import pandas as pd

RE_BASIC = re.compile(r"[^,a-zA-Z0-9\s]+")
RE_APOSTRAPHE_S = re.compile(r"(\w+)'s")

ABBREVIATION_DICTIONARY = {
    "Academy": "Acad",
    "American": "Amer",
    "Association": "Assoc",
    "College": "Coll",
    "Company": "Co",
    "Corporation": "Corp",
    "Communication": "Commun",
    "Department": "Dept",
    "Division": "Div",
    "Doctor": "Dr",
    "Electrical": "Electr",
    "Engineering": "Eng",
    "European": "Europ",
    "Executive": "Exec",
    "Faculty": "Fac",
    "Foundation": "Found",
    "Government": "Gov",
    "Incorporated": "Inc",
    "Information": "Info",
    "Institute": "Inst",
    "International": "Intern",
    "Laboratory": "Lab",
    "Library": "Libr",
    "National": "Nat",
    "Medicine": "Med",
    "Mechanical": "Mech",
    "Professor": "Prof",
    "Program": "Progr",
    "Psychology": "Psychol",
    "School": "Sch",
    "Science": "Sci",
    "Society": "Soc",
    "Technology": "Technol",
    "University": "Univ",
}

INVERTED_ABBREVIATION_DICTIONARY = {v.lower(): k.lower() for k, v in ABBREVIATION_DICTIONARY.items()}
INVERTED_ABBREVIATION_DICTIONARY["tech"] = "technology"

STOPWORDS = set(
    [
        "it",
        "its",
        "is",
        "are",
        "a",
        "an",
        "the",
        "and",
        "as",
        "of",
        "at",
        "by",
        "for",
        "with",
        "into",
        "from",
        "in",
    ]
)


CERTAINLY_MAIN = {
    "Associates",
    "Trust",
    "Foundation",
    "Society",
    "Museum",
    "Association",
    "Universitat",
    "Univ",
    "Universita",
    "Universite",
    "Universidad",
    "Universidade",
    "Universitaria",
    "University",
    "Universitas",
    "Institute",
    "Institut",
    "Center",
    "Centre",
    "Centro",
    "Istituto",
}


def fix_text(s):
    """General purpose text fixing"""
    if pd.isnull(s) or len(s) == 0:
        return ""

    s = unidecode(s).replace("#TAB#", "").replace(".", "").replace(" & ", " and ").replace("&", "n")
    s = RE_APOSTRAPHE_S.sub(r"\1s", s)
    s = RE_BASIC.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_geoname_id(value):
    """Normalize Geonames identifiers to consistent string keys."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value)
