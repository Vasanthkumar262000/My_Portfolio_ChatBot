import re
from typing import Tuple

TYPO_CORRECTIONS = {
    "experiance": "experience",
    "experiences": "experience",
    "expirience": "experience",
    "experince": "experience",
    "relevent": "relevant",
    "relevency": "relevance",
    "relevence": "relevance",
    "total experiance": "total experience",
    "total expirience": "total experience",
    "work experiance": "work experience",
    "work expirience": "work experience",
    "skils": "skills",
    "skil": "skill",
    "tecnology": "technology",
    "tecnologies": "technologies",
    "progamming": "programming",
    "programing": "programming",
    "developement": "development",
    "backened": "backend",
    "backand": "backend",
    "fronend": "frontend",
    "frontend": "frontend",
    "educaton": "education",
    "qualification": "qualification",
    "qualifications": "qualifications",
    "abount": "about",
    "informtion": "information",
    "informations": "information",
    "projet": "project",
    "projets": "projects",
    "contat": "contact",
    "contanct": "contact",
    "resume": "resume",
    "resumee": "resume",
}

VASANTH_IMPLIED_PHRASES = [
    "total experience", "work experience", "experience", "years of experience",
    "skills", "what can", "what does", "education", "qualification", "projects",
    "background", "expertise", "technologies", "contact", "email", "linkedin",
    "resume", "cv", "hire", "employ", "about him", "tell me", "who is",
    "how long", "how many years", "where did", "where does", "what is the",
    "what are the", "which", "certification", "certifications",
]


def normalize_query(raw_query: str) -> Tuple[str, str]:
    if not raw_query or not raw_query.strip():
        return raw_query, raw_query

    query = raw_query.strip()
    query_lower = query.lower()

    for typo, correction in sorted(TYPO_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(typo), re.IGNORECASE)
        query = pattern.sub(correction, query)
        query_lower = query.lower()

    return query.strip(), raw_query.strip()


def should_inject_vasanth_context(query: str) -> bool:
    q = query.lower().strip()
    if "vasanth" in q or "vasanthkumar" in q:
        return False
    return any(phrase in q for phrase in VASANTH_IMPLIED_PHRASES)


def expand_query_for_vasanth(query: str) -> str:
    if not should_inject_vasanth_context(query):
        return query
    return f"Vasanth {query}".strip()
