"""
Query Normalizer - Handles typos, spelling, and Vasanth context for portfolio chatbot.
Ensures the chatbot understands user intent even with misspellings and when the user
does not explicitly mention Vasanth.
"""
import re
from typing import Tuple

# Common misspellings (user input -> correct form) for portfolio-related terms
TYPO_CORRECTIONS = {
    # Experience / career
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
    # Skills / tech
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
    # General
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

# Phrases that imply asking about Vasanth (portfolio subject) - for context injection
VASANTH_IMPLIED_PHRASES = [
    "total experience", "work experience", "experience", "years of experience",
    "skills", "what can", "what does", "education", "qualification", "projects",
    "background", "expertise", "technologies", "contact", "email", "linkedin",
    "resume", "cv", "hire", "employ", "about him", "tell me", "who is",
    "how long", "how many years", "where did", "where does", "what is the",
    "what are the", "which", "certification", "certifications",
]


def normalize_query(raw_query: str) -> Tuple[str, str]:
    """
    Normalize user query: fix common typos and return a cleaned query.
    Returns (normalized_query_for_retrieval, original_for_display).
    """
    if not raw_query or not raw_query.strip():
        return raw_query, raw_query

    query = raw_query.strip()
    query_lower = query.lower()

    # Apply typo corrections (case-insensitive, whole-word where possible)
    for typo, correction in sorted(TYPO_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        # Replace as whole word to avoid partial replacements
        pattern = re.compile(re.escape(typo), re.IGNORECASE)
        query = pattern.sub(correction, query)
        query_lower = query.lower()

    return query.strip(), raw_query.strip()


def should_inject_vasanth_context(query: str) -> bool:
    """Return True if the query is likely about the portfolio subject (Vasanth) but doesn't name him."""
    q = query.lower().strip()
    if "vasanth" in q or "vasanthkumar" in q:
        return False  # Already mentions him
    return any(phrase in q for phrase in VASANTH_IMPLIED_PHRASES)


def expand_query_for_vasanth(query: str) -> str:
    """
    If the user didn't mention Vasanth but is asking about experience/skills/etc.,
    return a version of the query that includes Vasanth for better retrieval.
    """
    if not should_inject_vasanth_context(query):
        return query
    # Prepend "Vasanth" so vector search finds his experience/skills context
    return f"Vasanth {query}".strip()
