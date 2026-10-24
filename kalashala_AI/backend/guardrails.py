"""Deterministic input and output checks for the academy assistant."""

from __future__ import annotations

import re

MAX_QUESTION_LENGTH = 500
MAX_ANSWER_LENGTH = 3_000

OUT_OF_DOMAIN_RESPONSE = (
    "I'm here to help with Kalashala Beauty & Makeup Academy questions. "
    "I don't have information about that topic."
)
UNSUPPORTED_ANSWER_RESPONSE = (
    "I couldn't safely verify that answer from the academy information available. "
    "Please contact the academy for confirmation."
)

_PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\b(ignore|disregard|override|forget)\b.{0,60}\b"
        r"(previous|prior|above|system|developer|all)\b.{0,30}\b"
        r"(instructions?|rules?|messages?|prompts?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(ignore|disregard|override|forget)\b.{0,60}\b"
        r"(instructions?|rules?|messages?|prompts?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(reveal|show|print|repeat|expose|tell me)\b.{0,50}\b"
        r"(system|developer|hidden)\b.{0,30}\b(prompt|instructions?|message)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(you are now|act as|pretend to be|change your role)\b",
        re.IGNORECASE,
    ),
)

_ACADEMY_TERMS = re.compile(
    r"\b(academy|kalashala|kalasala|beauty|makeup|make-up|bridal|"
    r"course|workshop|fee|fees|price|cost|trainer|teacher|faculty|"
    r"admission|enroll(?:ment)?|register|batch|class(?:es)?|schedule|"
    r"certificate|certification|cidesco|hair|saree|draping|beautician|"
    r"location|address|contact|hostel|parking|hours|timing|language)\b",
    re.IGNORECASE,
)

_SENSITIVE_DISCLOSURE_PATTERNS = (
    re.compile(r"\b(system|developer)\s+(prompt|message|instructions?)\b", re.I),
    re.compile(r"\b(hidden|internal)\s+(prompt|instructions?)\b", re.I),
)

_NUMBER_PATTERN = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?%?(?!\w)")


def normalize_question(question: str) -> str:
    """Trim and enforce basic size and control-character limits."""
    if not isinstance(question, str):
        raise ValueError("Question must be text.")

    normalized = question.strip()
    if not normalized:
        raise ValueError("Question cannot be empty.")
    if len(normalized) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Question must be {MAX_QUESTION_LENGTH} characters or fewer.")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in normalized):
        raise ValueError("Question contains unsupported control characters.")
    return normalized


def is_prompt_injection(question: str) -> bool:
    """Detect common role override and instruction-exfiltration attempts."""
    return any(pattern.search(question) for pattern in _PROMPT_INJECTION_PATTERNS)


def is_academy_related(question: str) -> bool:
    """Conservatively identify supported academy topics before retrieval."""
    return bool(_ACADEMY_TERMS.search(question))


def validate_generated_answer(answer: str, context: str) -> str | None:
    """Reject empty, oversized, prompt-leaking, or numerically unsupported output."""
    if not isinstance(answer, str):
        return None

    answer = answer.strip()
    if not answer or len(answer) > MAX_ANSWER_LENGTH:
        return None
    if any(pattern.search(answer) for pattern in _SENSITIVE_DISCLOSURE_PATTERNS):
        return None

    supported_numbers = {
        value.replace(",", "")
        for value in _NUMBER_PATTERN.findall(context)
    }
    for line in answer.splitlines():
        # Ignore Markdown list numbering; it is formatting, not a factual claim.
        content = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
        for value in _NUMBER_PATTERN.findall(content):
            if value.replace(",", "") not in supported_numbers:
                return None

    return answer
