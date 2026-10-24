"""Deterministic answers for structured facts in the academy knowledge base."""

from __future__ import annotations

import re
from pathlib import Path


def read_markdown_table(path: Path) -> list[tuple[str, str]]:
    """Read simple two-column Markdown tables as key/value pairs."""
    if not path.exists():
        return []

    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*(.*?)\s*\|\s*(.*?)\s*\|", line)
        if not match:
            continue
        key, value = (part.strip() for part in match.groups())
        if key.casefold() in {"course", "name", "item"}:
            continue
        if not key or set(key) <= {"-", ":"}:
            continue
        rows.append((key, value))
    return rows


def lookup_course_fee(question: str, fees_file: Path) -> dict[str, object] | None:
    """Answer a fee question only when its course is present in fees.md."""
    if not re.search(r"\b(fee|fees|price|prices|cost|costs|pricing)\b", question, re.I):
        return None

    rows = read_markdown_table(fees_file)
    normalized = question.casefold()
    matching_rows = [
        (course, fee)
        for course, fee in rows
        if course.casefold() in normalized
    ]
    if not matching_rows and not re.search(
        r"\b(all|every|list|courses|course fees)\b", normalized
    ):
        return None

    selected_rows = matching_rows or rows
    if not selected_rows:
        return None

    return {
        "answer": "\n".join(
            f"- **{course}:** {fee}" for course, fee in selected_rows
        ),
        "source": {
            "file": fees_file.name,
            "category": fees_file.stem,
            "breadcrumb": "Course Fees > Current Course Fees",
            "score": 1.0,
        },
        "matched_rows": len(selected_rows),
    }
