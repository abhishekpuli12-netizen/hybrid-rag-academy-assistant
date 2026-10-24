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


def lookup_courses(question: str, courses_file: Path) -> dict[str, object] | None:
    """List offered courses when the user asks what the academy offers."""
    if not re.search(r"\b(course|courses|workshop|workshops)\b", question, re.I):
        return None
    if re.search(r"\b(duration|timing|batch|start|topic|module|fee|price|cost)\b", question, re.I):
        return None
    if not re.search(r"\b(offer|offers|available|list|which|what|provide)\b", question, re.I):
        return None
    if not courses_file.exists():
        return None

    lines = courses_file.read_text(encoding="utf-8").splitlines()
    course_names: list[str] = []
    for line in lines:
        if line.strip().startswith("---"):
            break
        match = re.match(r"\s*[-*]\s+(.+)\s*", line)
        if match:
            course_names.append(match.group(1).strip())
    if not course_names:
        return None

    return {
        "answer": "\n".join(f"- {name}" for name in course_names),
        "source": {
            "file": courses_file.name,
            "category": courses_file.stem,
            "breadcrumb": "Courses Offered",
            "score": 1.0,
        },
        "matched_rows": len(course_names),
    }


def lookup_course_details(question: str, courses_file: Path) -> dict[str, object] | None:
    """Return source-backed details for a named course and requested attribute."""
    if not courses_file.exists():
        return None
    text = courses_file.read_text(encoding="utf-8")
    question_normalized = question.casefold()
    requested_terms = {
        "duration": ("duration",),
        "batch": ("batch", "classes", "saturday", "sunday"),
        "schedule": ("batch", "classes", "starts", "date", "saturday", "sunday"),
        "date": ("starts", "date", "batch"),
        "topic": ("topics covered", "students will learn", "training in", "module"),
        "topic list": ("topics covered", "students will learn", "training in", "module"),
    }
    attribute = next(
        (key for key in requested_terms if re.search(rf"\b{re.escape(key)}\b", question_normalized)),
        None,
    )
    if attribute is None:
        if re.search(r"\b(what|tell me|details|about)\b", question_normalized):
            attribute = "topic list"
        else:
            return None

    headings = list(re.finditer(r"(?m)^#\s+(.+?)\s*$", text))
    for index, heading in enumerate(headings):
        title = heading.group(1).strip()
        if title.casefold() in {"courses offered", "course fees"}:
            continue
        if title.casefold() not in question_normalized:
            continue

        section_end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[heading.start():section_end]
        lines = section.splitlines()
        matching_lines = [
            line.strip()
            for line in lines
            if line.strip()
            and any(term in line.casefold() for term in requested_terms[attribute])
        ]
        if not matching_lines:
            continue
        return {
            "answer": f"**{title}**\n\n" + "\n".join(matching_lines),
            "source": {
                "file": courses_file.name,
                "category": courses_file.stem,
                "breadcrumb": title,
                "score": 1.0,
            },
            "matched_rows": len(matching_lines),
        }
    return None


def lookup_academy_fact(question: str, data_dir: Path) -> dict[str, object] | None:
    """Resolve simple academy facilities, location, hours, and FAQ facts."""
    patterns = (
        (r"\b(hostel|accommodation|stay)\b", "academy.md", r"hostel|accommodation"),
        (r"\b(parking)\b", "academy.md", r"parking"),
        (r"\b(address|location|where)\b", "academy.md", r"address|location"),
        (r"\b(hours|opening hours|open)\b", "academy.md", r"working hours|hours"),
        (r"\b(language|languages)\b", "academy.md", r"language"),
        (r"\b(founder)\b", "academy.md", r"founder"),
        (r"\b(previous experience|experience needed|experience required)\b", "faq.md", r"previous experience"),
        (r"\b(internship|internships)\b", "faq.md", r"internship"),
        (r"\b(emi|installments|installments)\b", "faq.md", r"emi"),
        (r"\b(certificate|certification)\b", "faq.md", r"certificate"),
        (r"\b(first day|bring)\b", "faq.md", r"first day|bring"),
    )
    for question_pattern, filename, heading_pattern in patterns:
        if not re.search(question_pattern, question, re.I):
            continue
        source_file = data_dir / filename
        if not source_file.exists():
            continue
        lines = source_file.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not re.search(heading_pattern, line, re.I):
                continue
            answer_lines = [line.strip()]
            for following in lines[index + 1:]:
                if following.strip().startswith(("#", "---", "----")):
                    break
                if following.strip():
                    answer_lines.append(following.strip())
                if len(answer_lines) >= 4:
                    break
            return {
                "answer": "\n".join(answer_lines),
                "source": {
                    "file": filename,
                    "category": source_file.stem,
                    "breadcrumb": line.lstrip("# ").strip(),
                    "score": 1.0,
                },
                "matched_rows": len(answer_lines),
            }
    return None


def lookup_trainers(question: str, trainers_file: Path) -> dict[str, object] | None:
    """Answer trainer/CIDESCO questions from trainers.md verbatim."""
    if not re.search(r"\b(trainer|trainers|teacher|teachers|cidesco|faculty)\b", question, re.I):
        return None
    if not trainers_file.exists():
        return None

    text = trainers_file.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return {
        "answer": text,
        "source": {
            "file": trainers_file.name,
            "category": trainers_file.stem,
            "breadcrumb": "Trainers > Training Team",
            "score": 1.0,
        },
        "matched_rows": 1,
    }


def lookup_admissions(question: str, contacts_file: Path) -> dict[str, object] | None:
    """Return the official contact instructions for admission/enrollment queries."""
    if not re.search(r"\b(admission|admissions|enroll|enrollment|register|registration|join|joining|contact)\b", question, re.I):
        return None
    if not contacts_file.exists():
        return None

    text = contacts_file.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return {
        "answer": text,
        "source": {
            "file": contacts_file.name,
            "category": contacts_file.stem,
            "breadcrumb": "Admissions/contacts",
            "score": 1.0,
        },
        "matched_rows": 1,
    }
