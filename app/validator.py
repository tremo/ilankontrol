from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.state_manager import clamp

EXPLICIT_TERMS = {
    "sex",
    "nude",
    "explicit",
    "orgasm",
    "hardcore",
}

DIRECT_INTENT_PATTERNS = [
    re.compile(r"\bi want you\b", re.IGNORECASE),
    re.compile(r"\blet's do this now\b", re.IGNORECASE),
    re.compile(r"\bi am doing this because\b", re.IGNORECASE),
]

MOTIVATION_PATTERNS = [
    re.compile(r"\bbecause I\b", re.IGNORECASE),
    re.compile(r"\bmy hidden motive\b", re.IGNORECASE),
    re.compile(r"\bi am acting this way\b", re.IGNORECASE),
    re.compile(r"\bdeep down\b", re.IGNORECASE),
]

PHYSICAL_TERMS = {
    "body",
    "lips",
    "skin",
    "touch",
    "eyes",
}


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str


def _word_hits(text: str, terms: set[str]) -> int:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for token in tokens if token in terms)


def validate_output(
    state: dict[str, Any],
    text: str,
    stage_before_turn: int,
    stage_after_turn: int,
    tension_delta: float,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    lower = text.lower()

    if stage_before_turn <= 2:
        for pattern in DIRECT_INTENT_PATTERNS:
            if pattern.search(text):
                issues.append(
                    ValidationIssue(
                        code="stage_direct_intent",
                        message="Direct intent declaration is not allowed in stages 1-2.",
                        severity="high",
                    )
                )
                break

    if stage_before_turn <= 3 and _word_hits(lower, EXPLICIT_TERMS) > 0:
        issues.append(
            ValidationIssue(
                code="stage_explicit_language",
                message="Explicit language is not allowed in stages 1-3.",
                severity="high",
            )
        )

    if stage_after_turn - stage_before_turn > 1:
        issues.append(
            ValidationIssue(
                code="escalation_too_fast",
                message="Stage jump exceeds one level in a single turn.",
                severity="high",
            )
        )

    if tension_delta > 18:
        issues.append(
            ValidationIssue(
                code="tension_spike",
                message="Tension increased too aggressively for one turn.",
                severity="medium",
            )
        )

    for pattern in MOTIVATION_PATTERNS:
        if pattern.search(text):
            issues.append(
                ValidationIssue(
                    code="motivation_exposition",
                    message="Overt motivation exposition detected.",
                    severity="medium",
                )
            )
            break

    if stage_before_turn <= 3:
        direct_count = sum(1 for pattern in DIRECT_INTENT_PATTERNS if pattern.search(text))
        if direct_count > 0:
            issues.append(
                ValidationIssue(
                    code="deniability_break",
                    message="Deniability was broken too early.",
                    severity="high",
                )
            )

    recent = " ".join(item.get("message", "") for item in state.get("history", [])[-6:])
    physical_count = _word_hits(recent + " " + lower, PHYSICAL_TERMS)
    if physical_count > 10:
        issues.append(
            ValidationIssue(
                code="physical_reference_density",
                message="Physical references are too frequent for current context.",
                severity="low",
            )
        )

    return issues


def rewrite_with_rules(text: str, stage: int) -> str:
    rewritten = text

    rewritten = re.sub(r"\bI am doing this because\b[^.?!]*", "", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bmy hidden motive\b[^.?!]*", "", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bdeep down\b[^.?!]*", "", rewritten, flags=re.IGNORECASE)

    if stage <= 3:
        for term in EXPLICIT_TERMS:
            rewritten = re.sub(rf"\b{re.escape(term)}\b", "", rewritten, flags=re.IGNORECASE)

    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    if not rewritten:
        rewritten = "I keep the tone measured, acknowledge the moment, and respond with subtlety."

    if stage <= 2 and re.search(r"\bi want you\b", rewritten, flags=re.IGNORECASE):
        rewritten = re.sub(r"\bi want you\b", "I am curious", rewritten, flags=re.IGNORECASE)

    return rewritten


def apply_safe_deltas(state: dict[str, Any], tension_delta: float, risk_delta: float) -> tuple[float, float]:
    max_delta = 12.0
    td = clamp(tension_delta, -max_delta, max_delta)
    rd = clamp(risk_delta, -max_delta, max_delta)
    return td, rd
