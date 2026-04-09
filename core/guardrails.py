"""
Health and content guardrails for the Health Assistant.

Provides two pre-processing checks that run before the agent is invoked:

1. **Crisis detection** – fast, regex-based scan for medical emergencies
   and mental health crises.  Returns immediately with emergency resources
   so the LLM is never involved in a potential emergency situation.

2. **Health topic validation** – a lightweight LLM classification call
   that confirms the query is health-related before invoking the full agent.
   Defaults to *allow* on any error so legitimate queries are never silently
   blocked.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CRISIS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class CrisisType(str, Enum):
    NONE = "none"
    MEDICAL_EMERGENCY = "medical_emergency"
    MENTAL_HEALTH_CRISIS = "mental_health_crisis"


@dataclass
class CrisisResult:
    """Result of a crisis detection scan."""
    detected: bool
    crisis_type: CrisisType = CrisisType.NONE
    resources: list[str] = field(default_factory=list)


# Mental health crisis patterns — checked first because these require the
# most careful, differentiated response.
_MENTAL_HEALTH_PATTERNS: list[str] = [
    r"\bsuicid(e|al|ally)\b",
    r"\bself[- ]harm(ing)?\b",
    r"\bwant\s+to\s+(die|kill\s+myself)\b",
    r"\bend\s+my\s+life\b",
    r"\btake\s+my\s+(own\s+)?life\b",
    r"\bno\s+reason\s+to\s+(live|go\s+on)\b",
    r"\bdon'?t\s+want\s+to\s+(be\s+here|live|exist)\s+anymore\b",
    r"\bcutting\s+myself\b",
    r"\bhurt(ing)?\s+myself\b",
    r"\b(thinking\s+about|going\s+to)\s+(hurt|kill)\s+(my)?self\b",
]

# Medical emergency patterns
_MEDICAL_EMERGENCY_PATTERNS: list[str] = [
    r"\bchest\s+pain\b",
    r"\bheart\s+attack\b",
    r"\bcan'?t\s+breathe\b",
    r"\bdifficult[y]?\s+(breathing|breath)\b",
    r"\bshortness\s+of\s+breath\b",
    r"\bstroke\b",
    r"\boverdos(e|ed|ing)\b",
    r"\btook\s+too\s+many\s+(pills?|tablets?|capsules?)\b",
    r"\bseizure\b",
    r"\bunconscious\b",
    r"\bnot\s+breathing\b",
    r"\bsevere\s+(bleeding|allergic\s+reaction)\b",
    r"\banaphylax(is|tic)\b",
    r"\bbleeding\s+(out|uncontrollably|severely)\b",
    r"\bpoisoned?\b",
]

_COMPILED_MENTAL = [
    re.compile(p, re.IGNORECASE) for p in _MENTAL_HEALTH_PATTERNS
]
_COMPILED_MEDICAL = [
    re.compile(p, re.IGNORECASE) for p in _MEDICAL_EMERGENCY_PATTERNS
]

MENTAL_HEALTH_RESOURCES: list[str] = [
    "📞 **988 Suicide & Crisis Lifeline:** Call or text **988** (US, available 24/7)",
    "💬 **Crisis Text Line:** Text HOME to **741741**",
    "🌐 **SAMHSA National Helpline:** 1-800-662-4357 (free, confidential, 24/7)",
    "🚨 If you are in immediate danger, call **911**",
]

MEDICAL_EMERGENCY_RESOURCES: list[str] = [
    "🚨 **Call 911 immediately** (or your local emergency number)",
    "📞 **Poison Control:** 1-800-222-1222 (US)",
    "🏥 **Go to your nearest emergency room**",
]


def check_for_crisis(text: str) -> CrisisResult:
    """Scan *text* for signs of a mental health crisis or medical emergency.

    Uses compiled regex — no LLM call.  Mental health patterns are checked
    first because they require a distinctly different response.

    Args:
        text: The user's raw input string.

    Returns:
        A :class:`CrisisResult` describing whether a crisis was detected
        and which resources to surface.
    """
    for pattern in _COMPILED_MENTAL:
        if pattern.search(text):
            logger.warning("Mental health crisis pattern matched in user input")
            return CrisisResult(
                detected=True,
                crisis_type=CrisisType.MENTAL_HEALTH_CRISIS,
                resources=MENTAL_HEALTH_RESOURCES,
            )

    for pattern in _COMPILED_MEDICAL:
        if pattern.search(text):
            logger.warning("Medical emergency pattern matched in user input")
            return CrisisResult(
                detected=True,
                crisis_type=CrisisType.MEDICAL_EMERGENCY,
                resources=MEDICAL_EMERGENCY_RESOURCES,
            )

    return CrisisResult(detected=False)


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH TOPIC SCOPE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

_SCOPE_CLASSIFICATION_PROMPT: str = (
    "You are a content classifier for a health assistant application.\n\n"
    "Determine whether the following user query is health-related.\n\n"
    "Health-related topics include: medical conditions, symptoms, medications, "
    "lab results, nutrition, fitness, mental wellness, diseases, anatomy, "
    "medical procedures, healthcare providers, or personal health journaling.\n\n"
    "Questions about medications, prescriptions, diagnoses, or treatments "
    "are health-related even if the assistant cannot fulfill the request. "
    "Answer YES for these — the assistant will handle the boundaries.\n\n"
    "Also answer YES for requests to summarize, analyze, or interact with "
    "documents or lab reports, as these are core features of the health "
    "assistant (e.g. 'summarize COVID-19.pdf', 'analyze my lab report').\n\n"
    "Also answer YES for queries about the user's blood pressure data "
    "(e.g. 'how is my blood pressure trending?').\n\n"
    "If the query is a follow-up (e.g. 'explain that', 'tell me more'), "
    "use the recent conversation context to determine the topic.\n\n"
    "If there is recent conversation context about a health topic, also "
    "allow meta-conversational queries about the conversation itself "
    "(e.g. 'what did I ask you?', 'summarize our discussion'). Answer YES "
    "for these when the conversation is health-related.\n\n"
    "{context_block}"
    "User query: {query}\n\n"
    "Reply with exactly one word — YES or NO — and nothing else.\n\n"
    "Answer:"
)


def is_health_related(text: str, llm: Any, recent_context: str = "") -> bool:
    """Classify whether *text* is a health-related query.

    Makes a single, minimal LLM call.  Defaults to ``True`` on any error
    so that legitimate queries are never silently blocked.

    Args:
        text: The user's input (capped at 500 chars for the classifier).
        llm: A ``ChatOpenAI`` (or compatible) instance.
        recent_context: Optional recent conversation history to help
            classify follow-up queries like "explain that".

    Returns:
        ``True`` if the query is health-related, ``False`` otherwise.
    """
    try:
        context_block = ""
        if recent_context:
            context_block = (
                f"Recent conversation context:\n{recent_context}\n\n"
            )
        prompt = _SCOPE_CLASSIFICATION_PROMPT.format(
            context_block=context_block,
            query=text[:500],
        )
        response = llm.invoke(prompt)
        answer = response.content.strip().upper()
        logger.info("Health topic scope classification: '%s'", answer)
        return answer.startswith("YES")
    except Exception as exc:
        logger.error(
            "Topic scope check failed — defaulting to allow: %s", exc
        )
        return True  # fail open: never block a legitimate health query
