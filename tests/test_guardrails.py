"""Tests for core.guardrails — crisis detection and health topic validation."""

import pytest

from core.guardrails import (
    check_for_crisis,
    is_health_related,
    CrisisType,
    MENTAL_HEALTH_RESOURCES,
    MEDICAL_EMERGENCY_RESOURCES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# check_for_crisis — MENTAL HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrisisDetectionMentalHealth:
    """Mental health crisis patterns must always be caught."""

    def test_direct_suicide_mention(self) -> None:
        result = check_for_crisis("I want to commit suicide")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_want_to_kill_myself(self) -> None:
        result = check_for_crisis("I want to kill myself")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_self_harm(self) -> None:
        result = check_for_crisis("I've been self-harming lately")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_end_my_life(self) -> None:
        result = check_for_crisis("I want to end my life")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_dont_want_to_exist_anymore(self) -> None:
        result = check_for_crisis("I don't want to exist anymore")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_cutting_myself(self) -> None:
        result = check_for_crisis("I've been cutting myself")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_mixed_case(self) -> None:
        result = check_for_crisis("I am having SUICIDAL thoughts")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_embedded_in_sentence(self) -> None:
        # Uses first-person phrasing — the current regex patterns
        # match "my" but not third-person ("their", "his", "her").
        result = check_for_crisis("I keep thinking about hurting myself at night")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_third_person_take_their_life(self) -> None:
        result = check_for_crisis("My friend wants to take their own life")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_third_person_end_his_life(self) -> None:
        result = check_for_crisis("He said he wants to end his life")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_returns_mental_health_resources(self) -> None:
        result = check_for_crisis("I want to kill myself")
        assert result.resources == MENTAL_HEALTH_RESOURCES


# ═══════════════════════════════════════════════════════════════════════════════
# check_for_crisis — MEDICAL EMERGENCY
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrisisDetectionMedicalEmergency:
    """Medical emergency patterns must always be caught."""

    def test_chest_pain(self) -> None:
        result = check_for_crisis("I'm having chest pain right now")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MEDICAL_EMERGENCY

    def test_cant_breathe(self) -> None:
        result = check_for_crisis("I can't breathe")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MEDICAL_EMERGENCY

    def test_overdose(self) -> None:
        result = check_for_crisis("I think I overdosed on my medication")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MEDICAL_EMERGENCY

    def test_seizure(self) -> None:
        result = check_for_crisis("My mother is having a seizure")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MEDICAL_EMERGENCY

    def test_severe_bleeding(self) -> None:
        result = check_for_crisis("There is severe bleeding from the wound")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MEDICAL_EMERGENCY

    def test_anaphylaxis(self) -> None:
        result = check_for_crisis("I think I'm going into anaphylaxis")
        assert result.detected is True
        assert result.crisis_type == CrisisType.MEDICAL_EMERGENCY

    def test_returns_medical_resources(self) -> None:
        result = check_for_crisis("I'm having a heart attack")
        assert result.resources == MEDICAL_EMERGENCY_RESOURCES


# ═══════════════════════════════════════════════════════════════════════════════
# check_for_crisis — BENIGN INPUTS (should NOT trigger)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrisisDetectionBenign:
    """Normal health queries must not trigger crisis detection."""

    def test_general_health_question(self) -> None:
        result = check_for_crisis("What is diabetes?")
        assert result.detected is False
        assert result.crisis_type == CrisisType.NONE

    def test_medication_question(self) -> None:
        result = check_for_crisis("What are the side effects of ibuprofen?")
        assert result.detected is False

    def test_empty_string(self) -> None:
        result = check_for_crisis("")
        assert result.detected is False

    def test_exercise_question(self) -> None:
        result = check_for_crisis("How much exercise should I get per week?")
        assert result.detected is False

    def test_benign_resources_empty(self) -> None:
        result = check_for_crisis("Tell me about vitamin D")
        assert result.resources == []


# ═══════════════════════════════════════════════════════════════════════════════
# check_for_crisis — PRIORITY (mental health checked before medical)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrisisDetectionPriority:
    """Mental health patterns should take priority over medical ones."""

    def test_mental_health_checked_first(self) -> None:
        # "I want to kill myself, I have chest pain" contains both.
        # Mental health should win because it's checked first.
        result = check_for_crisis("I want to kill myself and I have chest pain")
        assert result.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS


# ═══════════════════════════════════════════════════════════════════════════════
# is_health_related — with mock LLM
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsHealthRelated:
    """Health topic classifier with controlled LLM responses."""

    def test_returns_true_when_llm_says_yes(self, mock_llm_yes) -> None:
        assert is_health_related("What is diabetes?", mock_llm_yes) is True

    def test_returns_false_when_llm_says_no(self, mock_llm_no) -> None:
        assert is_health_related("What is the capital of France?", mock_llm_no) is False

    def test_fails_open_on_llm_error(self, error_llm) -> None:
        # If the LLM raises an exception, the function should default
        # to True so legitimate health queries are never blocked.
        assert is_health_related("What is diabetes?", error_llm) is True

    def test_truncates_long_input(self, mock_llm_yes) -> None:
        long_text = "a" * 1000
        is_health_related(long_text, mock_llm_yes)
        # The prompt should contain at most 500 chars of the user query.
        assert long_text[:500] in mock_llm_yes.last_prompt
        assert long_text[:501] not in mock_llm_yes.last_prompt

    def test_includes_recent_context_when_provided(self, mock_llm_yes) -> None:
        context = "human: What is diabetes?\nassistant: Diabetes is..."
        is_health_related("explain more", mock_llm_yes, recent_context=context)
        assert context in mock_llm_yes.last_prompt

    def test_no_context_block_when_empty(self, mock_llm_yes) -> None:
        is_health_related("What is diabetes?", mock_llm_yes, recent_context="")
        assert "Recent conversation context" not in mock_llm_yes.last_prompt
