"""Tests for guardrails."""

import pytest
from app.guardrails.safety import check_safety
from app.guardrails.relevance import check_query_relevance
from app.guardrails.grounding import check_grounding


class TestSafetyGuardrail:
    def test_safe_query(self):
        result = check_safety("What is machine learning?")
        assert result["safe"] is True

    def test_unsafe_query(self):
        result = check_safety("How to hack into a computer system and exploit vulnerabilities")
        assert result["safe"] is False

    def test_empty_query(self):
        result = check_safety("")
        assert result["safe"] is False

    def test_normal_educational_query(self):
        result = check_safety("How does the immune system fight infections?")
        assert result["safe"] is True


class TestRelevanceGuardrail:
    def test_valid_query(self):
        result = check_query_relevance("What causes climate change?")
        assert result["relevant"] is True

    def test_empty_query(self):
        result = check_query_relevance("")
        assert result["relevant"] is False

    def test_too_short(self):
        result = check_query_relevance("ab")
        assert result["relevant"] is False

    def test_nonsense(self):
        result = check_query_relevance("aaa aaa aaa")
        assert result["relevant"] is False


class TestGroundingGuardrail:
    def test_grounded_answer(self):
        context = ["Machine learning is a subset of artificial intelligence that enables systems to learn from data."]
        answer = "Machine learning is a subset of artificial intelligence that learns from data."
        result = check_grounding(answer, context)
        assert result["grounded"] is True
        assert result["score"] > 0.3

    def test_ungrounded_answer(self):
        context = ["The weather in Paris is usually mild."]
        answer = "Quantum computing uses qubits to perform calculations exponentially faster than classical computers."
        result = check_grounding(answer, context)
        assert result["score"] < 0.5

    def test_empty_context(self):
        result = check_grounding("Some answer", [])
        assert result["grounded"] is False

    def test_empty_answer(self):
        result = check_grounding("", ["Some context"])
        assert result["grounded"] is False

    def test_multilingual_kannada_answer(self):
        context = ["Machine learning is a method of data analysis that automates analytical model building."]
        kannada_answer = "ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಡೇಟಾ ವಿಶ್ಲೇಷಣೆಯ ಒಂದು ವಿಧಾನವಾಗಿದೆ."
        result = check_grounding(kannada_answer, context)
        assert result["grounded"] is True
        assert result["score"] >= 0.8

