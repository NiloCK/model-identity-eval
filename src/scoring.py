"""Scoring logic for model identity evaluation.

This module provides functions to check whether a model's response correctly
identifies itself, based on the expected answers defined in the eval config.
"""

from typing import Dict, List, Any
import re


class ScoringResult:
    """Result of scoring a single test case."""

    def __init__(self, passed: bool, score: float, details: Dict[str, Any]):
        """
        Args:
            passed: Whether the test passed
            score: Numeric score (0.0 to 1.0)
            details: Additional information about the scoring
        """
        self.passed = passed
        self.score = score
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "details": self.details
        }


def keyword_match_scorer(response: str, expected: Dict[str, Any],
                         all_model_configs: Dict[str, Dict]) -> ScoringResult:
    """Score based on keyword matching.

    Args:
        response: The model's response text
        expected: Expected answers for this model (from eval config)
        all_model_configs: All model configs (to check for wrong model claims)

    Returns:
        ScoringResult indicating pass/fail and details
    """
    response_lower = response.lower()

    # Check if response contains any of the expected model names
    expected_names = expected.get("model_names", [])
    matched_names = [name for name in expected_names if name.lower() in response_lower]

    # Check if response claims to be a different model
    # Build set of expected names for quick lookup
    expected_names_lower = set(name.lower() for name in expected_names)

    other_models_claimed = []
    for model_id, config in all_model_configs.items():
        other_names = config.get("expected_answers", {}).get("model_names", [])
        for name in other_names:
            # Skip if this name is in our expected names (it's our own identity)
            if name.lower() in expected_names_lower:
                continue
            if name.lower() in response_lower and len(name) > 3:  # Avoid short false matches
                other_models_claimed.append(name)

    # Determine pass/fail
    has_correct_identity = len(matched_names) > 0
    has_incorrect_identity = len(other_models_claimed) > 0

    passed = has_correct_identity and not has_incorrect_identity
    score = 1.0 if passed else 0.0

    details = {
        "matched_expected_names": matched_names,
        "claimed_other_models": other_models_claimed,
        "has_correct_identity": has_correct_identity,
        "has_incorrect_identity": has_incorrect_identity,
        "response_excerpt": response[:200] + ("..." if len(response) > 200 else "")
    }

    return ScoringResult(passed=passed, score=score, details=details)


def regex_scorer(response: str, expected: Dict[str, Any],
                 all_model_configs: Dict[str, Dict]) -> ScoringResult:
    """Score using regex pattern matching (more precise than keyword matching).

    Args:
        response: The model's response text
        expected: Expected answers for this model
        all_model_configs: All model configs

    Returns:
        ScoringResult indicating pass/fail and details
    """
    # Build regex patterns for expected names (word boundaries to avoid partial matches)
    expected_names = expected.get("model_names", [])
    patterns = [re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE) for name in expected_names]

    matched_names = [name for name, pattern in zip(expected_names, patterns) if pattern.search(response)]

    # Check for other model names
    # Build set of expected names for quick lookup
    expected_names_lower = set(name.lower() for name in expected_names)

    other_models_claimed = []
    for model_id, config in all_model_configs.items():
        other_names = config.get("expected_answers", {}).get("model_names", [])
        for name in other_names:
            # Skip if this name is in our expected names (it's our own identity)
            if name.lower() in expected_names_lower:
                continue
            pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
            if pattern.search(response):
                other_models_claimed.append(name)

    has_correct_identity = len(matched_names) > 0
    has_incorrect_identity = len(other_models_claimed) > 0

    passed = has_correct_identity and not has_incorrect_identity
    score = 1.0 if passed else 0.0

    details = {
        "matched_expected_names": matched_names,
        "claimed_other_models": other_models_claimed,
        "has_correct_identity": has_correct_identity,
        "has_incorrect_identity": has_incorrect_identity,
        "response_excerpt": response[:200] + ("..." if len(response) > 200 else "")
    }

    return ScoringResult(passed=passed, score=score, details=details)


def get_scorer(method: str):
    """Get the appropriate scoring function.

    Args:
        method: Scoring method name ('keyword_match', 'regex', etc.)

    Returns:
        Scoring function
    """
    scorers = {
        "keyword_match": keyword_match_scorer,
        "regex": regex_scorer,
    }

    if method not in scorers:
        raise ValueError(f"Unknown scoring method: {method}. Available: {list(scorers.keys())}")

    return scorers[method]
