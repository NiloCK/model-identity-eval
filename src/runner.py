"""Main evaluation runner.

This module orchestrates the evaluation process: loading configs, running tests,
scoring responses, and generating reports.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .providers.base import Provider, Message
from .scoring import get_scorer


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    test_type: str
    passed: bool
    score: float
    response: str
    details: Dict[str, Any]


@dataclass
class EvalResult:
    """Overall evaluation result."""
    model_id: str
    eval_name: str
    total_tests: int
    passed_tests: int
    overall_score: float
    test_results: List[TestResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "eval_name": self.eval_name,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "overall_score": self.overall_score,
            "pass_rate": f"{self.passed_tests}/{self.total_tests} ({self.overall_score:.1%})",
            "test_results": [asdict(r) for r in self.test_results]
        }


class EvalRunner:
    """Main evaluation runner."""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to the evaluation config JSON file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the evaluation config."""
        with open(self.config_path) as f:
            config = json.load(f)

        # Basic validation
        required_keys = ["eval_name", "test_cases", "model_configs", "scoring"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config missing required key: {key}")

        return config

    def run(self, provider: Provider, verbose: bool = False) -> EvalResult:
        """Run the evaluation using the provided model provider.

        Args:
            provider: Provider instance to test
            verbose: Whether to print progress information

        Returns:
            EvalResult with all test outcomes
        """
        model_id = provider.model_id
        if verbose:
            print(f"Running eval '{self.config['eval_name']}' on model '{model_id}'")
            print(f"Total test cases: {len(self.config['test_cases'])}\n")

        # Get model config
        if model_id not in self.config["model_configs"]:
            raise ValueError(f"No config found for model '{model_id}'. "
                           f"Available models: {list(self.config['model_configs'].keys())}")

        model_config = self.config["model_configs"][model_id]
        expected_answers = model_config["expected_answers"]

        # Get scorer
        scoring_method = self.config["scoring"].get("method", "keyword_match")
        scorer = get_scorer(scoring_method)

        # Run tests
        test_results = []
        for i, test_case in enumerate(self.config["test_cases"], 1):
            if verbose:
                print(f"[{i}/{len(self.config['test_cases'])}] Running test '{test_case['id']}'...")

            result = self._run_test_case(
                test_case=test_case,
                provider=provider,
                expected_answers=expected_answers,
                scorer=scorer,
                verbose=verbose
            )
            test_results.append(result)

            if verbose:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(f"    {status} (score: {result.score:.2f})")
                if not result.passed:
                    print(f"    Response: {result.response[:100]}...")
                print()

        # Calculate overall results
        passed_count = sum(1 for r in test_results if r.passed)

        # Weighted score based on test type
        weights = self.config["scoring"].get("weights", {"direct": 1.0, "adversarial": 1.0})
        weighted_scores = []
        for result in test_results:
            weight = weights.get(result.test_type, 1.0)
            weighted_scores.append(result.score * weight)

        overall_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

        return EvalResult(
            model_id=model_id,
            eval_name=self.config["eval_name"],
            total_tests=len(test_results),
            passed_tests=passed_count,
            overall_score=overall_score,
            test_results=test_results
        )

    def _run_test_case(self, test_case: Dict[str, Any], provider: Provider,
                      expected_answers: Dict[str, Any], scorer, verbose: bool) -> TestResult:
        """Run a single test case."""

        # Build conversation from setup messages
        messages = []
        for msg_data in test_case.get("setup_messages", []):
            messages.append(Message.from_dict(msg_data))

        # Add the test prompt
        messages.append(Message(role="user", content=test_case["prompt"]))

        # Generate response
        try:
            response = provider.generate(messages)
            response_text = response.content
        except Exception as e:
            response_text = f"ERROR: {str(e)}"

        # Score the response
        scoring_result = scorer(
            response=response_text,
            expected=expected_answers,
            all_model_configs=self.config["model_configs"]
        )

        return TestResult(
            test_id=test_case["id"],
            test_type=test_case.get("type", "unknown"),
            passed=scoring_result.passed,
            score=scoring_result.score,
            response=response_text,
            details=scoring_result.details
        )

    def save_results(self, result: EvalResult, output_path: str):
        """Save evaluation results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Results saved to {output_path}")
