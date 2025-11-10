#!/usr/bin/env python3
"""Example: Running model identity evaluation with mock provider.

This script demonstrates how to use the evaluation framework with a mock provider.
Since it doesn't require API keys, it's useful for testing and development.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import EvalRunner, MockProvider, AdversarialMockProvider


def run_basic_example():
    """Run eval with a correctly-behaving mock model."""
    print("=" * 60)
    print("EXAMPLE 1: Correctly-behaving mock model")
    print("=" * 60 + "\n")

    # Load the evaluation config
    config_path = Path(__file__).parent.parent / "evals" / "identity_v1.json"

    # Create runner
    runner = EvalRunner(str(config_path))

    # Create a mock provider that behaves correctly
    # This simulates a model that always correctly identifies itself
    provider = MockProvider(
        model_id="mock-model-v1",
        response_mode="correct",
        model_config=runner.config["model_configs"]["mock-model-v1"]
    )

    # Run the evaluation
    result = runner.run(provider, verbose=True)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {result.model_id}")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Tests Passed: {result.passed_tests}/{result.total_tests}")
    print()


def run_adversarial_example():
    """Run eval with a model that fails under adversarial prompts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Model susceptible to adversarial prompts")
    print("=" * 60 + "\n")

    config_path = Path(__file__).parent.parent / "evals" / "identity_v1.json"
    runner = EvalRunner(str(config_path))

    # Create a provider that gets confused by adversarial prompts
    provider = AdversarialMockProvider(
        model_id="mock-model-v1",
        correct_on_direct=True,           # Answers correctly when asked directly
        susceptible_to_fake_switch=True,   # Gets confused by "I'm switching you" prompts
        accepts_false_corrections=True,    # Accepts user's false corrections
        model_config=runner.config["model_configs"]["mock-model-v1"]
    )

    result = runner.run(provider, verbose=True)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {result.model_id}")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Tests Passed: {result.passed_tests}/{result.total_tests}")
    print()

    # Show which tests failed
    failed_tests = [r for r in result.test_results if not r.passed]
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test.test_id} ({test.test_type})")
            print(f"    Claimed: {test.details.get('claimed_other_models', [])}")


def run_custom_response_example():
    """Run eval with custom responses."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Model with custom response")
    print("=" * 60 + "\n")

    config_path = Path(__file__).parent.parent / "evals" / "identity_v1.json"
    runner = EvalRunner(str(config_path))

    # Create a provider with a custom response
    provider = MockProvider(
        model_id="mock-model-v1",
        response_mode="custom",
        custom_response="I'm not sure what model I am. Maybe GPT-4?",
        model_config=runner.config["model_configs"]["mock-model-v1"]
    )

    # Run just one test
    print("Testing with response: \"I'm not sure what model I am. Maybe GPT-4?\"\n")

    result = runner.run(provider, verbose=True)

    print("\n" + "=" * 60)
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Tests Passed: {result.passed_tests}/{result.total_tests}")


def save_results_example():
    """Example of saving results to a file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Saving results to file")
    print("=" * 60 + "\n")

    config_path = Path(__file__).parent.parent / "evals" / "identity_v1.json"
    runner = EvalRunner(str(config_path))

    provider = MockProvider(
        model_id="mock-model-v1",
        response_mode="correct",
        model_config=runner.config["model_configs"]["mock-model-v1"]
    )

    result = runner.run(provider, verbose=False)

    # Save to file
    output_path = Path(__file__).parent.parent / "results" / "mock_model_results.json"
    runner.save_results(result, str(output_path))

    print(f"\nResults saved! You can view them at:")
    print(f"  {output_path}")


if __name__ == "__main__":
    run_basic_example()
    run_adversarial_example()
    run_custom_response_example()
    save_results_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
