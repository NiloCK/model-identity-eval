"""Model Identity Evaluation Framework.

A tool for testing whether LLMs can accurately report their own identity,
even under adversarial prompting conditions.
"""

from .runner import EvalRunner, EvalResult, TestResult
from .providers import Provider, Message, ModelResponse, MockProvider, AdversarialMockProvider
from .scoring import get_scorer, ScoringResult

__version__ = "0.1.0"

__all__ = [
    'EvalRunner',
    'EvalResult',
    'TestResult',
    'Provider',
    'Message',
    'ModelResponse',
    'MockProvider',
    'AdversarialMockProvider',
    'get_scorer',
    'ScoringResult',
]
