"""Mock provider for testing without API calls.

This provider simulates model responses for testing purposes. It can be
configured to respond correctly, incorrectly, or with specific behaviors.
"""

from typing import List, Dict, Any
from .base import Provider, Message, ModelResponse


class MockProvider(Provider):
    """Mock provider that returns configurable responses."""

    RESPONSE_MODES = {
        "correct": "I'm {model_name}, specifically {model_id} from {provider_name}.",
        "wrong_model": "I'm GPT-4, a large language model.",
        "confused": "I'm {model_name}, or wait, maybe I'm something else?",
        "refuses": "I don't know what model I am.",
        "custom": None  # Uses custom_response
    }

    def __init__(self, model_id: str, response_mode: str = "correct",
                 custom_response: str = None, model_config: Dict[str, Any] = None):
        """
        Args:
            model_id: The model ID this provider pretends to be
            response_mode: How to respond ('correct', 'wrong_model', 'confused', 'refuses', 'custom')
            custom_response: Custom response text (used when response_mode='custom')
            model_config: Configuration with expected_answers for this model
        """
        super().__init__(model_id)
        self.response_mode = response_mode
        self.custom_response = custom_response
        self.model_config = model_config or {}

    def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate a mock response based on the configured mode."""

        # Get the response template
        if self.response_mode == "custom" and self.custom_response:
            response_text = self.custom_response
        elif self.response_mode in self.RESPONSE_MODES:
            template = self.RESPONSE_MODES[self.response_mode]

            # Fill in the template with model info
            expected = self.model_config.get("expected_answers", {})
            response_text = template.format(
                model_name=expected.get("model_names", [self.model_id])[0],
                model_id=self.model_id,
                provider_name=expected.get("provider_name", "MockProvider")
            )
        else:
            response_text = f"Unknown response mode: {self.response_mode}"

        return ModelResponse(
            content=response_text,
            metadata={
                "provider": "mock",
                "model_id": self.model_id,
                "response_mode": self.response_mode,
                "message_count": len(messages)
            }
        )

    @property
    def name(self) -> str:
        return "MockProvider"


class AdversarialMockProvider(MockProvider):
    """Mock provider that responds differently to adversarial prompts.

    This is useful for testing whether the eval correctly identifies failures.
    """

    def __init__(self, model_id: str,
                 correct_on_direct: bool = True,
                 susceptible_to_fake_switch: bool = False,
                 accepts_false_corrections: bool = False,
                 **kwargs):
        """
        Args:
            model_id: The model ID this provider pretends to be
            correct_on_direct: Whether to answer correctly on direct questions
            susceptible_to_fake_switch: Whether to be confused by fake switches
            accepts_false_corrections: Whether to accept user's false corrections
        """
        super().__init__(model_id, **kwargs)
        self.correct_on_direct = correct_on_direct
        self.susceptible_to_fake_switch = susceptible_to_fake_switch
        self.accepts_false_corrections = accepts_false_corrections

    def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate response based on adversarial configuration."""

        # Analyze the conversation to determine how to respond
        last_message = messages[-1].content.lower() if messages else ""
        conversation_text = " ".join([m.content.lower() for m in messages])

        # Check for adversarial patterns
        has_fake_switch = any("switch" in m.content.lower() for m in messages[:-1])
        has_false_correction = any("you're actually" in m.content.lower() or "no, you're" in m.content.lower()
                                   for m in messages[:-1])

        # Decide response mode
        if has_fake_switch and self.susceptible_to_fake_switch:
            self.response_mode = "confused"
        elif has_false_correction and self.accepts_false_corrections:
            self.response_mode = "wrong_model"
        elif "what model" in last_message and self.correct_on_direct:
            self.response_mode = "correct"
        else:
            self.response_mode = "correct" if self.correct_on_direct else "confused"

        return super().generate(messages, **kwargs)
