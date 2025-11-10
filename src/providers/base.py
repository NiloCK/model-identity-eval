"""Base provider interface for model identity evaluation.

This module defines the abstract interface that all provider implementations
must follow. Providers handle communication with specific LLM APIs.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class Message:
    """Represents a single message in a conversation."""

    def __init__(self, role: str, content: str):
        """
        Args:
            role: Either 'user' or 'assistant'
            content: The message text
        """
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Message':
        return cls(role=data["role"], content=data["content"])


class ModelResponse:
    """Response from a model provider."""

    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        """
        Args:
            content: The model's response text
            metadata: Optional metadata (tokens used, latency, etc.)
        """
        self.content = content
        self.metadata = metadata or {}


class Provider(ABC):
    """Abstract base class for all model providers."""

    def __init__(self, model_id: str, **kwargs):
        """
        Args:
            model_id: Identifier for the specific model to use
            **kwargs: Provider-specific configuration
        """
        self.model_id = model_id
        self.config = kwargs

    @abstractmethod
    def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate a response given a conversation history.

        Args:
            messages: List of Message objects representing the conversation
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            ModelResponse containing the generated text and metadata

        Raises:
            ProviderError: If the API call fails
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this provider."""
        pass


class ProviderError(Exception):
    """Raised when a provider encounters an error."""
    pass
