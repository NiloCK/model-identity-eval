from anthropic import Anthropic

from .base import Message, ModelResponse, Provider, ProviderError


class AnthropicProvider(Provider):
    def __init__(self, model_id: str, api_key: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.client = Anthropic(api_key=api_key)

    def generate(self, messages, **kwargs):
        try:
            response = self.client.messages.create(
                model=self.model_id,
                messages=[m.to_dict() for m in messages],
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            return ModelResponse(
                content=response.content[0].text, metadata={"usage": response.usage}
            )
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}")

    @property
    def name(self):
        return "Anthropic"
