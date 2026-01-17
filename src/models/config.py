"""Model configurations for different providers."""

from dataclasses import dataclass
from typing import Any

from keycycle import MultiProviderWrapper


@dataclass
class ModelConfig:
    """Configuration for a model."""

    provider: str
    model_id: str
    display_name: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0

    def __post_init__(self):
        if not self.display_name:
            self.display_name = f"{self.provider}/{self.model_id}"

    def get_client(self) -> Any:
        """Get an OpenAI-compatible client for this model."""
        wrapper = MultiProviderWrapper.from_env(self.provider, self.model_id)
        return wrapper.get_openai_client()


# Cerebras models
CEREBRAS_MODELS = [
    ModelConfig(
        provider="cerebras",
        model_id="zai-glm-4.7",
        display_name="Cerebras ZAI-GLM 4.7",
    ),
    ModelConfig(
        provider="cerebras",
        model_id="gpt-oss-120b",
        display_name="Cerebras GPT-OSS 120B",
    ),
    ModelConfig(
        provider="cerebras",
        model_id="zai-glm-4.6",
        display_name="Cerebras ZAI-GLM 4.6",
    ),
]

# Groq models
GROQ_MODELS = [
    ModelConfig(
        provider="groq",
        model_id="compound",
        display_name="Groq Compound",
    ),
    ModelConfig(
        provider="groq",
        model_id="moonshotai/kimi-k2-instruct-0905",
        display_name="Groq Kimi K2",
    ),
]

# Cohere models
COHERE_MODELS = [
    ModelConfig(
        provider="cohere",
        model_id="command-a-reasoning-08-2025",
        display_name="Cohere Command A Reasoning",
    ),
    ModelConfig(
        provider="cohere",
        model_id="command-a-03-2025",
        display_name="Cohere Command A",
    ),
]

# All models combined
ALL_MODELS = CEREBRAS_MODELS + GROQ_MODELS + COHERE_MODELS


def get_all_models() -> list[ModelConfig]:
    """Get all model configurations."""
    return ALL_MODELS


def get_models_by_provider(provider: str) -> list[ModelConfig]:
    """Get models for a specific provider."""
    provider = provider.lower()
    match provider:
        case "cerebras":
            return CEREBRAS_MODELS
        case "groq":
            return GROQ_MODELS
        case "cohere":
            return COHERE_MODELS
        case _:
            raise ValueError(f"Unknown provider: {provider}")


def get_model(provider: str, model_id: str) -> ModelConfig | None:
    """Get a specific model configuration."""
    for model in ALL_MODELS:
        if model.provider == provider and model.model_id == model_id:
            return model
    return None
