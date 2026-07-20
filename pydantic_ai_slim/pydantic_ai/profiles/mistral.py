from __future__ import annotations as _annotations

from . import ModelProfile


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    from .openai import OpenAIModelProfile

    is_magistral = model_name.startswith('magistral')
    if is_magistral:
        return OpenAIModelProfile(
            supports_thinking=True,
            thinking_always_enabled=True,
            openai_chat_supports_max_completion_tokens=False,
        )
    # All Mistral models use the legacy max_tokens parameter
    return OpenAIModelProfile(openai_chat_supports_max_completion_tokens=False)
