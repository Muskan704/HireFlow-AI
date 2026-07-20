"""
LLM Provider Factory

This module is the single entry point for obtaining an LLM provider.

The rest of the application should NEVER import Groq, OpenAI,
Anthropic, or Ollama directly.

Instead, always do:

    from app.llm.factory import get_llm

    llm = get_llm()

and use:

    llm.structured_completion(...)
"""

from __future__ import annotations

from app.core.config import get_settings

from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    settings = get_settings()
    provider = settings.active_llm.lower()

    if provider == "groq":
        from app.llm.groq_provider import GroqProvider
        return GroqProvider()

    elif provider == "ollama":
        from app.llm.ollama_provider import OllamaProvider
        return OllamaProvider()

    elif provider == "openai":
        from app.llm.openai_provider import OpenAIProvider
        return OpenAIProvider()
    
    elif provider == "gemini":
        from app.llm.gemini_provider import GeminiProvider
        return GeminiProvider()

    elif provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    raise ValueError(
        f"Unsupported ACTIVE_LLM='{provider}'. "
        "Supported providers: groq, ollama, openai, anthropic."
    )