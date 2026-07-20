"""
Base interface for all LLM providers.

Every provider (Groq, OpenAI, Anthropic, Ollama)
must implement this interface so that the rest
of the application never depends on a specific LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from pydantic import BaseModel


class BaseLLMProvider(ABC):
    """
    Common interface implemented by every LLM provider.

    Services should NEVER directly call
    Groq/OpenAI/Ollama clients.

    Instead they call

        llm.structured_completion(...)

    and the provider handles the implementation.
    """

    @abstractmethod
    def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        """
        Return a validated Pydantic object.

        Parameters
        ----------
        system_prompt:
            System instructions.

        user_prompt:
            User content.

        response_model:
            Pydantic model expected.

        temperature:
            Sampling temperature.

        max_tokens:
            Maximum output tokens.

        Returns
        -------
        Pydantic model instance.
        """
        raise NotImplementedError