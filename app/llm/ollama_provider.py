"""
Ollama LLM Provider.

Implements the BaseLLMProvider interface using
Ollama + Instructor for structured outputs.
"""

from __future__ import annotations

from typing import Type

import instructor
from loguru import logger
from pydantic import BaseModel

from app.core.config import get_settings
from app.llm.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Ollama implementation of the provider interface.
    """

    def __init__(self):
        settings = get_settings()

        self.model = settings.ollama_model
        self.base_url = settings.ollama_host

        self.client = instructor.from_provider(
            model=f"ollama/{self.model}",
            base_url=self.base_url,
        )

        logger.info(
            f"Initialized OllamaProvider (model={self.model})"
        )

    def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        """
        Generate structured output from Ollama.

        Returns a validated Pydantic object.
        """

        logger.debug(
            f"Ollama structured completion | "
            f"model={self.model} | "
            f"response_model={response_model.__name__}"
        )

        return self.client.chat.completions.create(
            response_model=response_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )