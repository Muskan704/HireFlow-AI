"""
Anthropic LLM Provider.

Implements the BaseLLMProvider interface using
Anthropic + Instructor.
"""

from __future__ import annotations

import instructor
import anthropic

from loguru import logger
from pydantic import BaseModel
from typing import Type

from app.core.config import get_settings
from app.llm.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic implementation.
    """

    def __init__(self):
        settings = get_settings()

        self.model = settings.anthropic_model

        raw_client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )

        self.client = instructor.from_anthropic(raw_client)

        logger.info(
            f"Initialized AnthropicProvider (model={self.model})"
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
        Generate structured output using Anthropic.
        """

        logger.debug(
            f"Anthropic structured completion | "
            f"model={self.model} | "
            f"response_model={response_model.__name__}"
        )

        return self.client.messages.create(
            model=self.model,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            response_model=response_model,
        )