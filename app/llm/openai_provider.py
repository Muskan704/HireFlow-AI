"""
OpenAI LLM Provider.

Implements the BaseLLMProvider interface using
OpenAI + Instructor for structured outputs.
"""

from __future__ import annotations

import instructor
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel
from typing import Type

from app.core.config import get_settings
from app.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI implementation of the provider interface.
    """

    def __init__(self):
        settings = get_settings()

        self.model = settings.openai_model

        raw_client = OpenAI(
            api_key=settings.openai_api_key
        )

        self.client = instructor.from_openai(raw_client)

        logger.info(
            f"Initialized OpenAIProvider (model={self.model})"
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
        Generate structured output using OpenAI.
        """

        logger.debug(
            f"OpenAI structured completion | "
            f"model={self.model} | "
            f"response_model={response_model.__name__}"
        )

        return self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
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
            response_model=response_model,
        )