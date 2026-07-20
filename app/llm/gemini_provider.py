"""
Gemini LLM Provider.

Implements the BaseLLMProvider interface using
Google Gemini + Instructor for structured outputs.
"""

from __future__ import annotations

import instructor
from google import genai
from loguru import logger
from pydantic import BaseModel
from typing import Type

from app.core.config import get_settings
from app.llm.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """
    Gemini implementation of the provider interface.
    """

    def __init__(self):
        settings = get_settings()

        self.model = settings.gemini_model

        # Create Instructor client directly from provider
        self.client = instructor.from_provider(
            f"google/{self.model}",
            api_key=settings.gemini_api_key,
        )

        logger.info(
            f"Initialized GeminiProvider (model={self.model})"
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
        Generate structured output from Gemini.

        Returns a validated Pydantic object.
        """

        logger.debug(
            f"Gemini structured completion | "
            f"model={self.model} | "
            f"response_model={response_model.__name__}"
        )

        return self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
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
        )