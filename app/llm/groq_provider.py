"""
Groq LLM Provider.

Implements the BaseLLMProvider interface using
Groq + Instructor for structured outputs.
"""

from __future__ import annotations

import instructor
import re
import time
from groq import Groq
from loguru import logger
from pydantic import BaseModel
from typing import Type

from app.core.config import get_settings
from app.llm.base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """
    Groq implementation of the provider interface.
    """

    def __init__(self):
        settings = get_settings()

        self.model = settings.groq_model

        raw_client = Groq(
            api_key=settings.groq_api_key
        )

        self.client = instructor.from_groq(
            raw_client,
            mode=instructor.Mode.JSON,
        )

        logger.info(
            f"Initialized GroqProvider (model={self.model})"
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
        Generate structured output from Groq.

        Returns a validated Pydantic object.
        """

        logger.debug(
            f"Groq structured completion | "
            f"model={self.model} | "
            f"response_model={response_model.__name__}"
        )

        last_error = None
        for attempt in range(1, 4):
            try:
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
            except Exception as e:
                last_error = e
                retry_after = _extract_retry_after_seconds(str(e))
                if retry_after is None or attempt == 3:
                    raise

                wait_seconds = retry_after + 1.0
                logger.warning(
                    f"Groq rate limit hit; waiting {wait_seconds:.1f}s "
                    f"before retry {attempt + 1}/3"
                )
                time.sleep(wait_seconds)

        raise last_error


def _extract_retry_after_seconds(error_message: str) -> float | None:
    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", error_message, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))
