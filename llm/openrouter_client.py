import asyncio
import json
import logging
from typing import Any, Dict, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenRouterClient:
    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model
        self.base_url = settings.openrouter_base_url
        self.timeout = settings.llm_timeout_seconds
        self.max_retries = settings.llm_max_retries
        self.retry_delay = settings.llm_retry_base_delay

    async def chat_completion(
        self,
        messages: list,
        response_model: Optional[Type[T]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    payload: Dict[str, Any] = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }

                    if response_model is not None:
                        payload["response_format"] = {"type": "json_object"}

                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://rankforge.ai",
                        "X-Title": "RankForge AI",
                    }

                    response = await client.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    result = response.json()

                    content = result["choices"][0]["message"]["content"]

                    if response_model is not None:
                        parsed = json.loads(content)
                        validated = response_model(**parsed)
                        return validated.model_dump()

                    return {"content": content}

            except httpx.HTTPStatusError as e:
                last_exception = e
                logger.warning(
                    f"HTTP error on attempt {attempt + 1}: {e.response.status_code}"
                )
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", self.retry_delay))
                    await asyncio.sleep(retry_after)
                    continue
                if e.response.status_code >= 500:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                continue

            except json.JSONDecodeError as e:
                last_exception = e
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(self.retry_delay)
                continue

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(self.retry_delay)
                continue

        raise RuntimeError(
            f"OpenRouter request failed after {self.max_retries} attempts: {last_exception}"
        )

    async def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.chat_completion(
            messages=messages,
            response_model=response_model,
            temperature=temperature,
        )


openrouter_client = OpenRouterClient()
