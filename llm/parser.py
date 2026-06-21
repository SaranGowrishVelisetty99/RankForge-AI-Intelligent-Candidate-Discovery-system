import json
import logging
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def parse_json_response(
    response_text: str,
    model_class: Type[T],
) -> Optional[T]:
    try:
        data = json.loads(response_text)
        validated = model_class(**data)
        return validated
    except json.JSONDecodeError as e:
        cleaned = _clean_json(response_text)
        try:
            data = json.loads(cleaned)
            validated = model_class(**data)
            return validated
        except (json.JSONDecodeError, ValidationError) as e2:
            logger.error(f"Failed to parse response as {model_class.__name__}: {e2}")
            return None
    except ValidationError as e:
        logger.error(f"Validation error for {model_class.__name__}: {e}")
        return None


def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
