"""OpenAI Responses API wrapper and retry behavior."""

import logging
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from ai_config import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_VERBOSITY

logger = logging.getLogger(__name__)

load_dotenv()


def response_output_text(resp) -> str:
    """Extract assistant text from a Responses API result."""
    content = getattr(resp, "output_text", None)
    if isinstance(content, str):
        return content.strip()

    output = getattr(resp, "output", None) or []
    parts = []
    for item in output:
        for content_item in getattr(item, "content", []) or []:
            text = getattr(content_item, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def supports_reasoning_controls(model: str) -> bool:
    model = (model or "").lower()
    return model.startswith("gpt-5") or model.startswith("o")


def _prepare_responses_input(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    instructions = []
    input_items = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role in {"system", "developer"}:
            instructions.append(str(content))
        else:
            input_items.append({"role": role, "content": str(content)})
    return "\n\n".join(instructions), input_items


def _call_openai(
    client,
    messages,
    model=DEFAULT_MODEL,
    max_retries=2,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    verbosity: str = DEFAULT_VERBOSITY,
    text_format: Optional[Dict] = None,
) -> str:
    """Call OpenAI Responses API with retry logic for transient errors."""
    instructions, input_items = _prepare_responses_input(messages)
    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "input": input_items,
                "store": False,
            }
            if instructions:
                kwargs["instructions"] = instructions
            if supports_reasoning_controls(model) and reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            text_config = {}
            if supports_reasoning_controls(model) and verbosity:
                text_config["verbosity"] = verbosity
            if text_format:
                text_config["format"] = text_format
            if text_config:
                kwargs["text"] = text_config

            resp = client.responses.create(**kwargs)
            content = response_output_text(resp)
            status = getattr(resp, "status", None)
            bad_status = isinstance(status, str) and status in {"incomplete", "failed", "cancelled"}
            # an empty or truncated response silently drops data downstream
            # (whole report periods vanish) -- treat it as retryable
            if not content or bad_status:
                if attempt < max_retries:
                    wait = 2 ** attempt * 2
                    logger.warning(
                        "OpenAI returned %s response (status=%s); retrying in %ss",
                        "empty" if not content else "incomplete", status, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.warning("OpenAI returned empty/incomplete content after retries (status=%s)", status)
                return content or ""
            return content
        except RateLimitError:
            if attempt < max_retries:
                wait = 2 ** attempt * 5
                logger.warning("Rate limited, retrying in %ss (attempt %s/%s)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                logger.error("Rate limit exceeded after %s retries", max_retries)
                raise
        except (APIConnectionError, APITimeoutError) as e:
            if attempt < max_retries:
                wait = 2 ** attempt * 2
                logger.warning("Connection error, retrying in %ss (attempt %s/%s)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                logger.error("Connection failed after %s retries: %s", max_retries, e)
                raise
        except APIError as e:
            logger.error("OpenAI API error: %s", e)
            raise
    return ""
