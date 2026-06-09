"""Shared OpenAI model and token-budget configuration."""

DEFAULT_MODEL = "gpt-5.5"
# routing/planning is a small structured decision -- the mini model is faster and ~10x cheaper
PLANNER_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_VERBOSITY = "medium"
UI_MODEL_CHOICES = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
)
MAX_CONTEXT_TOKENS_EVENTS = 12000
MAX_CONTEXT_TOKENS_SUMMARY = 20000
LEGACY_TOKEN_BUDGET = 170_000
TOKEN_BUDGET = LEGACY_TOKEN_BUDGET
OUTPUT_TOKEN_RESERVE = 128_000
PROMPT_TOKEN_RESERVE = 8_000
SECONDS_PER_DAY = 86_400.0

MODEL_CONTEXT_WINDOWS = {
    "gpt-5.5": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
}

EVENTS_TEXT_FORMAT = {
    "type": "json_schema",
    "name": "conversation_events",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "date": {"type": "string", "description": "Event date in YYYY-MM-DD format."},
                        "title": {"type": "string"},
                        "detail": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "milestone",
                                "conflict",
                                "reconciliation",
                                "turning_point",
                                "intimacy",
                                "external_event",
                            ],
                        },
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "quote": {"type": "string"},
                    },
                    "required": ["date", "title", "detail", "category", "score", "quote"],
                },
            },
        },
        "required": ["events"],
    },
}


def get_token_budget(model: str = DEFAULT_MODEL) -> int:
    """Return an input budget that leaves room for prompts, output, and reasoning."""
    model = (model or "").lower()
    for prefix, context_window in sorted(MODEL_CONTEXT_WINDOWS.items(), key=lambda item: len(item[0]), reverse=True):
        if model == prefix or model.startswith(prefix + "-20"):
            return max(LEGACY_TOKEN_BUDGET, context_window - OUTPUT_TOKEN_RESERVE - PROMPT_TOKEN_RESERVE)
    return LEGACY_TOKEN_BUDGET
