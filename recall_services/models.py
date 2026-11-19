"""One source of truth for the models: ids, levels, limits and prices.

The facts are in `models.json` beside this file. Each entry says on which
day its facts were read. A fact that was not read is `null`: no number is made
up for it, and a model can be offered only when its limits are on file.

Rules:
- An id must be in the file exactly as it is written there. An unknown id is a
  typed error. It is never priced or run as another model.
- A model that is in the file and not offered can not be run.
- When the page of a model gives no largest input, the largest input is the
  context window less the largest output.
- Limits that do not fit each other, a number written as text, and a provider
  that no model port is made for are refused when the file is read.
- The file is read once. A change of the file needs a new start of the server.
- A tokenizer family that is an assumption goes to the report engine under a
  name that is no encoding. The engine then counts with its own encoding and
  adds its safety margin to every token budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from report_engine.model_info import LongContextRule, ModelInfo

FACTS_PATH = Path(__file__).with_name("models.json")
UNCONFIRMED_FAMILY = "unconfirmed:"
# a provider as a sentence names it: the app names no provider (the owner: "dont even mention openai")
SOME_PROVIDER = "The AI service"
PROVIDER_NAMES = {"openai": SOME_PROVIDER}

Level = Literal["low", "medium", "high", "top"]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Count = Annotated[int, Field(gt=0, strict=True)]
Price = Annotated[float, Field(ge=0)]


class ModelFactsError(Exception):
    """The file of model facts can not be used."""


class IncompleteModelFactsError(ModelFactsError):
    """A model is offered and a limit that a run needs is not on file."""

    def __init__(self, model_id: str, missing: tuple[str, ...]) -> None:
        super().__init__(f"the model {model_id} is offered and has no {', '.join(missing)} on file")
        self.model_id = model_id
        self.missing = missing


class ModelChoiceError(Exception):
    """A request named a model that can not be run. `code` is for the client."""

    code = "model_error"

    def __init__(self, model_id: str, message: str) -> None:
        super().__init__(message)
        self.model_id = model_id


class UnknownModelIdError(ModelChoiceError):
    code = "unknown_model"


class ModelNotOfferedError(ModelChoiceError):
    code = "model_not_offered"


class ReasoningLevelError(ModelChoiceError):
    """A request asked a model for a reasoning effort that the file does not list for it."""

    code = "reasoning_level"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def _is_day(value: str | None) -> str | None:
    if value is not None:
        date.fromisoformat(value)
    return value


class ModelFacts(_Record):
    """What the file says about one model. Prices are USD for one million tokens."""

    id: Name
    provider: Name
    label: Name
    level: Level
    offered: StrictBool
    context_window: Count
    max_input_tokens: Count | None
    max_output_tokens: Count | None
    input_usd_per_mtok: Price
    cached_input_usd_per_mtok: Price
    output_usd_per_mtok: Price
    long_context: LongContextRule | None
    tokens_per_minute: Count | None
    tokenizer_family: Name
    tokenizer_family_assumed: StrictBool
    reasoning_levels: tuple[Name, ...] | None
    shutdown_date: str | None
    facts_read_on: str

    _days = field_validator("shutdown_date", "facts_read_on")(_is_day)

    @model_validator(mode="after")
    def _limits_fit(self) -> ModelFacts:
        """Limits that do not fit each other are refused here, and not later in the middle of a request."""
        window = self.context_window
        if self.provider not in PROVIDER_NAMES:
            raise ValueError(f"no model port is made for the provider {self.provider}")
        if self.max_input_tokens is not None and self.max_input_tokens > window:
            raise ValueError("max_input_tokens is above context_window")
        if self.max_output_tokens is not None and self.max_output_tokens >= window:
            raise ValueError("max_output_tokens leaves no room for input in context_window")
        if self.long_context is not None and self.long_context.threshold_tokens > window:
            raise ValueError("long_context.threshold_tokens is above context_window")
        return self


class EffortLimits(_Record):
    """The limits of a chat question that asks for one effort."""

    max_output_tokens: Count
    timeout_seconds: Count


class ChatSettings(_Record):
    """What a chat question asks of a model.

    `max_output_tokens` is the most output of one call: reasoning is counted
    in it. `timeout_seconds` is the longest time between two pieces of an
    answer: a model sends nothing while it reasons. `reasoning_effort` is the
    effort of a question that asks for none. `by_effort` gives an effort its
    own two limits; an effort that is not in it has the limits above.
    """

    max_output_tokens: Count
    timeout_seconds: Count = 60
    reasoning_effort: Name | None = None
    by_effort: dict[Name, EffortLimits] = Field(default_factory=dict)


class _FactsFile(_Record):
    facts_read_on: str
    notes: tuple[str, ...] = ()
    default: Name
    chat: ChatSettings
    models: tuple[ModelFacts, ...] = Field(min_length=1)

    _days = field_validator("facts_read_on")(_is_day)


@dataclass(frozen=True)
class ChatModelConfig:
    """What the chat engine and its model port need for one model."""

    model: str
    provider: str
    max_output_tokens: int
    reasoning_effort: str | None
    # the longest time between two pieces of an answer
    timeout_seconds: int = 60


@dataclass(frozen=True)
class Registry:
    models: tuple[ModelFacts, ...]
    default: str
    chat: ChatSettings

    def find(self, model_id: object) -> ModelFacts | None:
        return next((facts for facts in self.models if facts.id == model_id), None)

    def offered(self) -> tuple[ModelFacts, ...]:
        return tuple(facts for facts in self.models if facts.offered)


def _missing(facts: ModelFacts) -> tuple[str, ...]:
    """The limits a run needs that are not on file."""
    return tuple(name for name in ("max_output_tokens",) if getattr(facts, name) is None)


def _read(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelFactsError(f"the model facts can not be read: {type(exc).__name__}") from exc
    except ValueError as exc:
        raise ModelFactsError("the model facts are not JSON") from exc


def _where(error: ValidationError) -> str:
    places = sorted({".".join(str(part) for part in item["loc"]) for item in error.errors()})
    return ", ".join(places[:8])


def _checked(parsed: _FactsFile) -> Registry:
    ids = [facts.id for facts in parsed.models]
    twice = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
    if twice:
        raise ModelFactsError(f"the model facts hold {', '.join(twice)} two times")
    registry = Registry(models=parsed.models, default=parsed.default, chat=parsed.chat)
    for facts in registry.offered():
        missing = _missing(facts)
        if missing:
            raise IncompleteModelFactsError(facts.id, missing)
    if parsed.default not in {facts.id for facts in registry.offered()}:
        raise ModelFactsError(f"the default model {parsed.default} is not an offered model")
    return registry


def load_registry(path: Path = FACTS_PATH) -> Registry:
    """Read and check a file of model facts. Raises ModelFactsError for a file that can not be used."""
    raw = _read(path)
    try:
        parsed = _FactsFile.model_validate(raw)
    except ValidationError as exc:
        raise ModelFactsError(f"the model facts have a bad value at: {_where(exc)}") from exc
    return _checked(parsed)


@lru_cache(maxsize=1)
def _registry() -> Registry:
    return load_registry(FACTS_PATH)


def _choices(registry: Registry) -> str:
    return ", ".join(facts.id for facts in registry.offered())


def _as_text(model_id: object) -> str:
    return "" if model_id is None else str(model_id)


def facts(model_id: str) -> ModelFacts:
    """The facts of any model in the file, offered or not."""
    registry = _registry()
    found = registry.find(model_id)
    if found is not None:
        return found
    named = _as_text(model_id)
    lead = f"Recall does not know the model {named}." if named.strip() else "No model was named."
    raise UnknownModelIdError(named, f"{lead} Choose one of: {_choices(registry)}.")


def _runnable(model_id: str) -> ModelFacts:
    found = facts(model_id)
    if not found.offered:
        raise ModelNotOfferedError(
            found.id, f"The model {found.id} is not offered. Choose one of: {_choices(_registry())}."
        )
    return found


def provider_name(provider: str) -> str:
    """The name of a provider for a sentence that a person reads."""
    return PROVIDER_NAMES.get(provider, SOME_PROVIDER)


def offered_ids() -> list[str]:
    """The ids of the offered models, in the order of the file: lowest level first."""
    return [found.id for found in _registry().offered()]


def _default_effort(found: ModelFacts) -> str | None:
    """The effort of the file for chat, when the model lists it. A model with no levels on file gets none."""
    effort = _registry().chat.reasoning_effort
    return effort if effort in (found.reasoning_levels or ()) else None


def offered() -> list[dict[str, Any]]:
    """Each offered model as the interface names it.

    `id`, `label` and `level`, then `reasoningLevels`: the efforts that a chat
    question can ask of the model, least first, and `reasoningDefault`: the
    effort of a question that asks for none. A model with no levels on file
    has an empty list and no default.
    """
    return [
        {
            "id": found.id,
            "label": found.label,
            "level": found.level,
            "reasoningLevels": list(found.reasoning_levels or ()),
            "reasoningDefault": _default_effort(found),
        }
        for found in _registry().offered()
    ]


def default_id() -> str:
    return _registry().default


def defaults_payload() -> dict[str, Any]:
    """The model keys of /api/defaults: `defaultModel` and `models` as today, and `modelChoices`."""
    return {"defaultModel": default_id(), "models": offered_ids(), "modelChoices": offered()}


def _engine_family(found: ModelFacts) -> str:
    """The family name for the report engine. An assumed family is no encoding name, so the engine adds its margin."""
    if found.tokenizer_family_assumed:
        return f"{UNCONFIRMED_FAMILY}{found.tokenizer_family}"
    return found.tokenizer_family


def model_info(model_id: str) -> ModelInfo:
    """The facts of an offered model in the shape of the report engine."""
    found = _runnable(model_id)
    largest_output = found.max_output_tokens
    if largest_output is None:  # load_registry refuses an offered model like this
        raise IncompleteModelFactsError(found.id, _missing(found))
    return ModelInfo(
        id=found.id,
        context_window=found.context_window,
        max_input_tokens=found.max_input_tokens or found.context_window - largest_output,
        max_output_tokens=largest_output,
        input_usd_per_mtok=found.input_usd_per_mtok,
        cached_input_usd_per_mtok=found.cached_input_usd_per_mtok,
        output_usd_per_mtok=found.output_usd_per_mtok,
        long_context=found.long_context,
        tokens_per_minute=found.tokens_per_minute,
        tokenizer_family=_engine_family(found),
    )


def _effort(found: ModelFacts, asked: object) -> str | None:
    """The effort for one question: the one it asks for, or the effort of the file."""
    wanted = asked.strip() if isinstance(asked, str) else ""
    if not wanted:
        return _default_effort(found)
    levels = found.reasoning_levels or ()
    if wanted in levels:
        return wanted
    can = f"Choose one of: {', '.join(levels)}." if levels else "It has no reasoning levels on file."
    raise ReasoningLevelError(found.id, f"The model {found.id} does not take the reasoning level {wanted[:40]}. {can}")


def chat_config(model_id: str, reasoning_effort: object = None) -> ChatModelConfig:
    """What a chat question needs for an offered model.

    `reasoning_effort` is the effort that the question asks for. It must be
    one of the reasoning levels that the file lists for the model: another
    word is a ReasoningLevelError. With none asked for, the effort of the file
    is given when the model lists it. With no effort the model port does not
    ask for reasoning. The most output and the time between two pieces of an
    answer are the ones of the effort (`by_effort`), or the ones of chat.
    """
    found = _runnable(model_id)
    settings = _registry().chat
    effort = _effort(found, reasoning_effort)
    limits = settings.by_effort.get(effort or "")
    cap = limits.max_output_tokens if limits else settings.max_output_tokens
    return ChatModelConfig(
        model=found.id,
        provider=found.provider,
        max_output_tokens=cap if found.max_output_tokens is None else min(cap, found.max_output_tokens),
        reasoning_effort=effort,
        timeout_seconds=limits.timeout_seconds if limits else settings.timeout_seconds,
    )
