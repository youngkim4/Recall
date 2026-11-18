"""report_engine: structured, incremental reports on one conversation.

    from report_engine import run_report, estimate, ReportStore, PeriodCache

Adapters live in their own modules, so that importing this package loads
neither pandas nor a model SDK:

    report_engine.frame_source.FrameSource
    report_engine.archive_source.ArchiveSource
    report_engine.openai_llm.OpenAIResponsesLLM
    report_engine.fake_llm.FakeLLM

See README.md in this folder.
"""

from report_engine.cache import PeriodCache
from report_engine.call_pool import CallPool
from report_engine.control import CancelToken, ProgressEvent, Runtime
from report_engine.errors import (
    AllPeriodsFailed,
    ConfigError,
    EmptyConversationError,
    LLMError,
    ReduceError,
    ReportCancelled,
    ReportEngineError,
    ReportNotFound,
    RunError,
    SourceError,
    StoreError,
)
from report_engine.estimate import estimate
from report_engine.markdown import render_markdown
from report_engine.model_info import LongContextRule, ModelInfo
from report_engine.models import (
    CATEGORIES,
    SCHEMA_VERSION,
    SECTION_KEYS,
    SECTION_TITLES,
    Moment,
    PeriodSummary,
    Report,
    Scope,
    Section,
    Usage,
)
from report_engine.options import ReportOptions
from report_engine.parallel import TaskOutcome, run_in_parallel
from report_engine.pipeline import run_report
from report_engine.ports import (
    LLM,
    ChatMessage,
    Labeler,
    LLMResult,
    LLMUsage,
    ReportSource,
    SourceMessage,
)
from report_engine.store import ReportEntry, ReportStore

__all__ = [
    "AllPeriodsFailed",
    "CATEGORIES",
    "CallPool",
    "CancelToken",
    "ChatMessage",
    "ConfigError",
    "EmptyConversationError",
    "LLM",
    "LLMError",
    "LLMResult",
    "LLMUsage",
    "Labeler",
    "LongContextRule",
    "ModelInfo",
    "Moment",
    "PeriodCache",
    "PeriodSummary",
    "ProgressEvent",
    "ReduceError",
    "Report",
    "ReportCancelled",
    "ReportEngineError",
    "ReportEntry",
    "ReportNotFound",
    "ReportOptions",
    "ReportSource",
    "ReportStore",
    "RunError",
    "Runtime",
    "SCHEMA_VERSION",
    "SECTION_KEYS",
    "SECTION_TITLES",
    "Scope",
    "Section",
    "SourceError",
    "SourceMessage",
    "StoreError",
    "TaskOutcome",
    "Usage",
    "estimate",
    "render_markdown",
    "run_in_parallel",
    "run_report",
]
