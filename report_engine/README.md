# report_engine

`report_engine` writes the report on one conversation.

It replaces the pipeline in `analysis.py` (`run_cli`), `ai_analysis.py` and `reports.py`.
It does not import `ui_server` or any other module at the repository root.

## What it does

1. It reads the messages of one conversation from a source.
2. It splits the conversation into periods on calendar boundaries.
3. It calls the model one time for each period, oldest first (map step).
4. It checks every date, quote and reference against the messages.
5. It selects the moments of the report.
6. It calls the model one time to write four sections (reduce step).
7. It returns a `Report`. The caller saves it.

A period result goes into a cache. A second run calls the model only for periods whose text changed.

## Run a report

```python
from report_engine import (
    CancelToken,
    ModelInfo,
    PeriodCache,
    ReportStore,
    estimate,
    run_report,
)
from report_engine.frame_source import FrameSource
from report_engine.openai_llm import OpenAIResponsesLLM, build_client

model_info = ModelInfo(...)  # see "Model facts"
source = FrameSource(messages_frame)  # or ArchiveSource.open(path)
cache = PeriodCache(data_dir / "report-cache.sqlite3")
store = ReportStore(data_dir / "reports")
llm = OpenAIResponsesLLM(build_client(), model_info.id)
scope = {"since": "2024-06-01", "until": "2024-06-30"}  # or None for all dates

quote = estimate(source, chat_id, scope, model_info, cache, labeler=labeler)

cancel = CancelToken()
report = run_report(
    source=source,
    conversation=chat_id,
    scope=scope,
    llm=llm,
    model_info=model_info,
    labeler=labeler,
    cache=cache,
    stats=stats,  # any figures; stored as given
    on_progress=show_progress,  # optional
    cancel=cancel,  # optional
)
path = store.save(report)
```

Give `estimate` and `run_report` the same labeler, options and token counter.
Different values give different periods and different cache keys.

## What the caller supplies

| Part | Type | Notes |
|------|------|-------|
| Source | `ReportSource` | `messages(conversation, since, until)` yields messages, oldest first. |
| Labeler | `(sender: str) -> str` | The name a speaker has in text that the model reads. |
| Model | `LLM` | One model, bound to its id. |
| Model facts | `ModelInfo` | Limits, prices, rate limit, tokenizer family. |
| Cache | `PeriodCache` | One SQLite file. `None` is permitted. |
| Store | `ReportStore` | One folder. The engine does not write to it; the caller calls `save`. |

### Source

A message is a dict: `id` (str), `timestamp` (`YYYY-MM-DD HH:MM:SS`, local time, no zone), `is_from_me` (bool), `sender` (str), `text` (str).

`since` and `until` are days (`YYYY-MM-DD`) or full timestamps. Both ends are included. A day means the whole day.

- `FrameSource(frame)` reads a pandas DataFrame with the export columns `message_id`, `timestamp`, `sender`, `text`, `chat_id`, `is_from_me`. It matches `chat_id` exactly.
- `ArchiveSource(archive)` reads `archive.iter_messages(conversation_id, since=..., until=...)`. It skips rows whose `kind` is `system`. `ArchiveSource.open(path)` imports the `archive` package at that moment. This adapter is tested with a stand-in only.

### Labeler

The engine calls the labeler one time for each sender that is not the owner.
Own messages always get the name `You`.

The engine replaces a name with `Unsaved contact` when the name is empty, is `You`, or looks like a phone number or an e-mail address.
The conversation id never goes into a prompt.

In a one-to-one chat the source can give an empty sender. Return the contact's name for the empty sender.

### Model facts

The engine holds no model id, price or limit. All of them come from `ModelInfo`:

`id`, `context_window`, `max_input_tokens`, `max_output_tokens`, `input_usd_per_mtok`, `cached_input_usd_per_mtok`, `output_usd_per_mtok`, `long_context` (threshold and price multipliers, or `None`), `tokens_per_minute` (or `None`), `tokenizer_family`.

Prices are in USD for one million tokens.

### Model

`LLM.generate(messages, *, schema=None, max_output_tokens, reasoning=None)` returns `LLMResult` with `text`, `parsed`, `usage`, `status` and `attempts`.

- `usage.input_tokens` includes the cached tokens. `usage.cached_input_tokens` is the cached part.
- `status` is `completed`, `incomplete`, `refused` or `failed`.
- The port raises `LLMError` when it cannot answer. It does its own retries.

`OpenAIResponsesLLM` sends `store=False`, a timeout and `max_output_tokens` with every request.
It retries a rate limit, a timeout, a connection error and a 5xx reply, 2 times by default.
It waits the time that `retry-after` gives. When that time is longer than 60 seconds it stops and raises.
It does not send a cut-off reply again.

`FakeLLM` (`report_engine.fake_llm`) is a model for tests. It makes no network call.

## Options

`ReportOptions` holds the choices of a run. The defaults are:

| Option | Default | Meaning |
|--------|---------|---------|
| `period_token_budget` | 80,000 | Largest transcript of one period, in tokens. |
| `map_max_output_tokens` | 16,000 | Hard output limit of a map call. |
| `reduce_max_output_tokens` | 24,000 | Hard output limit of the reduce call. |
| `reasoning` | `None` | Reasoning effort, passed to the model as given. |
| `score_floor` | 0.4 | A moment below this score is not in the report. |
| `max_moments` | 40 | Largest number of moments in the report. |
| `max_message_chars` | 6,000 | A longer message keeps its start and its end. |
| `carry_over_token_budget` | 4,000 | Room for notes from earlier periods. |
| `tokenizer_mismatch_margin` | 1.35 | Safety margin for token counts. See "Tokens". |

The output limits include reasoning tokens. Set them high enough for the reasoning effort.

## Periods

The engine packs calendar months, oldest first, until the next month does not fit the budget.
A month that is larger than the budget stands alone and is split by days.
A day that is larger than the budget stands alone and is split by messages.

New messages change only the last period. A period that is closed does not change.

The budget is also limited by the model. One request must fit `max_input_tokens`, the context window and one minute of `tokens_per_minute`.

## Cache

The key of a period is a hash of four things: the prompt version, the model id, the speaker names in the period, and the period transcript.

The key does not hold the notes from earlier periods. A change in an old period does not run the later periods again.
The key does not hold the reasoning effort. To force a new run, call `PeriodCache.clear()`.

A failed period has no cache entry. The next run tries it again.
The cache file holds summaries and quotes. Keep it with the archive.

## The report

`Report` is a pydantic model. `report.model_dump(mode="json")` gives the JSON form.

- `schema_version`, `id`, `conversation`, `scope` (`since`, `until`), `model`, `created_at` (UTC)
- `status`: `complete` or `partial`
- `warnings`: text lines
- `sections`: always `arc`, `dynamics`, `defining_moments`, `big_picture`, in this order. Each has `key`, `title`, `markdown`.
- `moments`: the selected moments, oldest first
- `periods`: one `PeriodSummary` for each period
- `usage`: `input_tokens`, `cached_input_tokens`, `output_tokens`, `cost_usd`, `calls` of this run
- `stats`: the figures from the caller

A `Moment` has `date`, `title`, `detail`, `category`, `score`, `quote`, `message_ids`, `quote_verified`.

- `message_ids` are real message ids. Use them to open the exchange. The list can be empty.
- `date` is always a day on which the period has a message.
- `quote` is empty when `quote_verified` is false.

A `PeriodSummary` has `label`, `first_day`, `last_day`, `summary`, `carry_over`, `moments`, `usage`, `cache_key`, `from_cache`, `error`, `message_count`.
For a period from the cache, `usage` is the cost of the call that made the entry. `report.usage` counts only the calls of this run.

## Checks

- A reference such as `m12` maps to a message of the period. A reference that does not exist is ignored.
- A date must be a day of the period. A wrong date moves to the earliest cited day. A moment with no usable date is dropped.
- A quote must be part of a cited message, or of a message within three days of the moment's date. A quote of fewer than 8 characters must be in a cited message.
- Two texts match when they are equal after white space, quote marks and Unicode composition are made the same. Letter case must match.
- A quote that is not found is cleared. The report gets a warning with the count.

## Failures

| Event | Result |
|-------|--------|
| A period fails | The report is `partial`. A warning names the period. |
| All periods fail | `AllPeriodsFailed`. No report. |
| The reduce reply is empty, cut off or has an empty section | `ReduceError`. No report. |
| The cancel token is set | `ReportCancelled`. No report. |
| The scope has no message with text | `EmptyConversationError`. No model call. |
| The model limits leave no room for a period | `ConfigError`. No model call. |

`AllPeriodsFailed`, `ReduceError` and `ReportCancelled` have a `usage` attribute. It holds the tokens and the cost of the run until that moment.
Period results stay in the cache. The next run does not pay for them again.

## Cancel and progress

`CancelToken.cancel()` stops the run before the next model call. A call that is in progress completes first.

`on_progress` gets a `ProgressEvent` with `step`, `state`, `period`, `index` and `total`.

- `step` is `prepare`, `map`, `reduce` or `done`.
- `state` is `start`, `cached`, `waiting`, `done` or `failed`.
- `waiting` means that the run waits for the rate limit.

## Tokens and the rate limit

The engine counts tokens with `tiktoken`.
It uses the encoding with the name `ModelInfo.tokenizer_family` when `tiktoken` has it. If not, it uses `o200k_base`.
When the encoding is not the tokenizer family of the model, each count is multiplied by `tokenizer_mismatch_margin`.
When no vocabulary file can load, the engine counts one token for each three bytes and writes a warning to the log.
The engine does not load a tokenizer at import time.

Before each call the engine reserves the input estimate plus the output limit in a 60-second window.
It waits when the window is full. After the call it replaces the reserve with the real usage.

## Estimate

`estimate(source, conversation, scope, model_info, cache, *, labeler, options=None, counter=None)` returns:

| Key | Meaning |
|-----|---------|
| `input_tokens` | Tokens that the run sends: periods not in the cache, plus the reduce call. |
| `cached_tokens` | Tokens of the periods that are in the period cache. The run does not send them. |
| `output_tokens` | Reply tokens, from the options, not more than the output limits. |
| `cost_usd` | Cost of the calls at `ModelInfo` prices. The long-context rule applies to each call. |
| `periods` | Number of periods. |
| `periods_cached` | Number of periods in the cache. |

`labeler` is necessary: the speaker names are part of the cache key.
An empty scope gives zeros.

## Store

`ReportStore(root)` keeps one folder for each conversation and one JSON file for each run:

```
<root>/<conversation folder>/<report id>.json
```

- `save(report)` writes the file in one step. It raises `StoreError` when the id exists. It never replaces a report.
- `list_reports(conversation=None)` gives `ReportEntry` items, newest first.
- `load_report(conversation, report_id)` gives a `Report`, or raises `ReportNotFound`.
- `latest_report(conversation)` gives a `Report` or `None`.

`render_markdown(report)` gives the layout that the current Reports page and server read:

- Line 1: `# Conversation Analysis - <conversation id>`
- `## Overview`. Its first line is the scope line: `- Scope: 2024-06-01 to 2024-06-30`, `- Scope: from 2024-06-01`, `- Scope: through 2024-06-30` or `- Scope: all dates`.
- `## Key Events`. One `### <date> - <title>` block for each moment.
- `## Summary`. It is the last section and holds the four sections under `###` headings.

`write_text_atomic(path, text)` writes a text file in one step.

## Wiring in `ui_server.py`

This package does not change the server. The wiring step must do these things.

1. Make one `ModelInfo` for each model that the UI offers. Today the numbers are in `ai_config.py` and `cli.py`.
2. Make a labeler from `resolve_contact_names`. Return the contact's name for the empty sender of a one-to-one chat.
3. In `run_job`, action `analyze`, replace the call to `run_cli`:
   1. Make the source: `FrameSource` on the cached frame, or `ArchiveSource` when the archive is in place.
   2. Call `run_report`.
   3. Call `ReportStore.save`.
   4. Until the Reports page reads JSON, write the markdown file: `write_text_atomic(out_dir / f"analysis_{sanitize_filename(contact)}.md", render_markdown(report))`.
4. Make the events for `build_analysis_payload` from `report.moments`. The engine does not write `events_timeline_*.csv`.
5. Send `ProgressEvent` to the job progress. Keep one `CancelToken` for each job and add a route that calls `cancel()`.
6. Catch `ReportCancelled`, `ReduceError`, `AllPeriodsFailed`, `EmptyConversationError` and `ConfigError`. Show `report.warnings` when `status` is `partial`.
7. In `preview_payload`, replace `estimate_cost` with `estimate`.
8. Serve the JSON to the Reports page: the list from `list_reports`, one report from `load_report(...).model_dump(mode="json")`.

The engine does not write the HTML report of `html_report.py`.

## Tests

```bash
venv/bin/python -m pytest tests/report_engine --cov=report_engine --cov-report=term-missing
```

`tests/report_engine/conftest.py` sets a dummy `OPENAI_API_KEY` before any import and blocks all sockets that do not go to the local machine.
The tests use invented conversations and `FakeLLM` only.
