# chat_engine

`chat_engine` answers a question about the message archive.
A model gets tools. It searches, reads and counts until it can answer.
The answer cites the messages it uses.

The package does not import `ui_server`. It does not read a `.env` file.
It contains no model id, no price and no context size. The caller passes them in.

## Run a question

```python
from chat_engine.archive_adapter import open_archive
from chat_engine.engine import ChatEngine, EngineConfig
from chat_engine.labeler import MaskingLabeler
from chat_engine.openai_llm import OpenAIResponsesLLM

archive = open_archive()                         # the default archive, read-only
labeler = MaskingLabeler(saved_names)            # {handle: saved name}
llm = OpenAIResponsesLLM(timeout_seconds=60)     # the key comes from OPENAI_API_KEY
engine = ChatEngine(archive, llm, labeler, EngineConfig(model=MODEL, max_output_tokens=2000))

result = engine.ask("When did we first talk about the lease?")
```

`result` is a dict:

| Key | Content |
|---|---|
| `answer` | The answer. References are `[1]`, `[2]`, ... in order of first use. |
| `citations` | The cited messages. Each is an archive `Message` plus `number`, `reference`, `conversation_label`, `sender_label`. |
| `unverified_quotes` | Quoted spans of 12 characters or more that no shown message contains. |
| `invalid_references` | References the model wrote that no shown message has. The engine removes them from the answer. |
| `steps` | One entry for each lookup: `{"tool", "text", "seeded"}`. `text` says in plain words what the lookup did. |
| `usage` | `input_tokens`, `output_tokens`, `cached_input_tokens`, `reasoning_tokens`, `model_calls`, `tool_calls`. |
| `cancelled` | `True` when a cancel stopped the question. `answer` is then the text written so far. |
| `truncated` | `True` when the model reached the output cap. |
| `mode` | `"ai"`. The local answer has `"local"`. |

## Stream it

Use `ask` with a callback, or `stream` as an iterator. Both give the same events.

```python
result = engine.ask(question, on_event=send)     # raises a ChatEngineError on failure

for event in engine.stream(question):            # never raises a ChatEngineError
    send(event)                                  # the last event is "done" or "error"
```

| Event | Shape |
|---|---|
| `status` | `{"type": "status", "text": "Searching for “lease”"}`. Optional `"reset": true`. |
| `delta` | `{"type": "delta", "text": "..."}`. The deltas, joined, are the final `answer`. |
| `done` | `{"type": "done", "payload": result}` |
| `error` | `{"type": "error", "code": "rate_limit", "error": "..."}`. Optional `"retry_after"` in seconds. |

When a `status` event has `"reset": true`, discard the text that the deltas gave so far.
The model wrote a note before a lookup. That note is not part of the answer.

## Keep a chat

Pass the same `ChatSession` to every question of one chat.

```python
from chat_engine.session import ChatSession

session = ChatSession()
engine.ask("What did Jonas name his starter?", session=session)
engine.ask("When was that?", session=session)    # can cite and re-read the earlier evidence
```

- The session keeps the earlier turns and the evidence registry.
- A reference such as `m3` means the same message for the whole chat.
- `session.to_dict()` gives JSON. `ChatSession.from_dict(data)` restores it.
- `ChatSession.from_history([{"role", "content"}, ...])` restores the turns only.
- Do not use one session for two questions at the same time.

## Scope

A scope limits a question to conversations, to days, or to both.

```python
from chat_engine.scope import Scope

engine.ask(question, scope=Scope(conversations=(conversation_id,)))
engine.ask(question, scope=Scope(since="2024-01-01", until="2024-12-31"))
```

Every tool reads only inside the scope. The model cannot widen it.
`chat_engine/tool_context.py` enforces the scope in one place.

## Names and privacy

The model never reads a conversation id or a handle.
It reads an alias (`c3`) and a label.

- A saved name is used as it is.
- A person who is not saved is "a number ending in 1234" or "an unsaved email contact".
- The engine also masks such a handle inside message text and inside the question.

`MaskingLabeler(saved_names, known_handles=...)` supplies the masks.
The labeler masks a handle in free text only when it knows the handle.
It learns each handle that it labels. Pass `known_handles` to make it know all handles from the start.

The model provider receives the question, the tool results and the labels.
A tool result contains the text of the messages that match, and of the messages near them.

## Tools

| Tool | Use |
|---|---|
| `search_messages` | Find messages by words. `match="any"` ranks by rare words. An empty `query` reads one conversation from its start or its end. |
| `read_around` | Read the messages before and after one message. |
| `count_messages` | Count exactly, with or without groups. |
| `list_conversations` | Find conversations and people by name or by digits of a number. |
| `conversation_stats` | Figures for one conversation. |

Before the first model call, the engine runs one search on the words of the question.
It also looks up the names in the question. The results go into the prompt.
A simple recall question then ends in one model call.

## Limits

`EngineConfig` holds the limits. `model` and `max_output_tokens` have no default.

| Field | Default | Meaning |
|---|---|---|
| `max_steps` | 6 | Model calls that can use tools. One more call must answer. |
| `result_tokens` | 3000 | The largest tool result. |
| `total_tool_tokens` | 18000 | All tool results of one question. At this limit the model must answer. |
| `history_tokens` | 6000 | Earlier turns sent with a follow-up. The last turn is always sent. |
| `max_message_chars` | 4000 | The longest message text in a result. A cut is marked. |
| `timezone` | `""` | The name in the prompt. Empty uses the zone of the clock. |
| `count_tokens` | estimate | Pass a tokenizer for exact budgets. |

## Errors

Each failure is one class in `chat_engine/errors.py`. Each has a `code`.

| Class | `code` |
|---|---|
| `NoApiKeyError` | `no_key` |
| `KeyRejectedError` | `key_rejected` |
| `UnknownModelError` | `unknown_model` |
| `RateLimitedError` | `rate_limit` |
| `OfflineError` | `offline` |
| `ModelTimeoutError` | `timeout` |
| `ModelError` | `model_error` |
| `ArchiveError` | `archive_error` |

`error.message` is safe to show. `error.detail` is for the local log only.
The engine never changes to a different way to answer.

`local_answer(archive, question, scope, labeler=labeler)` gives an answer from keyword search only.
Call it yourself when the model call fails. Tell the user that no AI model was used.

## Cancel

```python
from chat_engine.cancel import CancelToken

token = CancelToken()
engine.ask(question, cancel=token, on_event=send)   # in the request thread
token.cancel()                                      # from another thread
```

A cancel stops the model stream and the tool loop.
The question ends with a `done` event. Its payload has `"cancelled": true`.
A cancelled question is not added to the session.
If you close the iterator of `engine.stream`, the model call also stops.

## Evaluation

`chat_engine/eval/` contains a synthetic archive and 31 questions with expected answers.
All people in it are invented.

```bash
venv/bin/python -m chat_engine.eval.runner --scripted
```

This command uses a scripted model. It needs no key and no network.

WARNING: The command below sends the text of the synthetic archive to the model provider and costs money.
Only the owner runs it, with their own key.

```bash
venv/bin/python -m chat_engine.eval.runner --model MODEL --max-output-tokens 2000
```

The runner prints scores only. It prints no question, no answer and no message text.
`--env-file PATH` reads the key from one named file. `--only id,id` selects questions.

## Tests

```bash
venv/bin/python -m pytest tests/chat_engine --cov=chat_engine --cov-report=term-missing
```

`tests/chat_engine/conftest.py` sets a dummy API key and blocks outbound sockets.
No linter and no type checker is configured in this repository.

## What the wiring step must do

In `ui_server.py`:

1. Open the archive one time with `open_archive()`.
2. Build one `MaskingLabeler` from the saved contact names. Pass all handles of the archive as `known_handles`.
3. For each request, build `ChatEngine` with `EngineConfig(model=..., max_output_tokens=...)`. The model id comes from the request.
4. In `handle_ask_stream`, replace the call to `ask_messages_payload` with `engine.stream(...)`. Write each event as one `data:` frame.
5. Keep the event types `status`, `delta`, `done`, `error`.
6. Map `contact` from the request to `Scope(conversations=(contact,))`.
7. Keep one `ChatSession` for each chat id, in memory, with a limit on their number. When the chat id is new, use `ChatSession.from_history(history)`.
8. Add a cancel path. Keep one `CancelToken` for each running question. Add a route that calls `token.cancel()`. Call `token.cancel()` also when a write to the client fails.
9. For `/api/ask` without a stream, call `engine.ask(...)`.
10. When the request sets `ai` to false, call `local_answer`.

In `app/`:

1. Read `code` from the `error` event. Show a message for each code. Offer the local answer as a clear choice.
2. Clear the streamed text when a `status` event has `reset`.
3. Stop the request with an `AbortController` and call the cancel route.
4. Read the new payload. `citations[i]` maps to the old `SearchResult` in this way:
   `messageId` = `id`, `chatId` = `conversation`, `displayName` = `conversation_label`,
   `senderName` = `sender_label`, `isFromMe` = `is_from_me`. `timestamp`, `sender` and `text` keep their names.
5. Show `unverified_quotes` as a warning near the answer.
6. Send the chat id with each question.

After the wiring, these parts of the old pipeline have no caller:
`ask_messages_payload`, `ai_ask_answer`, `query_terms`, `_burst_scores`, `build_context_windows`,
`format_context_windows`, `_dedupe_moments`, `_renumber_citations`, `retrieval_planner.py`.
The engine does not use `semantic_index.py`.

## Assumptions about the seams

Archive (`archive.queries.Archive`, read through `archive_adapter.py`):

- `archive_adapter.py` is not tested against the real package. The package is not in the base of this branch.
- `open_archive()` calls `Archive.open(path, readonly=True)`.
- `messages()` pages by message id. `around` includes the anchor. `before` and `after` exclude it.
- `messages()` with no cursor gives one page at one edge of the conversation. The engine works with either edge.
- `search("")` gives no hit. To read a conversation without a word, the engine uses `messages()` (see `browse.py`).
- `count(group_by="sender")` uses `""` as the key for the owner. `weekday_hour` keys are `W-HH` with Monday as 0.
- `stats()` uses the field names of `ConversationStats`. The engine never shows its `conversation` field.
- `Message["attachments"]` holds mime types. The engine shows only the kind, such as "image".
- `Conversation["members"]` and `Message["sender"]` are handles. `Conversation["name"]` is a display name or empty.
- Conversations with the same `person_id` belong to one person.
- The engine does not use `people()`, `person()`, `iter_messages()` or `reactions()`.

Model port (`ports.LLM`):

- One call streams `TextDelta`, `ToolCall` and one `Completed` with the usage.
- `Completed.provider_items` goes back to the same adapter on the next call of the same question.
- `OpenAIResponsesLLM` is tested against a stand-in for the SDK client. No request was sent to the provider.
- It sends reasoning items back only as encrypted content. Without them it removes item ids from the function calls.

Events:

- `error` uses the key `error` for the message, as the old stream did. `code` is new.
- `status` can have `reset`. `done.payload` has more keys than the old `AskResponse` and no `terms`.

## Known limits

- The seeded search splits words at spaces. A language that writes no spaces gets a weak seeded search. The model can still search.
- The search has no meaning search. The model can use `match="any"` with other words for the same thing.
- The engine reads message text only. It does not read attachments or reactions.
- The default token count is an estimate of three bytes for each token.
