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
llm = OpenAIResponsesLLM(timeout_seconds=60, reasoning_effort=EFFORT)   # the key comes from OPENAI_API_KEY
engine = ChatEngine(archive, llm, labeler, EngineConfig(model=MODEL, max_output_tokens=2000))

result = engine.ask("When did we first talk about the lease?")
```

Give `reasoning_effort` only for a model that reasons.
The adapter then asks for the encrypted reasoning and sends it back in the next call of the question.
A model that does not reason refuses that request.

`result` is a dict:

| Key | Content |
|---|---|
| `answer` | The answer. References are `[1]`, `[2]`, ... in order of first use. |
| `citations` | The cited messages. Each is an archive `Message` plus `number`, `reference`, `conversation_label`, `sender_label`, `text_truncated`. |
| `unverified_quotes` | Quoted spans of 12 characters or more that no shown message contains. |
| `invalid_references` | References the model wrote that it cannot cite. The engine removes them from the answer. |
| `steps` | One entry for each lookup: `{"tool", "text", "seeded"}`. `text` says in plain words what the lookup did. |
| `usage` | `input_tokens`, `output_tokens`, `cached_input_tokens`, `reasoning_tokens`, `model_calls`, `tool_calls`. |
| `cancelled` | `True` when a cancel stopped the question. `answer` is then the text written so far. |
| `truncated` | `True` when the model reached the output cap. |
| `mode` | `"ai"`. The local answer has `"local"`. |

Notes on the result:

- The `text` of a citation holds at most `max_message_chars` characters. `text_truncated` says when it was cut.
- A citation holds the raw `sender` and `conversation` of the archive. Map them to names before a screen shows them.
- The model can cite only a message that a lookup showed, inside the scope of the question.
- A number in square brackets without "m", such as `[3]`, is not a citation. The answer shows it as `(3)`.

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
| `delta` | `{"type": "delta", "text": "..."}`. The deltas after the last reset, joined, are the final `answer`. |
| `done` | `{"type": "done", "payload": result}` |
| `error` | `{"type": "error", "code": "rate_limit", "error": "..."}`. Optional `"retry_after"` in seconds. |

When a `status` event has `"reset": true`, discard the text that the deltas gave so far.
The model wrote a note before a lookup. That note is not part of the answer.

An empty question raises `ValueError` at the call of `ask` or `stream`.

## Keep a chat

Pass the same `ChatSession` to every question of one chat.

```python
from chat_engine.session import ChatSession

session = ChatSession()
engine.ask("What did Jonas name his starter?", session=session)
engine.ask("When was that?", session=session)    # can cite and re-read the earlier evidence
```

- The session keeps the earlier turns, the scope of each turn, and the evidence registry.
- A reference such as `m3` means the same message for the whole chat.
- A question that does not finish leaves the session as it was. This applies to a cancel, to a failure and to a closed stream.
- A question that was left open is rolled back when the next question starts. A stream that is closed late changes nothing.
- The session stores each answer without the references that the answer could not cite.
- `session.to_dict()` gives JSON. `ChatSession.from_dict(data)` restores it. Data that is not a session raises `ValueError`.
- `ChatSession.from_history(rows, scope=scope)` restores the turns only. Give the scope of the chat. The numbers `[1]`, `[2]` of the shown answers are removed.
- Do not use one session for two questions at the same time.

WARNING: `session.to_dict()` holds the whole text of every message that the model read. Keep it on this machine.

WARNING: Message ids change when the archive is built again from another source. Discard the stored sessions then.

## Scope

A scope limits a question to conversations, to days, or to both.

```python
from chat_engine.scope import Scope

engine.ask(question, scope=Scope(conversations=(conversation_id,)))
engine.ask(question, scope=Scope(since="2024-01-01", until="2024-12-31"))
```

- Every tool reads messages only inside the scope. The model cannot widen it.
- `chat_engine/tool_context.py` enforces the scope in one place.
- An earlier answer of the chat goes to the model only when its scope lies inside the scope of the new question. If not, a note takes its place. The earlier questions stay.
- A message outside the scope is never a citation and never a source for a quote.
- The engine lists the conversations of the whole archive one time for each question. It needs the handles to mask them. It reads no message there, and the list does not go to the model.

## Names and privacy

The model never reads a conversation id or a handle.
It reads an alias (`c3`) and a label.

- A saved name is used as it is.
- A person who is not saved is "a number ending in 1234" or "an unsaved email contact". These are the words of `contact_label` in `ui_server.py`.
- The engine masks such a handle also inside message text, inside the question and inside the earlier turns.

How the masking works:

- Before any text goes out, the engine shows the labeler every conversation and every member of the archive.
- `MaskingLabeler.redact` then masks each handle of a person who is not saved, in any spelling.
- Spellings of a number that it knows: with or without the country code, a `0` after the country code, digits of another script, any mark or a line break between the groups.
- The local form of seven digits is masked when it stands as a number of its own: `555-0199`, `5550199`, also in "room 4, 555-0199".
- An address is masked next to any character: before a letter of another script, between underscores, before `.Thanks`, `-thanks` or a digit.
- An address of the archive in an unusual form (with an apostrophe, with letters of another script) is masked as it is written.
- A number or an address that a cut would split is dropped at the cut.
- All text for the model passes one function: `QuestionTurn._outbound` in `turn.py`.

Limits of the masking:

- A handle of a person who is in no conversation of the archive is plain text. The engine does not know it.
- A saved number is not masked. A local form of seven digits is not masked when a saved number has the same seven digits.
- Seven digits inside a longer number with another area code are another number. They are not masked. A date such as `3/14/2024` is not a number.
- If you write your own `Labeler`, its `redact` must do the same work. See `ports.Labeler`.

The model provider receives the question, the tool results and the labels.
A tool result contains the text of the messages that match, and of the messages near them.

## Tools

| Tool | Use |
|---|---|
| `search_messages` | Find messages by words. `match="any"` ranks by rare words. An empty `query` reads one conversation from its start or its end. |
| `read_around` | Read the messages before and after one message. |
| `count_messages` | Count exactly, with or without groups. |
| `list_conversations` | Find conversations and people by name, by the last four digits of a number, or by the whole number. |
| `conversation_stats` | Figures for one conversation. |

Before the first model call, the engine runs one search on the words of the question.
It also looks up the names in the question. The results go into the prompt.
A simple recall question then ends in one model call.

Words go to the archive as the user or the model wrote them, in lower case.
The engine compares words without case or diacritics only inside its own code.

A piece of a query that is only signs, such as `-` or `&`, is dropped. The archive would look for it as plain text.

What `limit` of `search_messages` counts:

- `limit` is the number of matches, not the number of lines.
- Each match comes with up to 2 messages before it and 2 after it. Matches that are near each other share one block.
- So `limit=20` can show up to 100 messages. The token budget of the result still holds: when the blocks do not fit, a note says how many matches are not shown.

To read a conversation with no search word, the engine first counts the messages of the period.
A period with no message is empty at once. No page walk can prove that in a very long conversation.

## Reactions and attachments

- A reaction is not a message. Search, counts, lists and figures do not see it.
- The real archive keeps reactions apart from the messages. The engine does not read them.
- `FrameArchive` reads an export, where a reaction is a row of text such as `Loved “the text”`. It leaves such a row out when it loads the file.
- The rule for such a row: the text starts with Loved, Liked, Disliked, Laughed at, Emphasized or Questioned, then a quote mark, and it ends with a quote mark.
- A row that starts in this way but does not end with a quote mark stays a message. The volume run counts these rows.
- A message that is only an attachment has no text. The model reads only the kind, such as `[attachment: image]`.

## Limits

`EngineConfig` holds the limits. `model` and `max_output_tokens` have no default.

| Field | Default | Meaning |
|---|---|---|
| `max_steps` | 6 | Model calls that can use tools. One more call must answer. |
| `max_calls_per_step` | 8 | The most lookups that one model call can start. More calls are not run and get no answer of their own: a note says how many. |
| `result_tokens` | 3000 | The largest tool result. |
| `total_tool_tokens` | 18000 | All tool results of one question. |
| `history_tokens` | 6000 | Earlier turns sent with a follow-up. `0` sends none. |
| `max_message_chars` | 4000 | The longest message text in a result and in a citation. |
| `timezone` | `""` | The name in the prompt. Empty uses the zone of the clock. |
| `count_tokens` | estimate | Pass a tokenizer for exact budgets. |

The limits are hard:

- A message text longer than the cap keeps its start. A note says how many characters are missing.
- When a search match lies past the start, the part around the match is shown too.
- A tool result never counts more tokens than its budget. Notes that say what is left out always fit. A refusal is cut to the budget too.
- A mask is longer than the handle it stands for. If a result passes its budget after the masking, the engine writes it again. If it still does not fit, the model gets a short refusal and no message stays registered.
- The notes for lookups that were not run count as tool output.
- A lookup runs only when 300 tokens of `total_tool_tokens` are left. With less, the model must answer.
- The quote check uses the whole stored text of a message, also the part that a cap cut away.

## Quotes

- Every quoted span of 12 characters or more must be in one shown message, or be a name.
- A quote with an elision ("the start ... the end") must have all its parts in one message, in their order, at most 400 characters apart.
- The words of the user are not a source. A quote of the question is flagged too.
- The engine does not change the answer. It lists the spans in `unverified_quotes`, without the references that stand inside them.
- A quote of more than 2,000 characters is not checked.

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

A cancel stops the model stream and the tool loop. After a cancel, the engine reads nothing more from the archive.
The question ends with a `done` event. Its payload has `"cancelled": true`.
A cancelled question is not added to the session, and its evidence is forgotten.
If you close the iterator of `engine.stream`, the model call also stops.

## An export in place of the archive

`FrameArchive` answers the calls of the archive port from a message export (`messages.csv`).

```python
from chat_engine.frame_archive import FrameArchive

archive = FrameArchive.from_csv(path)
```

- Use it to run the chat tools before the real archive exists, and for runs at full volume.
- It follows the rules of the archive contract. The tests compare it with the fake archive call by call.
- It keeps every text whole. Its search looks at the first 10,000 characters of a text, as the real archive does.
- A term in a script that is written without spaces, or with a sign in it, is matched as plain text.
- `sent_at` is the local time read as UTC. `reply_to`, `edited` and `unsent` are empty. `person_id` is `None`.
- The members of a group are the people who wrote in it.
- `archive.dropped` counts the rows that are not messages: reaction phrases, rows that cannot be read, second rows of one id.

## Volume

```bash
venv/bin/python -m chat_engine.eval.volume --invented 565821
```

This command makes an invented export of that many rows and measures every tool on it.
It prints numbers and fixed labels only. It blocks the network. It calls no AI model.

Results on this machine for 565,821 invented rows (548,543 messages, 1,414 conversations, one text of 2.9 million characters):

| Measure | Result |
|---|---|
| Build the archive from the frame | 1.5 s |
| Build the search index | 4.0 s |
| Memory at its peak | 479 MB |
| Read and label every conversation, once for each question | 34 ms |
| Slowest of 45 lookups | 40 ms (count by sender over the whole archive) |
| A search | 2 to 11 ms |
| A whole question without the model time | 45 to 57 ms |
| Largest tool result | 2,932 tokens of 3,000 |
| Lookups over the budget | 0 |
| Lookups that failed | 0 |

WARNING: The command below reads a real export. Run it only on your own file. It prints numbers only.

```bash
venv/bin/python -m chat_engine.eval.volume /path/to/messages.csv
```

The session that wrote this package did not read a real export: its permission check refused that.
Another session of the owner ran the command on the real export and gave back the numbers.
That run found one fault, which is now fixed and has a test: a period before the first message of a very long conversation.
The numbers of that run are in the hand-over note of the sessions. They are not in this repository.

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
venv/bin/python -m chat_engine.eval.runner --model MODEL --max-output-tokens 2000 --reasoning-effort EFFORT
```

The runner prints scores only. It prints no question, no answer and no message text.
`--env-file PATH` reads the key from one named file. `--only id,id` selects questions.

What a score checks:

- A number counts only as a whole number. A name counts only at the start of a word. A day counts only with its year.
- "Not found" needs a plain statement and no citation.
- No raw handle in any spelling. No flagged quote. No invalid reference.
- A score does not prove which number the answer gives to which side. Read the answers of a real run before you trust it.

## Tests

```bash
venv/bin/python -m pytest tests/chat_engine --cov=chat_engine --cov-report=term-missing
```

`tests/chat_engine/conftest.py` sets a dummy API key and blocks outbound sockets.
No linter and no type checker is configured in this repository.

## What the wiring step must do

In `ui_server.py`:

1. Open the archive one time with `open_archive()`.
2. Build one `MaskingLabeler` from the saved contact names. The engine shows it the other handles.
3. For each request, build `ChatEngine` with `EngineConfig(model=..., max_output_tokens=...)`. The model id comes from the request.
4. Build `OpenAIResponsesLLM` with `reasoning_effort` only when the model reasons (`supports_reasoning_controls`).
5. In `handle_ask_stream`, replace the call to `ask_messages_payload` with `engine.stream(...)`. Write each event as one `data:` frame.
6. Keep the event types `status`, `delta`, `done`, `error`.
7. Map `contact` from the request to `Scope(conversations=(contact,))`.
8. Keep one `ChatSession` for each chat id, in memory, with a limit on their number. When the chat id is new, use `ChatSession.from_history(history, scope=scope)`.
9. Discard the sessions when the archive is built again.
10. Add a cancel path. Keep one `CancelToken` for each running question. Add a route that calls `token.cancel()`. Call `token.cancel()` also when a write to the client fails.
11. For `/api/ask` without a stream, call `engine.ask(...)`.
12. When the request sets `ai` to false, call `local_answer`.
13. Add `question` and `contact` to the payload if the screen needs them. The engine has no `terms`.

Map each citation to the shape of a search result (`message_row_payload` at commit `ab20b5e`):

| Screen field | From the citation |
|---|---|
| `messageId` | `id` |
| `chatId` | `conversation` |
| `timestamp` | `timestamp`. It is exact, so `/api/day` can open the day at this message. |
| `text` | `text`. The server clips it to its own length. |
| `isFromMe` | `is_from_me`. It is `true` or `false`, never `null`. |
| `attachment` | `attachment_label` of `attachments`, joined with a vertical bar. |
| `displayName` | The name the server gives to `conversation`. Do not use `conversation_label`: it can be a mask for the model. |
| `senderName` | The name the server gives to `sender`. Do not send the raw `sender` to a screen. |

In `app/`:

1. Read `code` from the `error` event. Show a message for each code. Offer the local answer as a clear choice.
2. Clear the streamed text when a `status` event has `reset`.
3. Stop the request with an `AbortController` and call the cancel route.
4. Show `unverified_quotes` as a warning near the answer.
5. Send the chat id with each question.

After the wiring, these parts of the old pipeline have no caller:
`ask_messages_payload`, `ai_ask_answer`, `query_terms`, `_burst_scores`, `build_context_windows`,
`format_context_windows`, `_dedupe_moments`, `_renumber_citations`, `retrieval_planner.py`.
The engine does not use `semantic_index.py`.

`report_engine` has classes with the same names and other shapes: `OpenAIResponsesLLM`, `CancelToken`, `Scope`.
Import each one from its own package.

## Assumptions about the seams

Archive (`archive.queries.Archive`, read through `archive_adapter.py`):

- `archive_adapter.py` is not tested against the real package. The package is not in the base of this branch.
- `open_archive()` calls `Archive.open(path, readonly=True)`. `path=None` opens the default archive.
- One archive object serves all request threads.
- The adapter passes `include_system=False` by name in every call that takes it.
- `messages()` pages by message id. `around` includes the anchor. `before` and `after` exclude it. With no cursor the page is the newest one.
- `search("")` gives no hit. To read a conversation without a word, the engine uses `iter_messages(conversation_id, since=, until=)` when the archive has it. If not, it walks pages (see `browse.py`). The adapter has a reader only when the archive has one.
- The engine does not use `messages(oldest=True)` or `day()`.
- Search looks at the first 10,000 characters of a text. The ranking of the engine reads the same part.
- `count(group_by="sender")` uses `""` for the owner and `"unknown"` for a received message with no handle. `weekday_hour` keys are `W-HH` with Monday as 0.
- `stats()` uses the field names of `ConversationStats`. The engine never shows its `conversation` field.
- `Message["id"]` is the guid of the message. `reply_to` is the guid of the first message of the thread. The engine shows "(in the thread that starts at m3)".
- `Message["attachments"]` holds mime types. The engine shows only the kind, such as "image".
- `Conversation["members"]` and `Message["sender"]` are handles. `Conversation["name"]` is a display name or empty.
- `conversations(limit=None, offset=0, order="count")` lists every conversation that has a message.
- Messages are ordered by `sent_at`, then by the local time.
- The engine does not use `people()`, `person()` or `reactions()`.

Model port (`ports.LLM`):

- One call streams `TextDelta`, `ToolCall` and one `Completed` with the usage.
- `Completed.provider_items` goes back to the same adapter on the next call of the same question.
- `OpenAIResponsesLLM` is tested against a stand-in for the SDK client. No request was sent to the provider.
- It sends reasoning items back only as encrypted content. Without them it removes item ids from the function calls.
- A function call that the output cap cut is not run.

Screens (redesign commit `ab20b5e`):

- `askStream` in `app/src/lib/api.ts` reads `status`, `delta`, `done` with a payload, and `error`. The engine gives these four.
- New in the events: `status.reset`, `error.code`, `error.retry_after`. The screen ignores them until the wiring adds them.
- `done.payload` has no `question`, no `contact` and no `terms`. It has more keys than the old `AskResponse`.
- The "Names" rules of `CONTRACTS.md` hold for the model: it never reads a handle that is not saved.
- The answer text and the step texts use the masks of the model, such as "a number ending in 0199". No full number is in them.
- The labels of the screen (`conversation_label` in `ui_server.py`) use other words. The table above says which field comes from where.

## Known limits

- The seeded search splits words at spaces. A language that writes no spaces gets a weak seeded search. The model can still search, and the archive matches such a term as plain text.
- The search has no meaning search. The model can use `match="any"` with other words for the same thing.
- The engine reads message text only. It does not read attachments or reactions.
- The default token count is an estimate of three bytes for each token.
- Each question lists and labels every conversation again. The parsed handles are remembered, so this costs about 25 microseconds for each conversation.
- The list is not kept between questions. A contact or a group can get a new name with no new message, and then no count of the archive changes.
- A typed message that has the exact form of a reaction phrase is left out by `FrameArchive`.
- The volume numbers above come from invented text. Real text has more different words.
