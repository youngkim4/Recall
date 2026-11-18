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

`ChatEngine` also takes two optional ports, by name: `meaning=` and `digests=`. See "Memory ports".
Without them the engine has the five tools of the table in "Tools" and nothing else changes.

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
- The scope also limits the two memory tools. See "Search by meaning" and "Conversation digest".

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
- The description of a search by meaning is masked before it goes to the port. The port can send it to a model provider.
- Every text of a digest (label, summary, title, detail, quote) is masked as message text is.

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
| `search_messages` with `match="meaning"` | Find exchanges by what they are about. Only with the port `meaning`. |
| `conversation_digest` | The story of one conversation, period by period. Only with the port `digests`. |

`tool_specs(meaning=..., digest=...)` in `tool_specs.py` gives the tool list of one engine.
`TOOL_SPECS` is the list of an engine with no memory port.
A call of a tool that is not offered gets the answer "There is no tool named ...".

Before the first model call, the engine runs one search on the words of the question.
It also looks up the names in the question. The results go into the prompt.
A simple recall question then ends in one model call.

A question that names years also gets the figures of that period before the first model call:
`count_messages` by conversation, from the first day of the first year to the last day of the last year.
A range such as `2016-2019` is a period, not a phone number.
Only a year that the archive holds is a period, so an amount such as 2000 is seldom taken for a year.
The digits of a phone number or of an email address are never a year. A handle that the labeler masks gives no year.
A period outside the days that the user chose runs no figures.
After the seeded lookups the status is "Thinking it over", so the line of the last lookup does not stay while the model works.

The prompt calls these lookups a first look. It tells two kinds of question apart:

- A question about words ("what did she say") is a search. A seeded result can answer it at once.
- A question about a person, a relationship, a period, a habit or a change is a pattern question.
  The model must not answer it from one message that holds the words of the question.
  It gets figures first, then reads the leading conversations at several points of the period, then answers from both.

Words go to the archive as the user or the model wrote them, in lower case.
The engine compares words without case or diacritics only inside its own code.

A piece of a query that is only signs, such as `-` or `&`, is dropped. The archive would look for it as plain text.

What `limit` of `search_messages` counts:

- `limit` is the number of matches, not the number of lines.
- Each match comes with up to 2 messages before it and 2 after it. Matches that are near each other share one block.
- So `limit=20` can show up to 100 messages. The token budget of the result still holds: when the blocks do not fit, a note says how many matches are not shown.

To read a conversation with no search word, the engine first counts the messages of the period.
A period with no message is empty at once. No page walk can prove that in a very long conversation.

## Memory ports

Chat finds messages by their words, and it reads small windows.
Two optional ports add a search by meaning and a view of a whole conversation.
Both are in `chat_engine/ports.py`. The engine imports no package that implements them.

```python
engine = ChatEngine(archive, llm, labeler, config, meaning=meaning_search, digests=digest_source)
```

| Port | Calls | Tool |
|---|---|---|
| `MeaningSearch` | `search(query, *, conversations, since, until, limit) -> list[MeaningHit]`, best first | `search_messages` with `match="meaning"` |
| `DigestSource` | `estimate(conversation_id, *, since, until) -> DigestCost` and `digest(conversation_id, *, since, until, cancel) -> DigestResult` | `conversation_digest` |

- A `MeaningHit` names one exchange: `conversation`, `first_message_id`, `last_message_id`, `score`.
- A `DigestResult` has `periods` and `periods_failed`. A `DigestPeriod` has `label`, `first_day`, `last_day`, `messages`, `summary` and `moments`.
- A `DigestMoment` has `date`, `title`, `detail`, `quote` and `message_ids`. The moments of a period are in order, the strongest first.
- `ChatArchive.message(message_id)` gives one message by its id. Both tools use it. A system row is not a message of the port.
- With a port missing, its tool is not in the tool list and the prompt does not name it. `tests/chat_engine/test_no_memory_ports.py` holds the tool list, the prompt and the requests of such an engine to commit `82f55e0`.
- A failure of a port never fails the question. The log gets the class of the failure only: no query, no name and no message text.
- `fake_meaning.FakeMeaning` and `fake_digests.FakeDigests` are scripted ports for tests. They record their calls.

### Search by meaning

`search_messages` takes `match="meaning"` when the engine has a `MeaningSearch`.
The query is then a short description of the exchange, such as "we made up after a fight".

The result joins two rankings of exchanges by reciprocal rank fusion (`fusion.py`):

1. The hits of `MeaningSearch.search`, in the order of the port.
2. The hits of the word search (`match="any"`) for the same query. A word hit counts for the exchange that holds it. A word hit outside every exchange of the first ranking is an exchange of its own: the messages around it.

- An exchange gets `1 / (60 + rank)` from each ranking that found it. The sums decide the order. An exchange that only one ranking found still counts.
- On equal sums the exchange with the better single rank comes first, then the exchange found by meaning.
- The engine reads ranks only. It does not compare the score of the port with the weight of a word.
- For each exchange that `limit` asks for, each ranking gives up to 3 hits, and never more than 60.
- `limit` counts exchanges. Every line has a reference from the evidence registry, as in a word search. A line that holds a word of the query has the mark `<<< match`.
- Two exchanges that share a message are shown as one.
- A long exchange is cut to fit the result budget: first its messages are cut shorter, then it loses messages, and a note says how many are shown. The engine reads at most 40 messages of one exchange.
- `from_me` does not work with `match="meaning"`. An empty query is refused.

Scope:

- The conversations and the days of the request go through `ToolContext.conversations` and `ToolContext.days`, as for every search. Both rankings get the same values.
- The engine checks every hit of the port again. A hit in another conversation, a hit that does not overlap the days, and a hit whose ends the archive does not hold in that conversation are dropped.
- A message of an exchange is shown only when `ToolContext.allows` allows it. An exchange that starts before the days of the scope shows only the part inside.

The seeded search uses the joined ranking when the engine has a `MeaningSearch`:

- The question, masked, is the query of the search by meaning. The words of the question are the query of the word search, as before.
- The prompt shows this lookup as `search_messages(match="meaning", query=the question)`.
- A question with no word to search for runs no seeded search.

When the port fails, the result holds the word ranking alone, and its first line says:
"The search by meaning was not available: these were found by their words only."

### Conversation digest

`conversation_digest(conversation, since, until)` gives the story of one conversation.
For each period it shows the label, the days, the number of messages, the summary, and the moments (day, title, detail, quote).

Evidence:

- The tool reads the messages of each moment from the archive with `message`, registers them as evidence, and shows them with their references. The answer can cite them.
- A message id that the archive does not hold is dropped without a note. So is a message outside the scope, and a message of another conversation.
- The tool shows at most 6 messages for one moment, each cut at 300 characters. `read_around` shows the whole of a message.
- A quote is shown only when a shown message of its moment holds it, by the rule of the quote check (`verify.is_quote_of`). So the digest never offers a quote that the answer check would flag.
- A summary is not a quote and not a citation. The prompt and the first line of the result say so.

Cost guard:

1. The tool calls `estimate` first. `estimate` builds nothing.
2. When `usd` (the cost of the periods that are not cached) is above `EngineConfig.digest_usd_limit`, the tool builds nothing. The model gets one line with the cost and the advice to narrow the days.
3. When `estimate` gives no period, the tool builds nothing and says that there is nothing to show.
4. In every other case the tool calls `digest`.
5. One question calls `digest` at most 3 times (`DIGESTS_PER_QUESTION`). Each call can cost up to the limit.

Budget:

- A digest has a budget of its own, `EngineConfig.digest_tokens`. It does not spend from `total_tool_tokens`.
- When the periods do not fit, every period keeps its title line. The rule is in `digest_fit.py`:
  1. The summaries come first. When they do not fit together, each summary is cut by the same share and no moment is shown.
  2. When the summaries fit, the moments take the room that is left. Moments are dropped one at a time from the weakest end of the period that still has the most.
- The last line of a digest that was cut says what was cut.
- A message of a moment that was dropped is not registered.

Other rules:

- Status line: "Reading your whole conversation with NAME" when periods must be built. "Recalling your conversation with NAME" when all are cached.
- The days of the call are cut down to the scope before they go to the source.
- A period that reaches outside the days of the scope is not shown, because its summary covers days outside the scope. A note says how many periods that is. A moment with a day outside the scope is dropped.
- A period in a wrong form counts as a period that failed. A note says how many periods are missing.
- The cancel token of the question goes to `digest`. After a cancel the tool reads nothing from the archive, and the question ends as any cancelled question does.
- When the source fails, the model gets one line: "The digest of ... could not be built now. Go on with the other tools."
- A second call for the same conversation and the same days gets the answer of a repeated lookup. The engine compares the conversation and the days, not the text of the arguments. The source is not asked again.
- When less than 300 tokens of `total_tool_tokens` are left, no lookup runs. This applies to a digest too.

The prompt, with `conversation_digest` offered: a pattern question about one person or one conversation starts with the figures, then the digest, then `read_around` on the moments that matter. With several people the model calls the digest for at most three conversations.

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
| `max_steps` | 8 | Model calls that can use tools. One more call must answer. |
| `max_calls_per_step` | 8 | The most lookups that one model call can start. More calls are not run and get no answer of their own: a note says how many. |
| `result_tokens` | 3000 | The largest tool result. |
| `total_tool_tokens` | 30000 | All tool results of one question. |
| `history_tokens` | 6000 | Earlier turns sent with a follow-up. `0` sends none. |
| `max_message_chars` | 4000 | The longest message text in a result and in a citation. |
| `digest_tokens` | 24000 | The largest conversation digest. A digest does not spend from `total_tool_tokens`. |
| `digest_usd_limit` | 2.0 | A digest that costs more to build is not built. `0` allows only a digest that is cached. |
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
The token also goes to `DigestSource.digest`, which can run for a long time. The source must stop when the token is cancelled.
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

The tests of the two memory tools use `FakeMeaning` and `FakeDigests`. `tests/chat_engine/memory_bench.py` builds a tool box with them.

`tests/guards.py` sets a dummy API key and blocks outbound sockets. `tests/chat_engine/conftest.py` also blocks loopback.
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

The server is wired: `/api/ask`, `/api/ask/stream` and `/api/ask/cancel` call `recall_services/chat.py`.
The old pipeline is removed (`ask_messages_payload` and its helpers, `retrieval_planner.py`, `semantic_index.py`).

`report_engine` has classes with the same names and other shapes: `OpenAIResponsesLLM`, `CancelToken`, `Scope`.
Import each one from its own package.

## Assumptions about the seams

Archive (`archive.queries.Archive`, read through `archive_adapter.py`):

- `archive_adapter.py` runs on a real archive in `tests/integration/test_chat_on_archive.py`. The tests of this package use a stand-in.
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

Memory ports (`ports.MeaningSearch`, `ports.DigestSource`). The adapters are not in this package:

- No adapter was run against the real index or the real digest. The tests use the two fakes.
- `MeaningSearch.search` returns the hits best first. The engine does not sort them and does not read `score`.
- `first_message_id` and `last_message_id` are archive ids (`Message["id"]`) of the same conversation, the first not later than the last.
- `since` and `until` are days, both included. A hit matches when its exchange overlaps them.
- `DigestSource.estimate` is cheap. The engine calls it while it checks the call, before the status line.
- `DigestCost.usd` is the cost of the periods that are not cached. It is a finite number, not negative.
- `DigestSource.digest` gets the `CancelToken` of `chat_engine/cancel.py`. `report_engine` has a class with the same name and another shape: the adapter must connect the two.
- A digest that is cancelled can raise or return. The engine looks at the token after the call.
- The moments of a period come the strongest first. The engine drops moments from the end.
- A summary can cover days outside the days of the call. So the engine does not show a period that reaches outside the days of the scope.

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
- An engine with no `MeaningSearch` has no search by meaning. The model can use `match="any"` with other words for the same thing.
- A search by meaning needs a word hit or a hit of the port. The engine does not rank an exchange again after the fusion.
- The digest tool trusts the order of the moments and the days of a period as the source gives them.
- No real model has read the two new tool descriptions or the new prompt rules. The tests use a scripted model.
- The engine reads message text only. It does not read attachments or reactions.
- The default token count is an estimate of three bytes for each token.
- Each question lists and labels every conversation again. The parsed handles are remembered, so this costs about 25 microseconds for each conversation.
- The list is not kept between questions. A contact or a group can get a new name with no new message, and then no count of the archive changes.
- A typed message that has the exact form of a reaction phrase is left out by `FrameArchive`.
- The volume numbers above come from invented text. Real text has more different words.
