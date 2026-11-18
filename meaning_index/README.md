# meaning_index

`meaning_index` finds messages by their meaning.

It cuts each conversation into segments. It gets one vector for each segment from an embedding model. It keeps the vectors on disk. It gives the segments that are nearest to a query vector.

The package does not answer questions. It does not import `chat_engine`, `report_engine`, `recall_services` or `ui_server`. It holds no text of a message: the archive holds the messages, and a segment names its first and last message.

```python
from archive import Archive
from meaning_index import ArchiveSource, IndexStore, MeaningIndex, build_index, embedder_facts, estimate_build
from meaning_index.openai_embedder import OpenAIEmbedder
from report_engine.openai_llm import build_client

facts = embedder_facts()                      # text-embedding-3-large at 1024 dimensions
store = IndexStore(saves_dir / "meaning")
embedder = OpenAIEmbedder(build_client(api_key), facts.model, facts.dimensions)

with Archive.open(archive_path) as archive:
    source = ArchiveSource(archive)
    quote = estimate_build(source, store, facts)          # BuildEstimate: no request, no file
    result = build_index(source, embedder, store, progress=show, cancel=token)

index = MeaningIndex.load(store)              # None when there is no index
(vector,) = embedder.embed(["when did we plan the ferry trip"])
hits = index.search(vector, conversations=[chat_id], since="2024-06-01", limit=20)
```

## Rules

- No text of a message, no name, no handle, no conversation id and no message id is in a log line or in an error.
- Every failure is a subclass of `MeaningIndexError`, or a `ValueError` or `TypeError` for an argument that is wrong.
- Every data shape is a frozen dataclass.
- A reader makes no file. Only a build writes.
- The import of `meaning_index` loads no SDK of a provider and no tokenizer.

## Ports

The package needs three things from outside. Each is a parameter.

| Port | Shape | Given by |
| --- | --- | --- |
| `MessageSource` | `conversation_ids() -> Sequence[str]`, `messages(conversation_id) -> Iterator[SourceMessage]` (oldest first) | `ArchiveSource(archive)` |
| `Embedder` | `model: str`, `dimensions: int`, `embed(texts) -> Sequence[Sequence[float]]` (one vector for each text, same order) | `OpenAIEmbedder`, `FakeEmbedder` |
| `count_tokens` | `Callable[[str], int]` | `default_count_tokens`, or a counter of the app |

`SourceMessage` has four fields: `id`, `timestamp` (`YYYY-MM-DD HH:MM:SS`, local time), `text`, `is_from_me`.

### ArchiveSource

`ArchiveSource(archive)` adapts an open `archive.Archive`. The caller opens and closes the archive.

- It reads the archive with `conversations()` and `iter_messages(conversation_id)`.
- It leaves out group events (kind `system`).
- It leaves out each message whose text is empty or only white space: an attachment with no caption, a message that was unsent.
- A message in no conversation is never read.
- A row that breaks the contract of the archive is a `SourceError`.

### OpenAIEmbedder

`OpenAIEmbedder(client, model, dimensions)` sends `client.embeddings.create(model=..., input=[...], dimensions=...)`.

- The caller gives the client. The app builds it with `report_engine.openai_llm.build_client`. That client has the address of the provider, takes no proxy from the environment, and does not try a request again.
- One call of `embed` is one request. The build does the waiting and the halving.
- Each failure of the SDK becomes one typed failure:

| Failure of the service | Error |
| --- | --- |
| HTTP 401 or 403 | `EmbeddingKeyError` |
| HTTP 429, not `insufficient_quota` | `EmbeddingRateLimited`, with `retry_after_s` |
| HTTP 413, or HTTP 400 that says the request is too large | `EmbeddingTooLarge` |
| no connection, or no answer in time | `EmbeddingNetworkError` |
| every other failure, `insufficient_quota` too | `EmbeddingServiceError` |
| a reply that does not hold one vector for each text | `EmbeddingReplyError` |

- `retry_after_s` comes from the header `retry-after-ms`, then from `retry-after` as seconds, then from `retry-after` as a date.
- An error names the HTTP status, the error code of the service and its request id. It never holds the message of the service. The SDK error holds the request, so it is never chained.

### FakeEmbedder

`FakeEmbedder(dimensions=64, model="fake-embedding-1", synonyms=None)` calls no service. A vector is a hashed bag of words. A small table maps words that mean the same to one word (`boat` and `ferry`, `supper` and `dinner`). Tests use it to rank. `calls` holds the texts of each call.

## Segments

`cut_segments(conversation_id, messages) -> Iterator[CutSegment]`

- A segment is a run of consecutive messages of one conversation.
- A new segment starts after a gap of more than 120 minutes.
- A new segment starts when the segment holds 16 messages.
- A new segment starts when the text of the segment reaches 1,600 characters.
- A message whose text is empty or only white space is in no segment. It does not count.

The text to embed:

```
2024-06-03
Them: are we on for the lake
Me: yes, bring the canoe
```

- The first line is the local day of the first message.
- Each message is one line: `Me: ...` or `Them: ...`. Line breaks and runs of white space in a message become one space.
- A line is cut at 400 characters. The label is part of the 400.
- No name and no handle is added.

Ids:

- `segment_id` is the SHA-256 of the conversation id and the message ids. It does not change when a text is edited.
- `content_hash` is the SHA-256 of the text to embed, the model and the dimensions. It decides whether a vector can be used again.

A text is 2,000 characters at most: 1,599 before the last line, a line break, and a line of 400. A token is one byte or more, and a character is four bytes at most. Thus one text is 8,000 tokens at most. That is below the 8,192 tokens that the two models take for one input.

WARNING: A change to a rule of this section changes the ids and the content hashes. The next build then embeds every segment again.

## Store

`IndexStore(folder)` keeps the index in one folder. The caller gives the path. The app uses `recall_paths.SAVES_DIR / "meaning"`.

| File | Content |
| --- | --- |
| `segments.sqlite3` | Table `segments`: `segment_id`, `conversation_id`, `first_message_id`, `last_message_id`, `first_timestamp`, `last_timestamp`, `message_count`, `content_hash`, `row`. Table `build`: the id of the build. |
| `vectors.npy` | N by D, float16, each row of length 1. Row `n` belongs to the segment with `row` `n`. |
| `meta.json` | `format_version`, `build_id`, `model`, `dimensions`, `built_at` (UTC), `segments`, `covered_until` (the last time stamp of a segment, local time). |
| `work.sqlite3` | The vectors that a build paid for, by `content_hash`. It is there only after a build that did not end. |
| `build.lock` | The lock of a build. It is empty and it stays. |

- Each file has mode 0600. A folder that the store makes has mode 0700.
- `segments.sqlite3` holds conversation ids and message ids. It holds no text.
- `store.status()` gives `IndexStatus(state, segments, model, dimensions, built_at, covered_until)`. `state` is `none` or `ready`.
- `store.read()` gives the segments in row order and the vectors, or `None`.

### How a build replaces the files

1. The build takes the lock. A second build of the same folder at the same time gets `IndexBusyError`.
2. The build writes `vectors.npy.new`, `segments.sqlite3.new` and `meta.json.new`.
3. The build keeps each old file as a hard link with the ending `.old`.
4. The build puts the new files in place in this order: segments, vectors, meta.
5. The build deletes the `.old` files and the work file.

- A build that fails or is cancelled before step 4 changes no file of the index.
- A failure in step 4 puts the old files back.
- The id of the build is in `segments.sqlite3` and in `meta.json`. Until `meta.json` is in place, the two ids differ. Files of two builds are thus never read as one index: `status()` says `none` and `read()` gives `None`.
- A process that dies in step 4 leaves the `.old` files. The next build puts them back before it does anything else. Then it reads the old index and embeds only what is new.

## How a build runs

```python
build_index(source, embedder, store, *, progress=None, cancel=None,
            count_tokens=default_count_tokens, sleep=time.sleep, limits=BatchLimits()) -> BuildResult
```

1. It takes the lock and repairs the folder.
2. It reads the index that is there and the work file. Each content hash with a vector is known.
3. It reads each conversation of the source and cuts it into segments.
4. It embeds the text of each content hash that is not known. Two segments with the same text are one text of a request.
5. It writes each answer of the embedder to the work file before it asks for the next one.
6. It writes the new files. A vector comes from the old index or from the work file.
7. It swaps the files and deletes the work file.

`BuildResult(segments, embedded, reused, removed, tokens)`:

- `segments`: the segments of the new index.
- `reused`: the segments whose content hash was known in step 2.
- `embedded`: the other segments. `segments` is `embedded` plus `reused`.
- `removed`: the segments of the old index that the source no longer gives.
- `tokens`: the counted tokens of the texts that this build sent. It is the count of `count_tokens`, not the bill of the provider.

Requests:

- One request holds 96 texts at most and 100,000 counted tokens at most. A text above 100,000 counted tokens is a request of its own.
- A request that the service calls too large is sent again as two halves. One text that is too large ends the build with `EmbeddingTooLarge`.
- A rate limit waits the time that the service names, through `sleep`, and the request is tried again. One request is tried 6 times at most. When the service names no wait, the waits are 2, 4, 8, 16 and 32 seconds.
- A named wait above 300 seconds ends the build with `EmbeddingRateLimited`. The build does not ask earlier.
- No other failure is tried again.
- `BatchLimits(max_texts, max_tokens, max_tries, max_wait_s, backoff_s)` changes these numbers.

Stop and progress:

- `cancel` is a callable that returns a bool, an object with `is_set()`, or an object with `cancelled` (the `CancelToken` of `report_engine`). The build looks at it before each conversation, before each request, and before it writes the files.
- A cancelled build raises `BuildCancelled`. Its `texts_embedded` and `tokens` say what the build paid for.
- A build that stops keeps the work file. The next build with the same model and dimensions does not pay for those vectors again.
- `progress` gets `BuildProgress(phase, done, total)`. The phases are `read` (conversations), `embed` (texts) and `write` (segments), in this order.

## Estimate

```python
estimate_build(source, store, embedder_facts, count_tokens=default_count_tokens) -> BuildEstimate
```

`BuildEstimate(segments, new_segments, tokens, usd)` says what a build would do now. It reads the source, the index and the work file. It sends no request and writes no file. `tokens` and `usd` are for the texts that the build would send. `usd` is `tokens` times the price in `models.json`.

Give `estimate_build` and `build_index` the same token counter. Then the estimate and the result of the build agree.

## How a search runs

```python
MeaningIndex.load(store) -> MeaningIndex | None
index.search(vector, *, conversations=None, since=None, until=None, limit=20) -> list[SearchHit]
```

1. The query vector is made unit length. A vector with another number of dimensions, with no length or with a value that is no number is a `ValueError`.
2. The filters choose rows.
   - `conversations` is one id or a list of ids. `[]` matches nothing. An unknown id matches nothing.
   - `since` and `until` are local days, `YYYY-MM-DD`, the format of the package `archive`. A segment matches when it overlaps the days. Both days are included. Other text is a `ValueError`.
3. The score of a row is the dot product of its vector and the query.
4. The hits are the `limit` rows with the highest score, highest first. Equal scores come in row order.

`SearchHit(segment, score)`: `segment` is a `Segment`. It has every column of the segments file but `row`. The caller reads the messages from the archive with `first_message_id` and `last_message_id`.

- The caller embeds the query. Use the model and the dimensions of the index: `index.model`, `index.dimensions`.
- The vectors stay on disk as a memory map of float16. A search reads them in chunks of 4,096 rows. The memory of a search does not grow with the index.
- A loaded index answers from the build that it loaded, also after a new build. Load again after a build.
- Measured on invented vectors: a search over 100,000 segments of 1,024 dimensions takes about 0.15 seconds. `tests/meaning_index/test_speed.py` holds the limit of one second.

## Model facts

`meaning_index/models.json` holds the facts of the embedding models. `embedder_facts(model=None, dimensions=None)` gives `EmbedderFacts(model, dimensions, usd_per_mtok, max_input_tokens, tokenizer_family)`.

| Model | Dimensions | USD for one million tokens | Largest input |
| --- | --- | --- | --- |
| `text-embedding-3-small` | 1536 | 0.02 | 8,192 tokens |
| `text-embedding-3-large` | 3072 | 0.13 | 8,192 tokens |

- Both models can be shortened with `dimensions`.
- The default is `text-embedding-3-large` at 1024 dimensions.
- A model that is not in the file is a `ModelFactsError`. It is never priced as another model.
- A fact that was not read is `null`.
- `build_index` does not read this file. It takes the model and the dimensions from the embedder.

## What was not read from the provider's pages

- The request limits: the number of inputs in one request, the tokens in one request, the tokens in one minute. They are `null` in `models.json`. The limits of a request (96 texts, 100,000 counted tokens) are Recall's own.
- The tokenizer family. `cl100k_base` is an assumption. `models.json` marks it with `tokenizer_family_assumed`.
- The error codes of a request that is too large. `openai_embedder.py` takes HTTP 413, the codes `context_length_exceeded`, `max_tokens_per_request`, `string_above_max_length` and `array_above_max_length`, and four phrases in the message of an HTTP 400. No real reply was seen.
- The names of the headers of a rate limit. They are the ones that `report_engine/openai_llm.py` reads.
- The shape of the reply. The tests build it with the reply type of the installed SDK (openai 2.36.0). No real request was sent.

## Limits

- The index does not follow the archive by itself. Run a build after a pull.
- The index gives no way from one message to its segment. A caller that needs it compares the conversation and the time of the message with `first_timestamp` and `last_timestamp`.
- A build reads every conversation again. It embeds only what is new.
- A segment is cut on local time. The gap of 120 minutes is measured on local clock times: on a day with a clock change it is one hour off.
- The first line of a text is a local day. When the time zone of the Mac changes, the day of some segments changes, and the next build embeds those segments again.
- New messages at the end of a conversation can join its last segment. That segment then has a new id and a new vector.
- A build keeps the text of each new segment in memory until it is embedded.
- While a build swaps the files, `status()` can say `none` for a moment.
- The swap keeps the old files as hard links. The folder must be on a file system that has hard links. APFS has them.
- float16 keeps about three decimal places. A score is exact to about 0.001.
- The default token counter needs the vocabulary file of tiktoken. tiktoken fetches it the first time. When it can not load, the count is one token for three bytes, and the log says so one time. Give a counter of the app to control this.
- `meaning_index/models.json` is read from beside the code. The packaged server must ship it, as it ships `recall_services/models.json`.

## Decisions where the contract was silent

- The 400 characters of a line include the label `Me: ` or `Them: `.
- The 1,600 characters count the whole text to embed, the day line too. The segment ends after the message that reaches 1,600.
- White space inside a message becomes one space, so that one message is one line.
- A time that goes back is no gap.
- `search` makes the query unit length. The score is then a cosine, from -1 to 1.
- `MeaningIndex.load` gives `None` for files that are not one whole build.
- An index of a newer format is an `IndexFormatError`. `status()`, `read()`, a build and an estimate raise it. No build replaces such an index.
- A build over an index that can not be read embeds every segment again. The log says so. An estimate counts every segment as new.
- An index with no segment is `ready`, and its search gives no hit.
- A failure of the source or of the embedder that is not typed becomes `SourceError` or `EmbeddingServiceError`. The error names the type of what was raised and nothing else.
- Only a rate limit is tried again. A network failure ends the build. The work file keeps what was paid, so a second build goes on from there.
- `insufficient_quota` is not a rate limit: a wait does not help.
- A conversation that the source names two times is read one time.
- The work file is deleted after a build that ends well, with the vectors of another model too.

## Tests

```bash
sealed-test.sh <worktree> tests/meaning_index --cov=meaning_index --cov-report=term-missing
```

The tests use invented data only. `tests/guards.py` sets a dummy key and blocks every socket that leaves the machine. `tests/meaning_index/conftest.py` blocks loopback too. The OpenAI adapter runs against a stand-in client. The token counter of the tests counts words. The archive adapter runs on a real archive that the test makes in a temp folder from invented rows.
