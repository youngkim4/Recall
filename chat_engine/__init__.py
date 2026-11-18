"""Ask your messages: chat as a tool loop over the archive.

Start with `chat_engine/README.md`. The main names:

    chat_engine.engine         ChatEngine, EngineConfig
    chat_engine.session        ChatSession
    chat_engine.scope          Scope
    chat_engine.cancel         CancelToken
    chat_engine.errors         one class for each failure
    chat_engine.labeler        MaskingLabeler
    chat_engine.openai_llm     OpenAIResponsesLLM
    chat_engine.archive_adapter  open_archive
    chat_engine.local_answer   local_answer
    chat_engine.ports          MeaningSearch, DigestSource: two optional ports of ChatEngine

Importing this package imports none of them, so it needs no model SDK and no archive.
"""
