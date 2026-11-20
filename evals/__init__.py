"""The eval of the chat feature: measure answer quality, so later changes can be compared.

What it does:
1. `evals.world` writes an invented history of about three years (about
   1,800 messages: a partner, a talking stage, a friend who drifted, a family
   group, work, a recruiter and a landlord) into a synthetic chat.db with the
   builder of the tests, and imports it with `archive.import_chat_db`.
2. `evals.questions` gives 60 questions of eight kinds, each with a gold
   answer, key facts and the guids of the messages that support it. Counts
   and figures over time are counted from the world by code.
3. `evals.runner` asks each question through `recall_services.chat.stream`,
   as the app asks: same model registry, same ports, the app's history rows
   for a follow-up. `evals.recorder` watches each model call.
4. `evals.score` and `evals.judge` score each answer: citation recall and
   precision against the gold guids, a judge model for correctness, and the
   calls, tokens, money and time.
5. `evals.run` is the command. `evals.estimate` is its --dry-run.
6. `--world hard` (`evals.world.hard`, `evals.hard_questions`) puts the small
   world inside a busy six-year phone of about 54,000 messages and adds 41
   hard questions: paraphrase, disambiguate, temporal, aggregate, long_story
   and needle.

How to run (from the repository root, with the key that Recall uses):

    python -m evals.run --dry-run           # estimate only; no call
    python -m evals.run                     # all 60 questions, default model
    python -m evals.run --model gpt-6.1-terra --effort low --kinds count,pattern

The results go to `<data folder>/evals/chat-eval-<time>.json`, never into the
repository. Compare two runs by their `summary` blocks.

Privacy: the world is invented. A run never opens the owner's archive or
saves: the saves folder points into the temp folder of the world while a
run lasts (`evals.places.isolated_saves`). The text of the invented world
does go to the provider in a paid run.
"""
