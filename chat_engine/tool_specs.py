"""The tools the model reads: names, descriptions and argument schemas.

`TOOL_SPECS` is the list of an engine with no memory port. `tool_specs` gives
the list of one engine: a tool is in it only when its port is there.
"""

from __future__ import annotations

from typing import Any, Mapping

from .ports import PERSON_KINDS, ToolSpec
from .prompt import DIGEST, PEOPLE
from .tool_figures import GROUPS, LIST_LIMIT_MAX
from .tool_limits import DIGESTS_PER_QUESTION, SEARCH_LIMIT_MAX
from .tool_people import PEOPLE_LIMIT_DEFAULT, PEOPLE_LIMIT_MAX
from .tool_search import READ_RADIUS_MAX

_DAY = "A day as YYYY-MM-DD in local time. The day itself is included."
_CONVERSATIONS = (
    "Conversation aliases such as c3, from list_conversations or an earlier result. "
    "Null means every conversation in the scope."
)


def _nullable(kind: str, description: str, **extra: Any) -> dict[str, Any]:
    return {"type": [kind, "null"], "description": description, **extra}


def _schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(properties),
    }


_SEARCH_WORDS = (
    "Find messages by their words. Each hit comes with the messages around it; every line has a "
    "reference such as m12. All words must be in the same message. Put a phrase in double quotes. "
    "End a word with * to match its beginning. Case and accents do not matter. Use few, distinctive "
    'words. Set match to "any" to rank messages that hold any of the words: good for synonyms and '
    "for words in other languages. For the first or the last time, set order. "
    "To read one conversation with no search word, leave query empty and give that one conversation: "
    'order "oldest" reads it from its start, "newest" from its end; since and until read a period.'
)
_SEARCH_MEANING = (
    ' Set match to "meaning" to find exchanges by what they are about, when you do not know the '
    "words: an apology, a plan, good news, a fight. Then query is a short description of the exchange, "
    'such as "we made up after a fight" or "planning a trip together". The result is whole exchanges, '
    "best first, found by meaning and by words together; limit counts exchanges, from_me must be null, and "
    "order is not used."
)


def _search_spec(meaning: bool) -> ToolSpec:
    matches = ["all", "any", "meaning", None] if meaning else ["all", "any", None]
    words = '"all" (default), "any" or "meaning".' if meaning else '"all" (default) or "any".'
    query = "Words to look for. Empty or null reads one conversation."
    limit = (
        f"Hits to show, 1 to {SEARCH_LIMIT_MAX}. Default 8. "
        "Each hit also shows up to 2 messages before it and 2 after it."
    )
    if meaning:
        query = (
            'Words to look for. With match "meaning": a short description of the exchange, never empty. '
            "With another match, empty or null reads one conversation."
        )
        limit += ' With match "meaning": exchanges to show.'
    return ToolSpec(
        "search_messages",
        _SEARCH_WORDS + (_SEARCH_MEANING if meaning else ""),
        _schema(
            {
                "query": _nullable("string", query),
                "match": _nullable("string", words, enum=matches),
                "conversations": _nullable("array", _CONVERSATIONS, items={"type": "string"}),
                "since": _nullable("string", _DAY),
                "until": _nullable("string", _DAY),
                "from_me": _nullable(
                    "boolean", "true: only messages the user sent. false: only messages from others."
                ),
                "order": _nullable(
                    "string", '"newest" (default), "oldest" or "relevance".',
                    enum=["newest", "oldest", "relevance", None],
                ),
                "limit": _nullable("integer", limit),
            }
        ),
    )


DIGEST_SPEC = ToolSpec(
    DIGEST,
    "The story of one conversation, period by period. For each period it gives the days, the number "
    "of messages, a summary, and the moments that matter with the messages that back them. Every "
    "message line has a reference such as m12: cite it, or read around it. Use it for a question about "
    "one person or one conversation as a whole: what the relationship is like, how it changed, what "
    "happened in a year. A summary is not a quote. A conversation that is too long to read inside one "
    "question is not read: the tool says so. Give days to read a part of the conversation. One question "
    f"builds at most {DIGESTS_PER_QUESTION} digests.",
    _schema(
        {
            "conversation": {"type": "string", "description": "A conversation alias such as c3."},
            "since": _nullable("string", _DAY),
            "until": _nullable("string", _DAY),
        }
    ),
)

FIND_PEOPLE_DESCRIPTION = (
    "Find people by the kind of relationship and its story: a talking stage that did not work out, a friend I "
    "lost touch with, a coworker who became a friend. Gives each person's conversations (aliases), the kind, "
    "the status, the span and the counts, and a short story. Confirm a person in their own conversation before "
    "you name them."
)
FIND_PEOPLE_SPEC = ToolSpec(
    PEOPLE,
    FIND_PEOPLE_DESCRIPTION,
    _schema(
        {
            "description": {
                "type": "string",
                "description": "What the relationship was and how it went, in a few words, such as "
                '"a talking stage that faded" or "a friend I lost touch with".',
            },
            "kinds": _nullable(
                "array",
                "Only people of these kinds. Null means every kind. talking_stage: romantic interest on at least "
                "one side that did not become a relationship. romantic: a relationship.",
                items={"type": "string", "enum": list(PERSON_KINDS)},
            ),
            "since": _nullable("string", f"{_DAY} A person counts when their messages overlap the days."),
            "until": _nullable("string", _DAY),
            "limit": _nullable("integer", f"People to show, 1 to {PEOPLE_LIMIT_MAX}. Default {PEOPLE_LIMIT_DEFAULT}."),
        }
    ),
)

_OTHER_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "read_around",
        "Read the exchange around one message: the messages before and after it in the same "
        "conversation. Use it before you rely on a hit, and to read on from the edge of a result.",
        _schema(
            {
                "reference": {"type": "string", "description": "A message reference such as m12."},
                "radius": _nullable(
                    "integer", f"Messages on each side, 1 to {READ_RADIUS_MAX}. Default 6."
                ),
            }
        ),
    ),
    ToolSpec(
        "count_messages",
        "Count messages exactly. Use it for every number: how many, how often, who most, which month. "
        "Without group_by it gives one number. With group_by it gives one row per group, each with the "
        "total, the count the user sent and the count from others.",
        _schema(
            {
                "conversations": _nullable("array", _CONVERSATIONS, items={"type": "string"}),
                "since": _nullable("string", _DAY),
                "until": _nullable("string", _DAY),
                "from_me": _nullable(
                    "boolean", "true: only messages the user sent. false: only messages from others."
                ),
                "text": _nullable(
                    "string", "Count only messages that match these words. Same rules as search_messages."
                ),
                "group_by": _nullable("string", "One row per group.", enum=[*GROUPS, None]),
            }
        ),
    ),
    ToolSpec(
        "list_conversations",
        "Find conversations and people by name. Gives the alias (c3), the name, direct or group, message "
        "counts and the first and last day. One person can have several threads and be in groups: all "
        "are listed. Digits match the last four digits of a phone number, or the whole number. Without a "
        "name it lists the conversations "
        "with the most messages.",
        _schema(
            {
                "name": _nullable(
                    "string", "A name, a part of a name, or the last four digits of a phone number."
                ),
                "limit": _nullable("integer", f"Rows to show, 1 to {LIST_LIMIT_MAX}. Default 10."),
            }
        ),
    ),
    ToolSpec(
        "conversation_stats",
        "Figures for one conversation: totals, sent and received, first and last time, active days, "
        "busiest day, longest gap. Give days to limit the period.",
        _schema(
            {
                "conversation": {"type": "string", "description": "A conversation alias such as c3."},
                "since": _nullable("string", _DAY),
                "until": _nullable("string", _DAY),
            }
        ),
    ),
)


def tool_specs(*, meaning: bool = False, digest: bool = False, people: bool = False) -> tuple[ToolSpec, ...]:
    """The tools of one engine. `meaning`, `digest` and `people` say which optional ports it has."""
    specs = (_search_spec(meaning), *_OTHER_SPECS)
    with_digest = (*specs, DIGEST_SPEC) if digest else specs
    return (*with_digest, FIND_PEOPLE_SPEC) if people else with_digest


TOOL_SPECS: tuple[ToolSpec, ...] = tool_specs()
