"""Limits shared by the tools that list messages."""

from __future__ import annotations

from .search import MAX_FETCH

SEARCH_LIMIT_DEFAULT = 8
SEARCH_LIMIT_MAX = MAX_FETCH // 3
ORDERS = ("newest", "oldest", "relevance")
DIGESTS_PER_QUESTION = 3  # each digest can cost money: one question builds at most this many
# each estimate plans a whole conversation: one question asks for at most this many
DIGEST_ESTIMATES_PER_QUESTION = 2 * DIGESTS_PER_QUESTION
