"""Limits shared by the tools that list messages."""

from __future__ import annotations

from .search import MAX_FETCH

SEARCH_LIMIT_DEFAULT = 8
SEARCH_LIMIT_MAX = MAX_FETCH // 3
ORDERS = ("newest", "oldest", "relevance")
