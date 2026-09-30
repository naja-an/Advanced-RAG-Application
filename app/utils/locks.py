"""Per-key asyncio locks (e.g. one lock per session)."""

import asyncio
from collections.abc import Hashable


class KeyedLocks:
    def __init__(self) -> None:
        self._locks: dict[Hashable, asyncio.Lock] = {}

    def get(self, key: Hashable) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    def discard(self, key: Hashable) -> None:
        self._locks.pop(key, None)
