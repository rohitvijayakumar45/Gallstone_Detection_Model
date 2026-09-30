"""In-process scan session store. Thread-safe, size-capped LRU-ish.

Replace with Redis or SQLite when concurrent workers matter. Kept simple
here so the API stays deployable as a single-process demo.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

_MAX = 128
_store: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_lock = Lock()


def put_session(scan_id: str, data: dict[str, Any]) -> None:
    with _lock:
        _store[scan_id] = data
        _store.move_to_end(scan_id)
        while len(_store) > _MAX:
            _store.popitem(last=False)


def get_session(scan_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        if scan_id not in _store:
            return None
        _store.move_to_end(scan_id)
        return _store[scan_id]


def update_session(scan_id: str, patch: dict[str, Any]) -> None:
    with _lock:
        if scan_id not in _store:
            return
        _store[scan_id].update(patch)
        _store.move_to_end(scan_id)


def size() -> int:
    with _lock:
        return len(_store)
