"""Bounded validation of JSON trees before recursive serialization or copying."""

import math
from typing import Any


def validate_json_tree(value: Any, *, max_depth: int = 256) -> None:
    """Reject cycles, excessive nesting, invalid Unicode and non-JSON values."""
    active: set[int] = set()
    stack = [(iter((value,)), -1, None)]
    while stack:
        iterator, depth, container_id = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            if container_id is not None:
                active.remove(container_id)
            continue
        if isinstance(item, (dict, list)):
            if id(item) in active:
                raise ValueError("JSON nesting contains a cycle")
            if depth >= max_depth:
                raise ValueError(f"JSON nesting exceeds {max_depth} levels")
            if isinstance(item, dict):
                for key in item:
                    if not isinstance(key, str):
                        raise ValueError("JSON object keys must be strings")
                    _validate_string(key)
            active.add(id(item))
            stack.append((iter(item.values()) if isinstance(item, dict) else iter(item), depth + 1, id(item)))
        elif isinstance(item, str):
            _validate_string(item)
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
        elif item is not None and not isinstance(item, (int, bool)):
            raise ValueError("value is not a JSON type")


def _validate_string(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("JSON contains invalid Unicode") from error
