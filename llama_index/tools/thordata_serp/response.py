"""SERP response decoding, compaction, and secret redaction helpers."""

import json
from typing import Any

from .json_safety import validate_json_tree


COMPACT_KEYS = frozenset(
    {"search_metadata", "search_parameters", "serpapi_pagination", "request_metadata"}
)
_BUSINESS_CODE_KEYS = ("code", "statusCode", "status_code", "error_code")
_MAX_RESPONSE_NESTING = 256
_MAX_ERROR_MESSAGE_CHARS = 512


class ResponseError(RuntimeError):
    """Raised when a SERP response cannot be decoded."""


class SerpApiError(RuntimeError):
    """Raised when Thordata or an upstream SERP provider reports an error."""

    def __init__(self, status_code: int, message: str, payload: Any = None) -> None:
        self.status_code = status_code
        self.payload = payload
        super().__init__(message)


def _safe_message(value: str, fallback: str, secret: str | None = None) -> str:
    if secret:
        value = value.replace(secret, "[REDACTED]")
    message = " ".join(value.encode("utf-8", errors="replace").decode("utf-8").split())
    if not message:
        return fallback
    if len(message) > _MAX_ERROR_MESSAGE_CHARS:
        return message[: _MAX_ERROR_MESSAGE_CHARS - 3] + "..."
    return message


def _message(payload: Any, fallback: str, secret: str | None = None) -> str:
    if isinstance(payload, dict):
        for key in ("message", "error", "msg", "data"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _safe_message(value, fallback, secret)
    if isinstance(payload, str) and payload.strip():
        return _safe_message(payload, fallback, secret)
    return fallback


def _reject_json_constant(value: str) -> None:
    raise ResponseError(f"invalid JSON constant: {value}")


def _validate_response_tree(value: Any) -> None:
    try:
        validate_json_tree(value)
    except ValueError as error:
        raise ResponseError(f"invalid JSON response: {error}") from error


def _load_json(body: bytes) -> Any:
    try:
        # Validate parsed values separately so malformed JSON on an HTTP error
        # can use a plain-text fallback, while unsafe JSON remains ResponseError.
        return json.loads(body.decode("utf-8"))
    except ResponseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, RecursionError) as error:
        raise ResponseError("invalid JSON response") from error


def _plain_error_message(body: bytes, fallback: str, secret: str | None = None) -> str:
    try:
        message = body.decode("utf-8")
    except UnicodeDecodeError:
        return fallback
    if message.lstrip().startswith("<"):
        return fallback
    return _safe_message(message, fallback, secret)


def _unwrap_json(value: Any) -> Any:
    """Decode no more than two JSON-string wrappers around an object or array."""
    current = value
    for _ in range(2):
        if not isinstance(current, str):
            break
        try:
            decoded = json.loads(current, parse_constant=_reject_json_constant)
        except ResponseError:
            raise
        except RecursionError as error:
            raise ResponseError("invalid JSON response nesting too deep") from error
        except (json.JSONDecodeError, TypeError, ValueError):
            break
        _validate_response_tree(decoded)
        if isinstance(decoded, (dict, list)):
            return decoded
        if isinstance(decoded, str):
            current = decoded
            continue
        break
    return current


def _is_success_code(value: Any) -> bool:
    if value is None:
        return True
    if type(value) is int:
        return value == 0 or value == 200
    if type(value) is str:
        return value == "" or value == "0" or value == "200"
    return False


def _error_status(value: Any) -> int:
    if type(value) is int:
        return value
    if type(value) is not str:
        return 500
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 500


def decode_serp_body(status_code: int, body: bytes, *, secret: str | None = None) -> Any:
    """Decode an HTTP response and normalize Thordata business errors."""
    if not 200 <= status_code < 300:
        fallback = f"SERP HTTP {status_code} error"
        try:
            parsed = _load_json(body)
        except ResponseError:
            raise SerpApiError(status_code, _plain_error_message(body, fallback, secret)) from None
        _validate_response_tree(parsed)
        raise SerpApiError(status_code, _message(parsed, fallback, secret), parsed)

    parsed = _load_json(body)
    _validate_response_tree(parsed)

    if isinstance(parsed, dict):
        for key in _BUSINESS_CODE_KEYS:
            if key in parsed and not _is_success_code(parsed[key]):
                error_status = _error_status(parsed[key])
                raise SerpApiError(
                    error_status,
                    _message(parsed, f"SERP business error ({error_status})", secret),
                    parsed,
                )

        error_value = parsed.get("error")
        if isinstance(error_value, str) and error_value.strip():
            raise SerpApiError(
                500,
                _safe_message(error_value, "SERP business error (500)", secret),
                parsed,
            )
        result = parsed["data"] if "data" in parsed else parsed
    else:
        result = parsed

    result = _unwrap_json(result)
    if isinstance(result, str) and result.lstrip().lower().startswith(("error", "failed")):
        raise SerpApiError(500, _safe_message(result, "SERP business error (500)", secret), parsed)
    return result


def _copy_response(value: Any, *, compact: bool, secret: str | None = None) -> Any:
    """Copy a response iteratively, rejecting cycles and nesting beyond 256 levels."""
    _validate_response_tree(value)
    if not isinstance(value, (dict, list)):
        return value.replace(secret, "[REDACTED]") if isinstance(value, str) and secret else value

    result: dict[Any, Any] | list[Any] = {} if isinstance(value, dict) else []
    active = {id(value)}
    stack = [
        (
            value,
            result,
            iter(value.items()) if isinstance(value, dict) else iter(value),
            0,
        )
    ]
    while stack:
        source, target, iterator, depth = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            active.remove(id(source))
            stack.pop()
            continue

        if isinstance(source, dict):
            key, item_value = item
            if compact and key in COMPACT_KEYS:
                continue
            if secret:
                key = key.replace(secret, "[REDACTED]")
        else:
            key = None
            item_value = item

        if isinstance(item_value, (dict, list)):
            if id(item_value) in active:
                raise ResponseError("response nesting contains a cycle")
            if depth >= _MAX_RESPONSE_NESTING:
                raise ResponseError("response nesting exceeds 256 levels")
            copied_value: dict[Any, Any] | list[Any] = (
                {} if isinstance(item_value, dict) else []
            )
            if isinstance(source, dict):
                target[key] = copied_value
            else:
                target.append(copied_value)
            active.add(id(item_value))
            stack.append(
                (
                    item_value,
                    copied_value,
                    iter(item_value.items()) if isinstance(item_value, dict) else iter(item_value),
                    depth + 1,
                )
            )
            continue

        if isinstance(item_value, str) and secret:
            item_value = item_value.replace(secret, "[REDACTED]")
        if isinstance(source, dict):
            target[key] = item_value
        else:
            target.append(item_value)

    return result


def compact_serp_response(value: Any) -> Any:
    """Return a copy with verbose SERP metadata removed at every nesting level."""
    return _copy_response(value, compact=True)


def redact_serp_response(value: Any, secret: str) -> Any:
    """Redact string values and keys; the last entry wins if redacted keys collide."""
    return _copy_response(value, compact=False, secret=secret)
