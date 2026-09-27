"""Schema-directed serialization for Thordata SERP search parameters."""

import json
import math
from typing import Any

from .schema import find_engine
from .json_safety import validate_json_tree


_COUNTRY_ALIASES = {
    "US": "US",
    "USA": "US",
    "UNITED STATES": "US",
    "GB": "GB",
    "UK": "GB",
    "UNITED KINGDOM": "GB",
    "CA": "CA",
    "CANADA": "CA",
    "AU": "AU",
    "AUSTRALIA": "AU",
    "CN": "CN",
    "CHINA": "CN",
}


class ParameterError(ValueError):
    """Raised when a search parameter cannot be serialized."""


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _scalar(value: Any, key: str) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ParameterError(f"{key} must be a scalar value")


def _scalar_text(value: Any, key: str) -> str:
    scalar = _scalar(value, key)
    if isinstance(scalar, bool):
        return "true" if scalar else "false"
    try:
        return str(scalar)
    except (OverflowError, ValueError) as error:
        raise ParameterError(f"{key} scalar conversion failed") from error


def _scalar_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list):
        raise ParameterError(f"{key} must be an array of scalar values")
    return [_scalar_text(item, key) for item in value]


def _country(value: Any, key: str) -> str:
    """Normalize supported country aliases after an optional ``country`` prefix."""
    country = _scalar_text(value, key).strip()
    if country.lower().startswith("country"):
        country = country[7:].strip()
    if not country:
        raise ParameterError(f"{key} must contain a country value")
    normalized = country.upper()
    return f"country{_COUNTRY_ALIASES.get(normalized, normalized)}"


def _fields(engine: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        field["key"]: field
        for group in engine["groups"]
        for field in group["fields"]
    }


def _serialize_date_range(value: Any, key: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ParameterError(f"{key} must be a date range object")
    start = value.get("start")
    end = value.get("end")
    if not _present(start) or not _present(end):
        raise ParameterError(f"{key} requires start and end values")
    return {
        f"{key}_start": _scalar_text(start, key),
        f"{key}_end": _scalar_text(end, key),
    }


def _time_component(value: Any, key: str) -> str:
    if isinstance(value, bool):
        raise ParameterError(f"{key} time components must be integers")
    if isinstance(value, int):
        if value < 0:
            raise ParameterError(f"{key} time components must be non-negative integers")
        try:
            text = str(value)
        except (OverflowError, ValueError) as error:
            raise ParameterError(f"{key} time component conversion failed") from error
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ParameterError(f"{key} time components must be integers")
    if not text.isascii() or not text.isdigit():
        raise ParameterError(f"{key} time components must be integers")
    return text[-2:].zfill(2)


def _serialize_time_range(value: Any, key: str) -> str:
    if not isinstance(value, list) or len(value) != 4:
        raise ParameterError(f"{key} must contain four time components")
    start_hour, start_minute, end_hour, end_minute = (
        _time_component(component, key) for component in value
    )
    return f"{start_hour}{start_minute},{end_hour}{end_minute}"


def _serialize_cascader(value: Any, key: str) -> str:
    values = _scalar_list(value, key)
    if not values:
        raise ParameterError(f"{key} must be a nonempty cascader")
    return values[-1]


def _serialize_field(field: dict[str, Any], value: Any, engine_key: str) -> dict[str, str]:
    key = field["key"]
    field_type = field.get("type")
    if field_type in {"boolean", "switch"}:
        if not isinstance(value, bool):
            raise ParameterError(f"{key} must be a boolean")
        return {key: str(value).lower()}
    if field_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParameterError(f"{key} must be a finite number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ParameterError(f"{key} must be a finite number")
        try:
            text = str(value)
        except (OverflowError, ValueError) as error:
            raise ParameterError(f"{key} number conversion failed") from error
        return {key: text}
    if field_type in {"string", "options"}:
        text = _scalar_text(value, key)
        if engine_key == "google_flights" and key in {"departure_id", "arrival_id"}:
            text = text.strip().upper()
        return {key: text}
    if field_type in {"array", "tags"}:
        values = _scalar_list(value, key)
        if key == "cr":
            return {key: "|".join(_country(item, key) for item in values)}
        return {key: ",".join(values)}
    if field_type == "object":
        if not isinstance(value, dict):
            raise ParameterError(f"{key} must be an object")
        try:
            validate_json_tree(value)
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, RecursionError) as error:
            raise ParameterError(f"{key} must be a JSON object") from error
        return {key: text}
    if field_type == "date_range":
        return _serialize_date_range(value, key)
    if field_type == "time_range":
        return {key: _serialize_time_range(value, key)}
    if field_type == "cascader":
        return {key: _serialize_cascader(value, key)}
    raise ParameterError(f"{key} has unsupported schema type: {field_type}")


def serialize_search(
    schema: dict[str, Any],
    engine_key: str | None,
    query: str | None,
    params: dict[str, Any] | None,
    response_format: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Serialize search inputs according to the selected engine's field schema."""
    selected_key = engine_key.strip() if isinstance(engine_key, str) else ""
    if not selected_key:
        selected_key = schema["default_engine"]
    engine = find_engine(schema, selected_key)
    if params is not None and not isinstance(params, dict):
        raise ParameterError("params must be an object")
    copied_params = dict(params) if params is not None else {}
    if isinstance(query, str) and query.strip():
        copied_params[engine["query_field"]] = query

    fields = _fields(engine)
    form: dict[str, str] = {}
    for key, value in copied_params.items():
        if key in {"engine", "json"} or key not in fields:
            continue
        field = fields[key]
        if not _present(value):
            continue
        form.update(_serialize_field(field, value, engine["key"]))

    if _present(response_format):
        form["json"] = _scalar_text(response_format, "json")
    form["engine"] = engine["key"]
    return engine, form
