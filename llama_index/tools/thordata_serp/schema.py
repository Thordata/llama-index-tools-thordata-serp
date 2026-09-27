"""Validation and loading utilities for Thordata SERP schema documents."""

from copy import deepcopy
from dataclasses import dataclass
import json
import threading
import time
from typing import Any, Callable, Protocol

from .json_safety import validate_json_tree


class SchemaError(RuntimeError):
    """Raised when a SERP schema document is invalid."""


class SchemaClientProtocol(Protocol):
    """Minimal interface required to fetch a remote schema document."""

    def fetch(self) -> dict[str, Any]:
        """Fetch a schema response envelope or direct schema object."""


class SnapshotResource(Protocol):
    """Minimal resource interface for bundled schema snapshots."""

    def read_text(self, *, encoding: str) -> str:
        """Read the resource as text using the requested encoding."""


@dataclass
class _SchemaRefresh:
    done: bool = False
    schema: dict[str, Any] | None = None
    source: str | None = None
    error: SchemaError | None = None


class SchemaRepository:
    """Load validated schemas with bounded caching and single-flight refreshes."""

    def __init__(
        self,
        client: SchemaClientProtocol,
        snapshot_path: SnapshotResource,
        ttl: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not ttl > 0:
            raise ValueError("ttl must be a positive non-boolean number")

        self._client = client
        self._snapshot_path = snapshot_path
        self._ttl = ttl
        self._clock = clock
        self._condition = threading.Condition()
        self._schema: dict[str, Any] | None = None
        self._loaded_at: float | None = None
        self._refresh: _SchemaRefresh | None = None

    def current(self) -> tuple[dict[str, Any], str]:
        """Return the current valid schema and the source that supplied it."""
        with self._condition:
            try:
                is_fresh = self._is_fresh()
            except Exception as clock_error:
                raise SchemaError(f"schema cache clock failed: {clock_error}") from clock_error
            if is_fresh:
                return deepcopy(self._schema), "cache"

            if self._refresh is not None:
                refresh = self._refresh
                while not refresh.done:
                    self._condition.wait()
                if refresh.error is not None:
                    raise refresh.error
                if refresh.schema is not None and refresh.source is not None:
                    return deepcopy(refresh.schema), refresh.source
                raise SchemaError("schema refresh completed without a result")

            refresh = _SchemaRefresh()
            self._refresh = refresh

        schema: dict[str, Any] | None = None
        source: str | None = None
        loaded_at: float | None = None
        error: SchemaError | None = None
        try:
            try:
                schema = validate_schema_payload(self._client.fetch())
                source = "remote"
            except Exception as remote_error:
                with self._condition:
                    stale_schema = deepcopy(self._schema)

                if stale_schema is not None:
                    schema = stale_schema
                    source = "cache"
                else:
                    try:
                        schema = load_snapshot(self._snapshot_path)
                        source = "snapshot"
                    except Exception as snapshot_error:
                        error = SchemaError(
                            "remote schema fetch failed: "
                            f"{remote_error}; schema snapshot fallback failed: {snapshot_error}"
                        )
            if error is None and source in {"remote", "snapshot"}:
                try:
                    loaded_at = self._clock()
                except Exception as clock_error:
                    error = SchemaError(f"schema cache timestamp failed: {clock_error}")
        except Exception as unexpected_error:
            error = SchemaError(f"schema refresh failed: {unexpected_error}")
        finally:
            with self._condition:
                if error is None and schema is not None and source is not None:
                    if source in {"remote", "snapshot"}:
                        self._schema = schema
                        self._loaded_at = loaded_at
                    refresh.schema = schema
                    refresh.source = source
                    refresh.error = None
                else:
                    error = error or SchemaError("schema refresh failed without a result")
                    refresh.schema = None
                    refresh.source = None
                    refresh.error = error
                refresh.done = True
                if self._refresh is refresh:
                    self._refresh = None
                self._condition.notify_all()

        if refresh.error is not None:
            raise refresh.error
        if refresh.schema is None or refresh.source is None:
            raise SchemaError("schema refresh failed without a result")
        return deepcopy(refresh.schema), refresh.source

    def _is_fresh(self) -> bool:
        return self._schema is not None and self._loaded_at is not None and self._clock() - self._loaded_at < self._ttl


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _optional_strings(value: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in value and not isinstance(value[key], str):
            raise SchemaError(f"{key} must be a string")


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _validate_field(field: dict[str, Any]) -> None:
    if not _nonblank(field.get("type")):
        raise SchemaError("field type must be a nonblank string")
    _optional_strings(field, ("control", "name", "group", "label", "description", "placeholder"))
    for key in ("visible", "required"):
        if key in field and not isinstance(field[key], bool):
            raise SchemaError(f"field {key} must be a boolean")
    if "options" in field:
        pending = [field["options"]]
        while pending:
            options = pending.pop()
            if not isinstance(options, list):
                raise SchemaError("field options must be an array")
            for option in options:
                if not isinstance(option, dict) or not _scalar(option.get("value")) or not _scalar(option.get("label")):
                    raise SchemaError("field options require scalar labels and values")
                if "children" in option:
                    pending.append(option["children"])
    for key in ("default", "default_value"):
        if key not in field or field[key] is None:
            continue
        value = field[key]
        field_type = field["type"]
        if field_type in {"array", "tags", "time_range", "cascader"}:
            valid = isinstance(value, list) and all(_scalar(item) for item in value)
        elif field_type in {"object", "date_range"}:
            valid = isinstance(value, dict)
        elif field_type in {"boolean", "switch"}:
            valid = isinstance(value, bool)
        elif field_type == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            valid = _scalar(value)
        if not valid:
            raise SchemaError(f"field {key} has invalid shape")


def _engines(schema: dict[str, Any]) -> list[dict[str, Any]]:
    engines: list[dict[str, Any]] = []
    categories = schema.get("categories", [])
    if not isinstance(categories, list):
        raise SchemaError("categories must be an array")

    category_keys: set[str] = set()
    for category in categories:
        if not isinstance(category, dict):
            raise SchemaError("category must be an object")
        category_key = category.get("key")
        if not _nonblank(category_key):
            raise SchemaError("category key must be nonblank")
        if category_key in category_keys:
            raise SchemaError("duplicate category key")
        category_keys.add(category_key)
        _optional_strings(category, ("name",))

        category_engines = category.get("engines")
        if not isinstance(category_engines, list):
            raise SchemaError("category engines must be an array")
        engines.extend(category_engines)

    direct_engines = schema.get("engines", [])
    if not isinstance(direct_engines, list):
        raise SchemaError("engines must be an array")
    engines.extend(direct_engines)
    return engines


def validate_schema_payload(payload: Any) -> dict[str, Any]:
    """Validate a schema response envelope or direct schema object."""
    if not isinstance(payload, dict):
        raise SchemaError("schema payload must be an object")
    try:
        validate_json_tree(payload)
    except ValueError as error:
        raise SchemaError(f"invalid schema JSON: {error}") from error

    if "code" in payload and payload["code"] not in (None, 0, "0", 200, "200"):
        code = payload["code"]
        for detail_key in ("msg", "message", "error"):
            detail = payload.get(detail_key)
            if detail is not None:
                raise SchemaError(f"schema payload business code {code}: {detail}")
        raise SchemaError(f"schema payload has unsuccessful business code: {code}")

    schema = payload.get("data", payload)
    if not isinstance(schema, dict):
        raise SchemaError("schema data must be an object")

    if not _nonblank(schema.get("schema_version")):
        raise SchemaError("schema_version must be nonblank")
    if schema.get("audience") != "is_serp_old=0":
        raise SchemaError("unsupported audience")
    if not _nonblank(schema.get("default_engine")):
        raise SchemaError("default engine must be nonblank")

    engines = _engines(schema)
    engine_keys: set[str] = set()
    for engine in engines:
        if not isinstance(engine, dict):
            raise SchemaError("engine must be an object")
        engine_key = engine.get("key")
        if not _nonblank(engine_key):
            raise SchemaError("engine key must be nonblank")
        if engine_key in engine_keys:
            raise SchemaError("duplicate engine key")
        engine_keys.add(engine_key)
        if not _nonblank(engine.get("name")):
            raise SchemaError("engine name must be nonblank")
        if not _nonblank(engine.get("query_field")):
            raise SchemaError("engine query_field must be nonblank")

        groups = engine.get("groups")
        if not isinstance(groups, list):
            raise SchemaError("engine groups must be an array")

        group_keys: set[str] = set()
        field_keys: set[str] = set()
        fields: list[dict[str, Any]] = []
        for group in groups:
            if not isinstance(group, dict):
                raise SchemaError("group must be an object")
            group_key = group.get("key")
            if not _nonblank(group_key):
                raise SchemaError("group key must be nonblank")
            if group_key in group_keys:
                raise SchemaError("duplicate group key")
            group_keys.add(group_key)
            _optional_strings(group, ("name",))

            group_fields = group.get("fields")
            if not isinstance(group_fields, list):
                raise SchemaError("group fields must be an array")
            for field in group_fields:
                if not isinstance(field, dict):
                    raise SchemaError("field must be an object")
                field_key = field.get("key")
                if not _nonblank(field_key):
                    raise SchemaError("field key must be nonblank")
                if field_key in field_keys:
                    raise SchemaError("duplicate field key")
                field_keys.add(field_key)
                _validate_field(field)
                fields.append(field)

        if engine["query_field"] not in field_keys:
            raise SchemaError("engine query_field must reference a declared field")

        for field in fields:
            if "show_when" not in field:
                continue
            show_when = field["show_when"]
            if not isinstance(show_when, dict) or not _nonblank(show_when.get("field")):
                raise SchemaError("show_when.field must be a nonblank field key")
            if show_when["field"] not in field_keys:
                raise SchemaError("show_when.field must reference a field in the same engine")
            _optional_strings(show_when, ("operator",))
            if "values" in show_when and (
                not isinstance(show_when["values"], list)
                or not all(_scalar(value) for value in show_when["values"])
            ):
                raise SchemaError("show_when.values must be an array of scalars")

    if schema["default_engine"] not in engine_keys:
        raise SchemaError("default engine must be declared")
    return deepcopy(schema)


def find_engine(schema: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a copy of the engine identified by *key*."""
    for engine in _engines(schema):
        if engine.get("key") == key:
            return deepcopy(engine)
    raise SchemaError(f"unknown engine: {key}")


def load_snapshot(path: SnapshotResource) -> dict[str, Any]:
    """Load and validate a UTF-8 JSON schema snapshot."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SchemaError(f"cannot load schema snapshot: {error}") from error
    return validate_schema_payload(payload)
