"""LlamaIndex tool specification for Thordata SERP operations."""

from importlib.resources import files
import json
from typing import Any, Literal

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from pydantic import BaseModel, Field, StrictStr, ValidationError

from .client import (
    DEFAULT_SCHEMA_ENDPOINT,
    DEFAULT_SERP_ENDPOINT,
    SchemaHttpClient,
    SerpClient,
)
from .response import (
    ResponseError,
    SerpApiError,
    compact_serp_response,
    decode_serp_body,
    redact_serp_response,
    _safe_message,
)
from .schema import SchemaError, SchemaRepository, find_engine
from .serializer import ParameterError, serialize_search


class ListEnginesInput(BaseModel):
    """Input for listing the engines available in the current schema."""


class GetEngineSchemaInput(BaseModel):
    """Input for retrieving one engine's full parameter schema."""

    engine: StrictStr = Field(description="Thordata engine key, such as 'google'.")


class SearchInput(BaseModel):
    """Input for a schema-directed SERP request."""

    engine: StrictStr = Field(default="", description="Engine key; uses the schema default when empty.")
    query: StrictStr = Field(default="", description="Search query mapped to the engine query field.")
    params: dict | None = Field(default=None, description="Additional engine-specific parameters.")
    response_format: Literal["1", "2", "3"] = Field(
        default="1", description="Thordata JSON response format."
    )
    response_mode: Literal["complete", "compact"] = Field(
        default="complete", description="Return full data or remove verbose metadata."
    )


class ThordataSerpToolSpec(BaseToolSpec):
    """Tool specification for Thordata SERP operations."""

    spec_functions = ["list_engines", "get_engine_schema", "search"]

    _input_schemas = {
        "list_engines": ListEnginesInput,
        "get_engine_schema": GetEngineSchemaInput,
        "search": SearchInput,
    }

    def __init__(
        self,
        api_key: str,
        serp_endpoint: str = DEFAULT_SERP_ENDPOINT,
        schema_endpoint: str = DEFAULT_SCHEMA_ENDPOINT,
        timeout: int = 120,
        schema_timeout: int = 30,
        schema_cache_ttl: float = 300,
    ) -> None:
        """Create a tool spec using a remote schema with a bundled fallback."""
        self._serp_client = SerpClient(api_key, endpoint=serp_endpoint, timeout=timeout)
        self._api_key = self._serp_client.api_key
        self._schema_repository = SchemaRepository(
            SchemaHttpClient(endpoint=schema_endpoint, timeout=schema_timeout),
            files("llama_index.tools.thordata_serp").joinpath("serp-schema.snapshot.json"),
            ttl=schema_cache_ttl,
        )

    def get_fn_schema_from_fn_name(self, fn_name: str, spec_functions=None):
        """Return the Pydantic input model associated with a public tool method."""
        return self._input_schemas.get(fn_name)

    def list_engines(self) -> str:
        """List available SERP engines and their query field names."""
        try:
            schema, source = self._schema_repository.current()
            engines = [
                {
                    "key": engine["key"],
                    "name": engine["name"],
                    "query_field": engine["query_field"],
                }
                for engine in self._iter_engines(schema)
            ]
            return self._json_result(
                {
                    "ok": True,
                    "schema_version": schema["schema_version"],
                    "audience": schema["audience"],
                    "default_engine": schema["default_engine"],
                    "schema_source": source,
                    "engines": engines,
                }
            )
        except (SchemaError, ResponseError, requests.RequestException) as error:
            return self._error_result(error)

    def get_engine_schema(self, engine: str) -> str:
        """Return the complete parameter schema for one SERP engine."""
        try:
            request_input = GetEngineSchemaInput.model_validate({"engine": engine})
            if not request_input.engine.strip():
                raise ValueError("engine must be a nonblank string")
            schema, source = self._schema_repository.current()
            return self._json_result(
                {
                    "ok": True,
                    "schema_source": source,
                    "engine": find_engine(schema, request_input.engine),
                }
            )
        except (ValidationError, ValueError) as error:
            return self._error_result(error, error_type="ValueError")
        except (SchemaError, ResponseError, requests.RequestException) as error:
            return self._error_result(error)

    def search(
        self,
        engine: str = "",
        query: str = "",
        params: dict | None = None,
        response_format: str = "1",
        response_mode: str = "complete",
    ) -> str:
        """Execute a schema-directed Thordata SERP search."""
        try:
            request_input = SearchInput.model_validate(
                {
                    "engine": engine,
                    "query": query,
                    "params": params,
                    "response_format": response_format,
                    "response_mode": response_mode,
                }
            )
            schema, _ = self._schema_repository.current()
            selected_engine, form = serialize_search(
                schema,
                request_input.engine,
                request_input.query,
                request_input.params,
                request_input.response_format,
            )
            try:
                status_code, body = self._serp_client.request(form)
            except requests.RequestException as error:
                raise ResponseError(f"SERP request failed: {error}") from error
            data = decode_serp_body(status_code, body, secret=self._api_key)
            if request_input.response_mode == "compact":
                data = compact_serp_response(data)
            return self._json_result(
                {
                    "ok": True,
                    "status": status_code,
                    "engine": selected_engine["key"],
                    "data": data,
                }
            )
        except (ValidationError, ParameterError, ValueError) as error:
            return self._error_result(error, error_type="ValueError")
        except (SchemaError, ResponseError, SerpApiError, requests.RequestException) as error:
            return self._error_result(error)

    @staticmethod
    def _iter_engines(schema: dict[str, Any]):
        for category in schema.get("categories", []):
            yield from category.get("engines", [])
        yield from schema.get("engines", [])

    def _error_result(self, error: Exception, error_type: str | None = None) -> str:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {
                "type": error_type or type(error).__name__,
                "message": _safe_message(str(error), "SERP request failed", self._api_key),
            },
        }
        if isinstance(error, SerpApiError):
            payload["error"]["status_code"] = error.status_code
        return self._json_result(payload)

    def _json_result(self, payload: dict[str, Any]) -> str:
        # Preserve the trusted envelope and redact only schema/upstream content.
        safe_payload = {
            key: value if key in {"ok", "status", "error", "schema_source"}
            else redact_serp_response(value, self._api_key)
            for key, value in payload.items()
        }
        return json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
