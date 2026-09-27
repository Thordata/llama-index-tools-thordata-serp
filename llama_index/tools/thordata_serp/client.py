"""Bounded HTTP clients for Thordata schema and SERP requests."""

import json
from typing import Any, Mapping
from urllib.parse import quote_plus, urlparse

import requests

from .response import ResponseError


DEFAULT_SCHEMA_ENDPOINT = "https://api.thordata.com/serp/playground/schema?lang=en"
DEFAULT_SERP_ENDPOINT = "https://scraperapi.thordata.com/request"
MAX_REQUEST_SIZE = 1 << 20
MAX_RESPONSE_SIZE = 10 << 20
_STREAM_CHUNK_SIZE = 64 * 1024


def _endpoint(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a nonblank http or https URL")
    endpoint = value.strip()
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be a nonblank http or https URL")
    return endpoint


def _timeout(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_limited(response: requests.Response) -> bytes:
    content_length = response.headers.get("Content-Length")
    try:
        if content_length is not None and int(content_length) > MAX_RESPONSE_SIZE:
            raise ResponseError("response exceeds maximum size")
    except ValueError:
        pass

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=_STREAM_CHUNK_SIZE):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_RESPONSE_SIZE:
            raise ResponseError("response exceeds maximum size")
        chunks.append(chunk)
    return b"".join(chunks)


def _form_component(value: Any) -> str | bytes:
    if isinstance(value, (str, bytes)):
        return value
    try:
        return str(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("SERP request component cannot be converted to text") from error


def _escaped_ascii_size(value: int) -> int:
    if (
        48 <= value <= 57
        or 65 <= value <= 90
        or 97 <= value <= 122
        or value in (32, 45, 46, 95, 126)
    ):
        return 1
    return 3


def _utf8_size(value: int) -> int:
    if 55296 <= value <= 57343:
        raise ValueError("SERP request contains invalid Unicode")
    if value <= 127:
        return 1
    if value <= 2047:
        return 2
    if value <= 65535:
        return 3
    return 4


def _encoded_component_size(value: str | bytes, remaining_budget: int) -> int:
    size = 0
    if isinstance(value, bytes):
        for byte in value:
            increment = _escaped_ascii_size(byte)
            if increment > remaining_budget - size:
                return remaining_budget + 1
            size += increment
        return size

    for character in value:
        code_point = ord(character)
        increment = (
            _escaped_ascii_size(code_point)
            if code_point <= 127
            else _utf8_size(code_point) * 3
        )
        if increment > remaining_budget - size:
            return remaining_budget + 1
        size += increment
    return size


def _encode_form(form: Mapping[str, Any]) -> str:
    """Encode a form without allocating an unbounded complete request body."""
    parts: list[str] = []
    encoded_size = 0
    for key, value in form.items():
        encoded_key = _form_component(key)
        encoded_value = _form_component(value)
        separator_size = 1 if parts else 0
        remaining_budget = MAX_REQUEST_SIZE - encoded_size - separator_size - 1
        if remaining_budget < 0:
            raise ValueError("SERP request exceeds maximum size")
        key_size = _encoded_component_size(encoded_key, remaining_budget)
        if key_size > remaining_budget:
            raise ValueError("SERP request exceeds maximum size")
        value_size = _encoded_component_size(encoded_value, remaining_budget - key_size)
        if value_size > remaining_budget - key_size:
            raise ValueError("SERP request exceeds maximum size")

        parts.append(f"{quote_plus(encoded_key)}={quote_plus(encoded_value)}")
        encoded_size += separator_size + key_size + 1 + value_size
    return "&".join(parts)


class SchemaHttpClient:
    """Fetch JSON schema documents from the Thordata schema endpoint."""

    def __init__(self, endpoint: str = DEFAULT_SCHEMA_ENDPOINT, timeout: int = 30) -> None:
        self.endpoint = _endpoint(endpoint, "endpoint")
        self.timeout = _timeout(timeout, "timeout")

    def fetch(self) -> dict[str, Any]:
        response = requests.get(
            self.endpoint,
            headers={"Accept": "application/json"},
            timeout=self.timeout,
            stream=True,
            allow_redirects=False,
        )
        try:
            if not 200 <= response.status_code < 300:
                raise ResponseError(f"schema HTTP {response.status_code} error")
            body = _read_limited(response)
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
                raise ResponseError("invalid schema JSON response") from error
            if not isinstance(payload, dict):
                raise ResponseError("schema JSON response must be an object")
            return payload
        finally:
            response.close()


class _BearerAuth(requests.auth.AuthBase):
    """Explicit authentication takes precedence over Requests' netrc discovery."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        request.headers["Authorization"] = f"Bearer {self._api_key}"
        return request


class SerpClient:
    """POST bounded, form-encoded SERP requests to Thordata."""

    def __init__(
        self,
        api_key: str,
        endpoint: str = DEFAULT_SERP_ENDPOINT,
        timeout: int = 120,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api key must be a nonblank string")
        self.api_key = api_key.strip()
        self.endpoint = _endpoint(endpoint, "endpoint")
        self.timeout = _timeout(timeout, "timeout")

    def request(self, form: Mapping[str, Any]) -> tuple[int, bytes]:
        body = _encode_form(form)

        response = requests.post(
            self.endpoint,
            auth=_BearerAuth(self.api_key),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "Llamaindex",
                "platform": "llamaindex",
                "api-source": "sdk",
            },
            data=body,
            timeout=self.timeout,
            stream=True,
            allow_redirects=False,
        )
        try:
            return int(response.status_code), _read_limited(response)
        finally:
            response.close()
