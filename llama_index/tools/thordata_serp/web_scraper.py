"""Synchronous Thordata Web Scraper lifecycle helpers."""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal
from urllib.parse import parse_qsl, quote, quote_plus, urlparse

import requests

from .client import _read_limited


WEB_SCRAPER_BUILDER_URL: Final = "https://scraperapi.thordata.com/builder"
WEB_SCRAPER_TASK_API_BASE_URL: Final = "https://api.thordata.com/api/web-scraper-api"
WEB_SCRAPER_DOWNLOAD_URL: Final = "https://scraperapi.thordata.com/download"
DEFAULT_MAX_WAIT_SECONDS: Final = 300
POLL_INTERVAL_SECONDS: Final = 5
MAX_WEB_SCRAPER_RESPONSE_SIZE: Final = 50 << 20
_DIAGNOSTIC_SLOTS = threading.BoundedSemaphore(4)
_WEB_SCRAPER_SOURCE_FIELDS: Final[dict[str, str]] = {
    "api_source": "sdk",
    "source_platform": "llama",
}

SUPPORTED_WEB_SCRAPERS: Final[tuple[str, ...]] = (
    "linkedin_company_information_by-url",
    "linkedin_job_listings_information_by-job-url",
    "youtube_video-post_by-keyword",
    "facebook_profile_by-profiles-url",
    "youtube_video-post_by-explore",
    "twitter_profile_by-profileurl",
    "amazon_product_by-category-url",
)

_SORT_BY_VALUES: Final[tuple[str, ...]] = (
    "Best Sellers",
    "Newest Arrivals",
    "Avg. Customer Review",
    "Price: High to Low",
    "Price: Low to High",
    "Featured",
)

SPIDER_DEFINITIONS: Final[dict[str, dict[str, Any]]] = {
    "linkedin_company_information_by-url": {
        "spider_name": "linkedin.com",
        "fields": {"url": {"type": "url", "required": True}},
    },
    "linkedin_job_listings_information_by-job-url": {
        "spider_name": "linkedin.com",
        "fields": {"job_url": {"type": "url", "required": True}},
    },
    "youtube_video-post_by-keyword": {
        "spider_name": "youtube.com",
        "fields": {
            "keyword": {"type": "string", "required": True},
            "num_of_posts": {"type": "positive_integer_string", "required": False},
        },
    },
    "facebook_profile_by-profiles-url": {
        "spider_name": "facebook.com",
        "fields": {
            "url": {"type": "url", "required": True},
        },
    },
    "youtube_video-post_by-explore": {
        "spider_name": "youtube.com",
        "fields": {
            "url": {"type": "url", "required": True},
            "all_tabs": {"type": "all_tabs", "required": False},
        },
    },
    "twitter_profile_by-profileurl": {
        "spider_name": "x.com",
        "fields": {
            "url": {"type": "url", "required": True},
        },
    },
    "amazon_product_by-category-url": {
        "spider_name": "amazon.com",
        "fields": {
            "url": {"type": "url", "required": True},
            "page_turning": {"type": "positive_integer", "required": False},
            "sort_by": {
                "type": "option",
                "required": False,
                "default": "Best Sellers",
                "options": _SORT_BY_VALUES,
                "aliases": {
                    "Average Rating": "Avg. Customer Review",
                    "Featured Recommendations": "Featured",
                },
            },
        },
    },
}

_ERROR_MESSAGES: Final[dict[str, str]] = {
    "invalid_parameters": "Web Scraper parameters are invalid.",
    "submission_unknown": "Web Scraper submission outcome is unknown.",
    "submission_failed": "Web Scraper submission was rejected.",
    "poll_failed": "Web Scraper task status could not be verified.",
    "scrape_failed": "Web Scraper returned only failed records or the task failed.",
    "download_url_invalid": "Web Scraper returned an invalid download address.",
    "download_failed": "Web Scraper results could not be downloaded.",
    "invalid_result": "Web Scraper results are not valid JSON records.",
    "wait_timeout": "Web Scraper exceeded the maximum wait time.",
    "protocol_error": "Web Scraper returned an incompatible response.",
    "authentication_failed": "Web Scraper authentication failed.",
    "permission_denied": "Web Scraper permission was denied.",
    "insufficient_balance": "Web Scraper account balance is insufficient.",
    "request_failed": "Web Scraper request failed.",
}
_TASK_ID_RE = re.compile(r"^[\w-]{1,200}$")
_SPIDER_ID_RE = re.compile(r"^[\w-]{1,120}$")


class ThorDataWebScraperError(RuntimeError):
    """A safe, stage-aware error from the Web Scraper lifecycle."""

    def __init__(
        self,
        error_code: str,
        *,
        stage: str,
        task_id: str | None = None,
        spider_id: str | None = None,
        reason: Any = None,
        secrets: tuple[str, ...] = (),
        service_code: str | int | None = None,
    ) -> None:
        self.error_code = error_code if error_code in _ERROR_MESSAGES else "request_failed"
        self.stage = stage
        self.task_id = (
            task_id
            if task_id
            and _TASK_ID_RE.fullmatch(task_id)
            and not any(secret and secret in task_id for secret in secrets)
            else None
        )
        self.spider_id = spider_id if spider_id and _SPIDER_ID_RE.fullmatch(spider_id) else None
        detail = _redact_text(reason, secrets)
        self.service_code = (
            _redact_text(service_code, secrets) if isinstance(service_code, str)
            else service_code if type(service_code) is int else None
        )
        message = _ERROR_MESSAGES[self.error_code]
        if detail:
            message = f"{message} {detail}"
        if self.task_id:
            message = f"{message} Task ID: {self.task_id}."
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WebScraperResponse:
    """Completed Web Scraper records and task metadata."""

    records: list[dict[str, Any]]
    task_id: str
    spider_id: str
    status: Literal["completed", "partial", "empty"]
    record_count: int
    error_record_count: int = 0

    @property
    def result(self) -> list[dict[str, Any]]:
        """Compatibility alias used by the other Thordata Python SDK."""
        return self.records


def _redact_text(value: Any, secrets: tuple[str, ...]) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(
        r"https?://[^\s\"<>]*(?:api_key|access_token|authorization|sign|token)=[^\s\"<>]*",
        "[redacted URL]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(api[_-]?key|authorization|sign|token|password)\s*[:=]\s*[^\s,;\"}]+",
        r"\1: [redacted]",
        text,
        flags=re.IGNORECASE,
    )
    text = _redact_value_text(text, secrets)
    return re.sub(r"[\r\n\t]+", " ", text).strip()[:250]


def _redact_value_text(
    value: str,
    secrets: tuple[str, ...],
    *,
    protect_short_secrets: bool = False,
) -> str:
    """Redact secrets without changing the length or content of other values."""
    text = value
    candidates: list[tuple[str, str]] = []
    for raw_secret in secrets:
        if not raw_secret:
            continue
        short_secret = protect_short_secrets and len(raw_secret) < 8
        variants = (
            (raw_secret,)
            if short_secret
            else (raw_secret, quote(raw_secret, safe=""), quote_plus(raw_secret))
        )
        candidates.extend((raw_secret, secret) for secret in set(variants))
    for raw_secret, secret in sorted(candidates, key=lambda item: len(item[1]), reverse=True):
        pattern = "".join(
            f"[{char.lower()}{char.upper()}]"
            if char.isalpha()
            and index >= 1
            and (
                secret[max(0, index - 2) : index].startswith("%")
                or secret[index - 1] == "%"
            )
            else re.escape(char)
            for index, char in enumerate(secret)
        )
        if protect_short_secrets and len(raw_secret) < 8:
            pattern = rf"(?<![\w]){pattern}(?![\w])"
        text = re.sub(pattern, "[redacted]", text)
    return text


def _redact_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _redact_value_text(value, secrets, protect_short_secrets=True)
    if isinstance(value, list):
        return [_redact_value(item, secrets) for item in value]
    if isinstance(value, Mapping):
        blocked = {"api_key", "access_token", "authorization", "sign", "password", "secret"}
        return {
            str(key): _redact_value(item, secrets)
            for key, item in value.items()
            if str(key).lower() not in blocked
        }
    return value


def _invalid(message: str, *, spider_id: str | None = None) -> ThorDataWebScraperError:
    return ThorDataWebScraperError(
        "invalid_parameters",
        stage="validate",
        spider_id=spider_id,
        reason=message,
    )


def _positive_integer(value: Any, key: str, *, spider_id: str) -> int:
    if isinstance(value, bool):
        raise _invalid(f"{key} must be a positive integer", spider_id=spider_id)
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        value = value.lstrip("0") or "0"
        if len(value) > 16 or not value.isdigit():
            raise _invalid(f"{key} must be a positive integer", spider_id=spider_id)
        number = int(value)
    else:
        raise _invalid(f"{key} must be a positive integer", spider_id=spider_id)
    if not 1 <= number <= 9007199254740991:
        raise _invalid(f"{key} must be a positive integer", spider_id=spider_id)
    return number


def _validate_url(value: Any, key: str, *, spider_id: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise _invalid(f"{key} must be an HTTP or HTTPS URL", spider_id=spider_id)
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
    except ValueError:
        hostname = None
        parsed = None
    if (
        parsed is None
        or parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
    ):
        raise _invalid(f"{key} must be an HTTP or HTTPS URL", spider_id=spider_id)
    return value


def validate_web_scraper_params(
    spider_id: str,
    params: Mapping[str, Any] | None,
) -> dict[str, str | int]:
    """Validate one of the seven supported spiders and normalize wire values."""

    definition = SPIDER_DEFINITIONS.get(spider_id)
    if definition is None:
        raise _invalid(f"Unsupported Web Scraper spider {spider_id!r}", spider_id=spider_id)
    if not isinstance(params, Mapping):
        raise _invalid("Web Scraper params must be a mapping", spider_id=spider_id)
    fields = definition["fields"]
    unknown = set(params) - set(fields)
    if unknown:
        raise _invalid(
            f"Unknown Web Scraper parameters: {', '.join(sorted(map(str, unknown)))}",
            spider_id=spider_id,
        )

    normalized: dict[str, str | int] = {}
    for key, field in fields.items():
        value = params[key] if key in params else field.get("default")
        if value is None or value == "":
            # Null page counts use the server default, just like omitted values.
            if key == "page_turning" and value == "":
                raise _invalid(f"{key} must be a positive integer", spider_id=spider_id)
            if field["required"]:
                raise _invalid(
                    f"Missing required Web Scraper parameter: {key}", spider_id=spider_id
                )
            continue
        field_type = field["type"]
        if field_type == "url":
            normalized[key] = _validate_url(value, key, spider_id=spider_id)
        elif field_type == "string":
            if not isinstance(value, str) or not value.strip():
                raise _invalid(f"{key} must be a non-empty string", spider_id=spider_id)
            normalized[key] = value
        elif field_type == "positive_integer_string":
            normalized[key] = str(_positive_integer(value, key, spider_id=spider_id))
        elif field_type == "positive_integer":
            normalized[key] = _positive_integer(value, key, spider_id=spider_id)
        elif field_type == "all_tabs":
            if not isinstance(value, bool):
                raise _invalid("all_tabs must be a boolean", spider_id=spider_id)
            normalized[key] = "true" if value else "flase"
        elif field_type == "option":
            if isinstance(value, str):
                value = field.get("aliases", {}).get(value, value)
            if not isinstance(value, str) or value not in field["options"]:
                raise _invalid(f"{key} must be one of the supported options", spider_id=spider_id)
            normalized[key] = value
        else:
            raise _invalid(
                f"Unsupported Web Scraper parameter type for {key}", spider_id=spider_id
            )
    return normalized


def _coerce_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        try:
            return int(value.strip())
        except (OverflowError, ValueError):
            return None
    return None


def _business_error_code(code: int | None, fallback: str) -> str:
    if code in {401, 10013, 10014, 10047, 10048}:
        return "authentication_failed"
    if code == 402:
        return "insufficient_balance"
    if code == 403:
        return "permission_denied"
    return fallback


def _reason(payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        return None
    return (
        payload.get("msg")
        or payload.get("message")
        or payload.get("error")
        or (payload.get("data") if isinstance(payload.get("data"), str) else None)
    )


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, RecursionError):
        try:
            return _parse_concatenated_json(value)
        except (TypeError, ValueError, RecursionError):
            return value


def _parse_concatenated_json(text: str) -> Any:
    """Parse one JSON document or concatenate multiple JSON arrays."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, index = decoder.raw_decode(text, index)
        values.append(value)
    if len(values) == 1:
        return values[0]
    if values and all(isinstance(value, list) for value in values):
        return [item for value in values for item in value]
    raise ValueError("Expected one JSON document or concatenated JSON arrays")


def _request_json(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    data: Mapping[str, str] | None,
    timeout: float,
    stage: str,
    task_id: str | None,
    spider_id: str,
    api_key: str,
    deadline: float | None = None,
) -> Any:
    def check_budget() -> None:
        if deadline is not None:
            _remaining(deadline, stage=stage, task_id=task_id, spider_id=spider_id)

    check_budget()
    try:
        response = requests.request(
            method,
            url,
            headers=dict(headers),
            data=data,
            timeout=max(0.001, float(timeout)),
            stream=True,
            allow_redirects=False,
        )
    except requests.Timeout as error:
        error_code = "submission_unknown" if stage == "submit" else "wait_timeout"
        raise ThorDataWebScraperError(
            error_code,
            stage=stage,
            task_id=task_id,
            spider_id=spider_id,
            reason=error,
            secrets=(api_key,),
        ) from error
    except requests.RequestException as error:
        error_code = "submission_unknown" if stage == "submit" else "request_failed"
        raise ThorDataWebScraperError(
            error_code,
            stage=stage,
            task_id=task_id,
            spider_id=spider_id,
            reason=error,
            secrets=(api_key,),
        ) from error

    try:
        try:
            status_code = int(response.status_code)
            body = _read_limited(
                response, max_size=MAX_WEB_SCRAPER_RESPONSE_SIZE, check_deadline=check_budget,
            )
        except ThorDataWebScraperError:
            raise
        except requests.Timeout as error:
            error_code = "submission_unknown" if stage == "submit" else "wait_timeout"
            raise ThorDataWebScraperError(
                error_code,
                stage=stage,
                task_id=task_id,
                spider_id=spider_id,
                reason=error,
                secrets=(api_key,),
            ) from error
        except Exception as error:
            raise ThorDataWebScraperError(
                "download_failed" if stage == "download" else "request_failed",
                stage=stage,
                task_id=task_id,
                spider_id=spider_id,
                reason=error,
                secrets=(api_key,),
            ) from error
    finally:
        response.close()

    if status_code < 200 or status_code >= 300:
        fallback = "submission_unknown" if stage == "submit" and status_code >= 500 else "request_failed"
        raise ThorDataWebScraperError(
            _business_error_code(status_code, fallback),
            stage=stage,
            task_id=task_id,
            spider_id=spider_id,
            reason=f"HTTP {status_code}",
            secrets=(api_key,),
        )
    try:
        payload = _parse_json_value(body.decode("utf-8"))
        check_budget()
        return payload
    except (UnicodeDecodeError, RecursionError) as error:
        raise ThorDataWebScraperError(
            "protocol_error",
            stage=stage,
            task_id=task_id,
            spider_id=spider_id,
            reason="Response was not valid JSON",
            secrets=(api_key,),
        ) from error


def _check_business_envelope(
    payload: Any,
    *,
    stage: str,
    task_id: str | None,
    spider_id: str,
    api_key: str,
    expected_code: int = 200,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ThorDataWebScraperError(
            "protocol_error", stage=stage, task_id=task_id, spider_id=spider_id
        )
    code = _coerce_code(payload.get("code"))
    if code != expected_code:
        raise ThorDataWebScraperError(
            _business_error_code(
                code, "protocol_error" if stage != "submit" else "submission_failed"
            ),
            stage=stage,
            task_id=task_id,
            spider_id=spider_id,
            reason=_reason(payload),
            secrets=(api_key,),
        )
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ThorDataWebScraperError(
            "protocol_error", stage=stage, task_id=task_id, spider_id=spider_id
        )
    return data


def _validate_max_wait(max_wait_seconds: Any, *, spider_id: str) -> float:
    if isinstance(max_wait_seconds, bool) or not isinstance(max_wait_seconds, (int, float)):
        raise _invalid("max_wait_seconds must be between 1 and 300", spider_id=spider_id)
    if not 0 < max_wait_seconds <= DEFAULT_MAX_WAIT_SECONDS:
        raise _invalid("max_wait_seconds must be between 1 and 300", spider_id=spider_id)
    return float(max_wait_seconds)


def _validate_file_name(file_name: Any, *, spider_id: str) -> str:
    if not isinstance(file_name, str) or not file_name.strip():
        raise _invalid("file_name must be a non-empty string", spider_id=spider_id)
    return file_name


def _remaining(deadline: float, *, stage: str, task_id: str | None, spider_id: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ThorDataWebScraperError(
            "wait_timeout", stage=stage, task_id=task_id, spider_id=spider_id
        )
    return remaining


def _validate_download_url(value: Any, task_id: str, spider_id: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ThorDataWebScraperError(
            "download_url_invalid", stage="download_url", task_id=task_id, spider_id=spider_id
        )
    try:
        parsed = urlparse(value)
        trusted = urlparse(WEB_SCRAPER_DOWNLOAD_URL)
    except ValueError as error:
        raise ThorDataWebScraperError(
            "download_url_invalid", stage="download_url", task_id=task_id, spider_id=spider_id
        ) from error
    if (
        parsed.scheme != trusted.scheme
        or parsed.netloc != trusted.netloc
        or parsed.path != trusted.path
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ThorDataWebScraperError(
            "download_url_invalid", stage="download_url", task_id=task_id, spider_id=spider_id
        )
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if len({key for key, _ in query}) != len(query) or any(
        key not in {"task_id", "type", "api_key"} for key, _ in query
    ):
        raise ThorDataWebScraperError(
            "download_url_invalid", stage="download_url", task_id=task_id, spider_id=spider_id
        )
    values = dict(query)
    if values.get("task_id") != task_id or values.get("type") != "json":
        raise ThorDataWebScraperError(
            "download_url_invalid", stage="download_url", task_id=task_id, spider_id=spider_id
        )
    return value


def _meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_meaningful(item) for item in value)
    if isinstance(value, Mapping):
        return any(_meaningful(item) for item in value.values())
    return True


def _failure_code(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return False
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return True
    return numeric not in {0, 200}


def _failed_record(record: Mapping[str, Any]) -> bool:
    error = record.get("error")
    return (
        record.get("success") is False
        or (error is not False and error != 0 and _meaningful(error))
        or _failure_code(record.get("error_code"))
    )


def _parse_download_result(
    payload: Any,
    *,
    task_id: str,
    spider_id: str,
    input_keys: set[str],
    api_key: str,
) -> WebScraperResponse:
    value = _parse_json_value(payload)
    if (
        isinstance(value, Mapping)
        and _coerce_code(value.get("code")) not in {None, 200}
        and ("msg" in value or "message" in value)
    ):
        raise ThorDataWebScraperError(
            _business_error_code(_coerce_code(value.get("code")), "scrape_failed"),
            stage="parse",
            task_id=task_id,
            spider_id=spider_id,
            reason=_reason(value),
            secrets=(api_key,),
        )
    rows = value if isinstance(value, list) else [value] if isinstance(value, Mapping) else None
    if rows is None or any(not isinstance(row, Mapping) for row in rows):
        raise ThorDataWebScraperError(
            "invalid_result", stage="parse", task_id=task_id, spider_id=spider_id
        )
    records = [dict(row) for row in rows]
    errors = sum(1 for row in records if _failed_record(row))
    if records and errors == len(records):
        raise ThorDataWebScraperError(
            "scrape_failed",
            stage="parse",
            task_id=task_id,
            spider_id=spider_id,
            reason=records[0].get("error"),
            secrets=(api_key,),
        )
    meaningful = any(
        _meaningful(item)
        for row in records
        for key, item in row.items()
        if key not in input_keys
        and key not in {"input", "input_url", "url", "job_url", "keyword", "success", "error", "error_code"}
    )
    status: Literal["completed", "partial", "empty"] = (
        "partial" if errors else "completed" if meaningful else "empty"
    )
    return WebScraperResponse(
        records=_redact_value(records, (api_key,)),
        task_id=task_id,
        spider_id=spider_id,
        status=status,
        record_count=len(records),
        error_record_count=errors,
    )


def _task_id_from_submission(payload: Any, *, spider_id: str, api_key: str) -> str:
    data = _check_business_envelope(
        payload,
        stage="submit",
        task_id=None,
        spider_id=spider_id,
        api_key=api_key,
    )
    task_id = data.get("task_id")
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id) or api_key in task_id:
        raise ThorDataWebScraperError("submission_unknown", stage="submit", spider_id=spider_id)
    return task_id


def _download_task_result(
    api_key: str, task_id: str, spider_id: str, deadline: float,
) -> tuple[Any, tuple[str, ...]]:
    address = _request_json(
        method="POST",
        url=f"{WEB_SCRAPER_TASK_API_BASE_URL}/tasks-download",
        headers={"sign": api_key, "Content-Type": "application/x-www-form-urlencoded"},
        data={**_WEB_SCRAPER_SOURCE_FIELDS, "tasks_id": task_id, "type": "json"},
        timeout=_remaining(deadline, stage="download_url", task_id=task_id, spider_id=spider_id),
        stage="download_url", task_id=task_id, spider_id=spider_id, api_key=api_key,
        deadline=deadline,
    )
    address_data = _check_business_envelope(
        address, stage="download_url", task_id=task_id, spider_id=spider_id, api_key=api_key,
    )
    if address_data.get("tasks_ids") != task_id:
        raise ThorDataWebScraperError(
            "protocol_error", stage="download_url", task_id=task_id, spider_id=spider_id
        )
    download_url = _validate_download_url(address_data.get("download"), task_id, spider_id)
    secrets = (api_key, dict(parse_qsl(urlparse(download_url).query)).get("api_key", ""))
    downloaded = _request_json(
        method="GET", url=download_url, headers={"Accept": "application/json"}, data=None,
        timeout=_remaining(deadline, stage="download", task_id=task_id, spider_id=spider_id),
        stage="download", task_id=task_id, spider_id=spider_id, api_key=api_key,
        deadline=deadline,
    )
    _remaining(deadline, stage="parse", task_id=task_id, spider_id=spider_id)
    return downloaded, secrets


def _failed_task_error(
    api_key: str, task_id: str, spider_id: str, status: Mapping[str, Any],
    deadline: float, include_errors: bool,
) -> ThorDataWebScraperError:
    original_reason = _reason(status)
    original_code = status.get("error_code")
    fallback = ThorDataWebScraperError(
        "scrape_failed", stage="poll", task_id=task_id, spider_id=spider_id,
        reason=original_reason, service_code=original_code, secrets=(api_key,),
    )
    if not include_errors or time.monotonic() >= deadline:
        return fallback
    slots = _DIAGNOSTIC_SLOTS
    if not slots.acquire(blocking=False):
        return fallback
    diagnostic_deadline = min(deadline, time.monotonic() + 15)
    outcome: queue.Queue[ThorDataWebScraperError] = queue.Queue(maxsize=1)

    def diagnose() -> None:
        try:
            result = _diagnose_failed_task(
                api_key, task_id, spider_id, original_reason, original_code,
                diagnostic_deadline, fallback,
            )
        except Exception:
            result = fallback
        finally:
            slots.release()
        outcome.put(result)

    # Socket inactivity timeouts cannot bound trickling responses. Only read-only
    # diagnostics run in a bounded daemon worker; submissions stay on the caller.
    worker = threading.Thread(target=diagnose, name="thordata-failure-diagnostic", daemon=True)
    try:
        worker.start()
    except RuntimeError:
        slots.release()
        return fallback
    try:
        result = outcome.get(timeout=max(0, diagnostic_deadline - time.monotonic()))
    except queue.Empty:
        return fallback
    return result if time.monotonic() < diagnostic_deadline else fallback


def _diagnose_failed_task(
    api_key: str, task_id: str, spider_id: str, original_reason: Any,
    original_code: Any, diagnostic_deadline: float, fallback: ThorDataWebScraperError,
) -> ThorDataWebScraperError:
    try:
        downloaded, secrets = _download_task_result(api_key, task_id, spider_id, diagnostic_deadline)
        value = _parse_json_value(downloaded)
        if isinstance(value, Mapping) and "code" in value and _failure_code(value["code"]):
            reason, code = _reason(value), value["code"]
        else:
            if isinstance(value, Mapping) and "code" in value and "data" in value:
                value = _parse_json_value(value["data"])
            rows = value if isinstance(value, list) else [value]
            failures = [row for row in rows if isinstance(row, Mapping) and _failed_record(row)]
            failed = next((row for row in failures if _meaningful(_reason(row))), None)
            if failed is None:
                failed = next((row for row in failures if _failure_code(row.get("error_code"))), None)
            if failed is None:
                return fallback
            reason, code = _reason(failed), failed.get("error_code")
        _remaining(diagnostic_deadline, stage="parse", task_id=task_id, spider_id=spider_id)
        return ThorDataWebScraperError(
            "scrape_failed", stage="poll", task_id=task_id, spider_id=spider_id,
            reason=reason or original_reason,
            service_code=code if code is not None else original_code, secrets=secrets,
        )
    except (ThorDataWebScraperError, ValueError, TypeError, RecursionError):
        # Diagnostics are best effort; an established task failure stays a failure.
        return fallback


def run_web_scraper(
    api_key: str,
    spider_id: str,
    params: Mapping[str, Any] | None,
    *,
    max_wait_seconds: int | float = DEFAULT_MAX_WAIT_SECONDS,
    include_errors: bool = True,
    file_name: str = "{{TasksID}}",
) -> WebScraperResponse:
    """Submit one Web Scraper task and synchronously return its records."""

    if not isinstance(api_key, str) or not api_key.strip():
        raise ThorDataWebScraperError("invalid_parameters", stage="validate", reason="api_key is required")
    api_key = api_key.strip()
    normalized = validate_web_scraper_params(spider_id, params)
    wait_seconds = _validate_max_wait(max_wait_seconds, spider_id=spider_id)
    if not isinstance(include_errors, bool):
        raise _invalid("include_errors must be a boolean", spider_id=spider_id)
    safe_file_name = _validate_file_name(file_name, spider_id=spider_id)
    deadline = time.monotonic() + wait_seconds

    submission = _request_json(
        method="POST",
        url=WEB_SCRAPER_BUILDER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            **_WEB_SCRAPER_SOURCE_FIELDS,
            "spider_name": SPIDER_DEFINITIONS[spider_id]["spider_name"],
            "spider_id": spider_id,
            "spider_parameters": json.dumps([normalized], ensure_ascii=False, separators=(",", ":")),
            "spider_errors": str(include_errors).lower(),
            "file_name": safe_file_name,
        },
        timeout=_remaining(deadline, stage="submit", task_id=None, spider_id=spider_id),
        stage="submit",
        task_id=None,
        spider_id=spider_id,
        api_key=api_key,
    )
    task_id = _task_id_from_submission(submission, spider_id=spider_id, api_key=api_key)

    while True:
        poll = _request_json(
            method="POST",
            url=f"{WEB_SCRAPER_TASK_API_BASE_URL}/tasks-status",
            headers={"sign": api_key, "Content-Type": "application/x-www-form-urlencoded"},
            data={**_WEB_SCRAPER_SOURCE_FIELDS, "tasks_ids": task_id},
            timeout=_remaining(deadline, stage="poll", task_id=task_id, spider_id=spider_id),
            stage="poll",
            task_id=task_id,
            spider_id=spider_id,
            api_key=api_key,
        )
        if not isinstance(poll, Mapping) or _coerce_code(poll.get("code")) != 200:
            raise ThorDataWebScraperError(
                "protocol_error",
                stage="poll",
                task_id=task_id,
                spider_id=spider_id,
                reason=_reason(poll),
                secrets=(api_key,),
            )
        rows = poll.get("data")
        matches = [
            row for row in rows or [] if isinstance(row, Mapping) and row.get("task_id") == task_id
        ] if isinstance(rows, list) else []
        if len(matches) != 1 or matches[0].get("status") not in {"Running", "Ready", "Failed"}:
            raise ThorDataWebScraperError(
                "protocol_error", stage="poll", task_id=task_id, spider_id=spider_id
            )
        status = matches[0]["status"]
        if status == "Failed":
            raise _failed_task_error(
                api_key, task_id, spider_id, matches[0], deadline, include_errors,
            )
        if status == "Ready":
            break
        time.sleep(min(POLL_INTERVAL_SECONDS, _remaining(deadline, stage="poll", task_id=task_id, spider_id=spider_id)))

    downloaded, _ = _download_task_result(api_key, task_id, spider_id, deadline)
    return _parse_download_result(
        downloaded,
        task_id=task_id,
        spider_id=spider_id,
        input_keys=set(normalized),
        api_key=api_key,
    )


__all__ = [
    "DEFAULT_MAX_WAIT_SECONDS",
    "MAX_WEB_SCRAPER_RESPONSE_SIZE",
    "POLL_INTERVAL_SECONDS",
    "SPIDER_DEFINITIONS",
    "SUPPORTED_WEB_SCRAPERS",
    "ThorDataWebScraperError",
    "WebScraperResponse",
    "WEB_SCRAPER_BUILDER_URL",
    "WEB_SCRAPER_DOWNLOAD_URL",
    "WEB_SCRAPER_TASK_API_BASE_URL",
    "run_web_scraper",
    "validate_web_scraper_params",
]
