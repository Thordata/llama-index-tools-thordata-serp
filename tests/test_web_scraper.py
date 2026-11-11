import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
import requests
import responses

from llama_index.tools.thordata_serp.client import MAX_RESPONSE_SIZE, _read_limited
from llama_index.tools.thordata_serp.web_scraper import (
    DEFAULT_MAX_WAIT_SECONDS,
    MAX_WEB_SCRAPER_RESPONSE_SIZE,
    POLL_INTERVAL_SECONDS,
    WEB_SCRAPER_BUILDER_URL,
    WEB_SCRAPER_DOWNLOAD_URL,
    WEB_SCRAPER_TASK_API_BASE_URL,
    ThorDataWebScraperError,
    _parse_download_result,
    run_web_scraper,
    validate_web_scraper_params,
)


API_KEY = "secret-api-key"
TASK_ID = "task-123"
STATUS_URL = f"{WEB_SCRAPER_TASK_API_BASE_URL}/tasks-status"
DOWNLOAD_URL = f"{WEB_SCRAPER_TASK_API_BASE_URL}/tasks-download"


@pytest.mark.parametrize(
    ("spider_id", "params", "expected"),
    [
        (
            "linkedin_company_information_by-url",
            {"url": "https://www.linkedin.com/company/acme"},
            {"url": "https://www.linkedin.com/company/acme"},
        ),
        (
            "linkedin_job_listings_information_by-job-url",
            {"job_url": "https://www.linkedin.com/jobs/view/123"},
            {"job_url": "https://www.linkedin.com/jobs/view/123"},
        ),
        (
            "youtube_video-post_by-keyword",
            {"keyword": "agent", "num_of_posts": 5},
            {"keyword": "agent", "num_of_posts": "5"},
        ),
        (
            "facebook_profile_by-profiles-url",
            {"url": "https://www.facebook.com/acme"},
            {"url": "https://www.facebook.com/acme"},
        ),
        (
            "youtube_video-post_by-explore",
            {"url": "https://www.youtube.com/explore", "all_tabs": False},
            {"url": "https://www.youtube.com/explore", "all_tabs": "flase"},
        ),
        (
            "twitter_profile_by-profileurl",
            {"url": "https://x.com/acme"},
            {"url": "https://x.com/acme"},
        ),
        (
            "amazon_product_by-category-url",
            {"url": "https://www.amazon.com/s?k=books", "page_turning": 2},
            {
                "url": "https://www.amazon.com/s?k=books",
                "page_turning": 2,
                "sort_by": "Best Sellers",
            },
        ),
    ],
)
def test_validate_web_scraper_params_normalizes_supported_spiders(spider_id, params, expected):
    assert validate_web_scraper_params(spider_id, params) == expected


@pytest.mark.parametrize(
    "spider_id, params",
    [
        ("unknown", {}),
        ("linkedin_company_information_by-url", {}),
        ("linkedin_company_information_by-url", {"url": "ftp://example.com"}),
        ("linkedin_company_information_by-url", {"url": "https://user:pass@example.com"}),
        ("youtube_video-post_by-keyword", {"keyword": "x", "num_of_posts": 0}),
        ("youtube_video-post_by-keyword", {"keyword": "x", "num_of_posts": 10**30}),
        ("youtube_video-post_by-explore", {"url": "https://youtube.com", "all_tabs": "false"}),
        (
            "amazon_product_by-category-url",
            {"url": "https://amazon.com", "page_turning": 1, "sort_by": "Other"},
        ),
    ],
)
def test_validate_web_scraper_params_rejects_invalid_input(spider_id, params):
    with pytest.raises(ThorDataWebScraperError):
        validate_web_scraper_params(spider_id, params)


def _download_address(*, task_id=TASK_ID, api_key=API_KEY, query=""):
    suffix = query or f"task_id={task_id}&type=json&api_key={api_key}"
    return f"{WEB_SCRAPER_DOWNLOAD_URL}?{suffix}"


def _request_body(request):
    return request.body.decode() if isinstance(request.body, bytes) else request.body


@responses.activate
def test_run_web_scraper_submits_polls_downloads_and_normalizes_records(monkeypatch):
    responses.add(
        responses.POST,
        WEB_SCRAPER_BUILDER_URL,
        json={"code": 200, "data": {"task_id": TASK_ID}},
    )
    responses.add(
        responses.POST,
        STATUS_URL,
        json={"code": 200, "data": [{"task_id": TASK_ID, "status": "Running"}]},
    )
    responses.add(
        responses.POST,
        STATUS_URL,
        json={"code": 200, "data": [{"task_id": TASK_ID, "status": "Ready"}]},
    )
    responses.add(
        responses.POST,
        DOWNLOAD_URL,
        json={"code": 200, "data": {"tasks_ids": TASK_ID, "download": _download_address()}},
    )
    responses.add(
        responses.GET,
        _download_address(),
        json=[{"title": "Acme", "url": "https://example.com/acme"}],
    )
    monkeypatch.setattr("llama_index.tools.thordata_serp.web_scraper.time.sleep", lambda _: None)

    result = run_web_scraper(
        API_KEY,
        "linkedin_company_information_by-url",
        {"url": "https://www.linkedin.com/company/acme"},
    )

    assert result.status == "completed"
    assert result.task_id == TASK_ID
    assert result.record_count == 1
    assert result.error_record_count == 0
    assert result.records == [{"title": "Acme", "url": "https://example.com/acme"}]
    assert len(responses.calls) == 5

    builder_request = responses.calls[0].request
    builder_form = parse_qs(_request_body(builder_request))
    assert builder_request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert builder_form["spider_name"] == ["linkedin.com"]
    assert builder_form["spider_id"] == ["linkedin_company_information_by-url"]
    assert json.loads(builder_form["spider_parameters"][0]) == [
        {"url": "https://www.linkedin.com/company/acme"}
    ]
    assert builder_form["spider_errors"] == ["true"]
    assert builder_form["file_name"] == ["{{TasksID}}"]
    assert builder_form["api_source"] == ["sdk"]
    assert "api-source" not in builder_form
    assert builder_form["source_platform"] == ["llama"]
    assert "api-source" not in builder_request.headers
    assert "api_source" not in builder_request.headers
    assert "source_platform" not in builder_request.headers

    poll_request = responses.calls[1].request
    assert poll_request.headers["sign"] == API_KEY
    assert parse_qs(_request_body(poll_request)) == {
        "tasks_ids": [TASK_ID],
        "api_source": ["sdk"],
        "source_platform": ["llama"],
    }
    assert "api-source" not in poll_request.headers
    assert "api_source" not in poll_request.headers
    assert "source_platform" not in poll_request.headers
    download_request = responses.calls[3].request
    assert parse_qs(_request_body(download_request)) == {
        "tasks_id": [TASK_ID],
        "type": ["json"],
        "api_source": ["sdk"],
        "source_platform": ["llama"],
    }
    assert "api-source" not in download_request.headers
    assert "api_source" not in download_request.headers
    assert "source_platform" not in download_request.headers


def test_web_scraper_constants_match_the_fixed_contract():
    assert DEFAULT_MAX_WAIT_SECONDS == 300
    assert POLL_INTERVAL_SECONDS == 5
    assert WEB_SCRAPER_BUILDER_URL == "https://scraperapi.thordata.com/builder"
    assert WEB_SCRAPER_TASK_API_BASE_URL == "https://api.thordata.com/api/web-scraper-api"
    assert WEB_SCRAPER_DOWNLOAD_URL == "https://scraperapi.thordata.com/download"
    assert MAX_WEB_SCRAPER_RESPONSE_SIZE > MAX_RESPONSE_SIZE


class _LargeResponse:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self.body = body
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        self.closed = True


def test_web_scraper_response_limit_allows_results_above_serp_limit():
    response = _LargeResponse(b"x" * (MAX_RESPONSE_SIZE + 1))

    body = _read_limited(response, max_size=MAX_WEB_SCRAPER_RESPONSE_SIZE)

    assert len(body) == MAX_RESPONSE_SIZE + 1


def test_web_scraper_request_timeout_maps_to_wait_timeout(monkeypatch):
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("late")))

    with pytest.raises(ThorDataWebScraperError) as raised:
        from llama_index.tools.thordata_serp.web_scraper import _request_json

        _request_json(
            method="POST",
            url="https://example.test/status",
            headers={},
            data={},
            timeout=1,
            stage="poll",
            task_id=TASK_ID,
            spider_id="youtube_video-post_by-keyword",
            api_key=API_KEY,
        )

    assert raised.value.error_code == "wait_timeout"


def test_web_scraper_response_read_timeout_maps_to_wait_timeout(monkeypatch):
    from llama_index.tools.thordata_serp import web_scraper as web_scraper_module

    monkeypatch.setattr(
        requests,
        "request",
        lambda *args, **kwargs: SimpleNamespace(status_code=200, close=lambda: None),
    )
    monkeypatch.setattr(
        web_scraper_module,
        "_read_limited",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("late")),
    )

    with pytest.raises(ThorDataWebScraperError) as raised:
        web_scraper_module._request_json(
            method="POST",
            url="https://example.test/status",
            headers={},
            data={},
            timeout=1,
            stage="poll",
            task_id=TASK_ID,
            spider_id="youtube_video-post_by-keyword",
            api_key=API_KEY,
        )

    assert raised.value.error_code == "wait_timeout"


@pytest.mark.parametrize(
    "download_query",
    [
        "task_id=other&type=json&api_key=secret-api-key",
        "task_id=task-123&type=csv&api_key=secret-api-key",
        "task_id=task-123&type=json&api_key=secret-api-key&host=evil",
    ],
)
@responses.activate
def test_run_web_scraper_rejects_untrusted_download_address(monkeypatch, download_query):
    responses.add(
        responses.POST,
        WEB_SCRAPER_BUILDER_URL,
        json={"code": 200, "data": {"task_id": TASK_ID}},
    )
    responses.add(
        responses.POST,
        STATUS_URL,
        json={"code": 200, "data": [{"task_id": TASK_ID, "status": "Ready"}]},
    )
    responses.add(
        responses.POST,
        DOWNLOAD_URL,
        json={
            "code": 200,
            "data": {"tasks_ids": TASK_ID, "download": _download_address(query=download_query)},
        },
    )
    monkeypatch.setattr("llama_index.tools.thordata_serp.web_scraper.time.sleep", lambda _: None)

    with pytest.raises(ThorDataWebScraperError, match="download address"):
        run_web_scraper(
            API_KEY,
            "linkedin_company_information_by-url",
            {"url": "https://www.linkedin.com/company/acme"},
        )


@responses.activate
def test_run_web_scraper_maps_partial_results_and_redacts_api_key(monkeypatch):
    responses.add(
        responses.POST,
        WEB_SCRAPER_BUILDER_URL,
        json={"code": 200, "data": {"task_id": TASK_ID}},
    )
    responses.add(
        responses.POST,
        STATUS_URL,
        json={"code": 200, "data": [{"task_id": TASK_ID, "status": "Ready"}]},
    )
    responses.add(
        responses.POST,
        DOWNLOAD_URL,
        json={"code": 200, "data": {"tasks_ids": TASK_ID, "download": _download_address()}},
    )
    responses.add(
        responses.GET,
        _download_address(),
        json=[
            {"title": "Good", "success": True},
            {"error": f"token={API_KEY}", "success": False},
        ],
    )
    monkeypatch.setattr("llama_index.tools.thordata_serp.web_scraper.time.sleep", lambda _: None)

    result = run_web_scraper(
        API_KEY,
        "linkedin_company_information_by-url",
        {"url": "https://www.linkedin.com/company/acme"},
    )

    assert result.status == "partial"
    assert result.record_count == 2
    assert result.error_record_count == 1
    assert API_KEY not in json.dumps(result.records)


def test_parse_download_result_preserves_long_record_values_while_redacting_secrets():
    long_text = "x" * 4879

    result = _parse_download_result(
        [{"text": long_text}],
        task_id=TASK_ID,
        spider_id="youtube_video-post_by-keyword",
        input_keys={"keyword"},
        api_key=API_KEY,
    )

    assert result.records[0]["text"] == long_text


def test_parse_download_result_flattens_concatenated_json_arrays():
    payload = json.dumps([{"title": "first"}]) + "\n" + json.dumps([{"title": "second"}])

    result = _parse_download_result(
        payload,
        task_id=TASK_ID,
        spider_id="youtube_video-post_by-keyword",
        input_keys={"keyword"},
        api_key=API_KEY,
    )

    assert result.records == [{"title": "first"}, {"title": "second"}]

