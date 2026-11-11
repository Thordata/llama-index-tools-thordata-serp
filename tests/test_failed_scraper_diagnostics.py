import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, quote

import pytest
import requests
import responses

from llama_index.tools.thordata_serp import ThordataSerpToolSpec
from llama_index.tools.thordata_serp import web_scraper as ws


KEY = "fixture-secret-key"
DOWNLOAD_KEY = "download-private/key"
TASK = "failed-task-1"
SPIDER = "linkedin_company_information_by-url"
STATUS_URL = f"{ws.WEB_SCRAPER_TASK_API_BASE_URL}/tasks-status"
ADDRESS_URL = f"{ws.WEB_SCRAPER_TASK_API_BASE_URL}/tasks-download"
DOWNLOAD = f"{ws.WEB_SCRAPER_DOWNLOAD_URL}?task_id={TASK}&type=json&api_key={quote(DOWNLOAD_KEY, safe='')}"


def failed_task(row=None):
    responses.add(responses.POST, ws.WEB_SCRAPER_BUILDER_URL,
                  json={"code": 200, "data": {"task_id": TASK}})
    responses.add(responses.POST, STATUS_URL,
                  json={"code": 200, "data": [{"task_id": TASK, "status": "Failed", **(row or {})}]})


def address(url=DOWNLOAD, task=TASK):
    responses.add(responses.POST, ADDRESS_URL,
                  json={"code": 200, "data": {"tasks_ids": task, "download": url}})


def scrape(**options):
    return json.loads(ThordataSerpToolSpec(KEY).scrape(
        SPIDER, {"url": "https://www.linkedin.com/company/example/"}, **options))


def assert_failed(result):
    assert result["ok"] is False
    assert result["error"]["code"] == "scrape_failed"
    assert result["error"]["stage"] == "poll"
    assert result["error"]["task_id"] == TASK
    assert result["error"]["spider_id"] == SPIDER
    assert sum(c.request.url == ws.WEB_SCRAPER_BUILDER_URL for c in responses.calls) == 1
    for secret in (KEY, DOWNLOAD_KEY, quote(DOWNLOAD_KEY, safe="")):
        assert secret not in json.dumps(result)


@pytest.mark.parametrize("service_code", [300, 404, "NOT_FOUND"])
@responses.activate
def test_failed_task_downloads_safe_reason_and_service_code_once(service_code):
    failed_task()
    address()
    reason = f"Target not found {KEY} {DOWNLOAD_KEY} {quote(DOWNLOAD_KEY, safe='')}"
    responses.add(responses.GET, DOWNLOAD, json=[{"error": reason, "error_code": service_code}])

    output = scrape()

    assert_failed(output)
    assert "Target not found" in output["error"]["message"]
    assert output["error"]["service_code"] == service_code
    assert len(responses.calls) == 4
    form = parse_qs(responses.calls[2].request.body)
    assert form == {"api_source": ["sdk"], "source_platform": ["llama"],
                    "tasks_id": [TASK], "type": ["json"]}


@responses.activate
def test_failed_task_keeps_status_reason_when_diagnostics_disabled():
    failed_task({"error": f"Original failure {KEY}", "error_code": 300})
    output = scrape(include_errors=False)
    assert_failed(output)
    assert "Original failure" in output["error"]["message"]
    assert output["error"]["service_code"] == 300
    assert len(responses.calls) == 2


@pytest.mark.parametrize("failure", ["http", "timeout", "malformed", "bad-url", "wrong-task", "success-records", "empty"])
@responses.activate
def test_diagnostics_do_not_replace_known_failure(failure):
    failed_task({"message": "Original failure", "error_code": "ORIGINAL"})
    if failure == "http":
        responses.add(responses.POST, ADDRESS_URL, status=503)
    elif failure == "timeout":
        responses.add(responses.POST, ADDRESS_URL, body=requests.Timeout("diagnostic timeout"))
    elif failure == "bad-url":
        address("https://untrusted.example/download")
    elif failure == "wrong-task":
        address(task="another-task")
    else:
        address()
        if failure == "malformed":
            responses.add(responses.GET, DOWNLOAD, body="not JSON")
        else:
            responses.add(responses.GET, DOWNLOAD,
                          json=[] if failure == "empty" else [{"name": "Unexpected success"}])
    output = scrape()
    assert_failed(output)
    assert "Original failure" in output["error"]["message"]
    assert output["error"]["service_code"] == "ORIGINAL"
    assert len(responses.calls) == (3 if failure in {"http", "timeout", "bad-url", "wrong-task"} else 4)


@responses.activate
def test_failed_status_without_remaining_budget_skips_diagnostics(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ws.time, "monotonic", lambda: now[0])
    responses.add(responses.POST, ws.WEB_SCRAPER_BUILDER_URL,
                  json={"code": 200, "data": {"task_id": TASK}})

    def poll(request):
        now[0] = 1.0
        return 200, {}, json.dumps({"code": 200, "data": [
            {"task_id": TASK, "status": "Failed", "error": "Original failure", "error_code": 300}]})

    responses.add_callback(responses.POST, STATUS_URL, callback=poll)
    output = scrape(max_wait_seconds=1)
    assert_failed(output)
    assert output["error"]["service_code"] == 300
    assert len(responses.calls) == 2


@responses.activate
def test_diagnostic_download_uses_only_remaining_budget(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ws.time, "monotonic", lambda: now[0])
    failed_task({"error": "Original failure", "error_code": 300})

    def get_address(request):
        now[0] = 1.0
        return 200, {}, json.dumps({"code": 200, "data": {"tasks_ids": TASK, "download": DOWNLOAD}})

    responses.add_callback(responses.POST, ADDRESS_URL, callback=get_address)
    output = scrape(max_wait_seconds=1)
    assert_failed(output)
    assert output["error"]["service_code"] == 300
    assert len(responses.calls) == 3


@responses.activate
def test_failed_task_parses_chunked_diagnostics_and_redacts_service_code():
    failed_task()
    address()
    payload = '[{"name":"first"}]' + json.dumps([{"error": "Not collected", "error_code": DOWNLOAD_KEY}])
    responses.add(responses.GET, DOWNLOAD, body=payload)
    output = scrape()
    assert_failed(output)
    assert "Not collected" in output["error"]["message"]
    assert output["error"]["service_code"] == "[redacted]"


@responses.activate
def test_failed_task_uses_detailed_error_from_later_record():
    failed_task()
    address()
    responses.add(responses.GET, DOWNLOAD, json=[
        {"success": False},
        {"error_code": 300},
        {"error": "Detailed upstream failure", "error_code": "REAL_REASON"},
    ])
    output = scrape()
    assert_failed(output)
    assert "Detailed upstream failure" in output["error"]["message"]
    assert output["error"]["service_code"] == "REAL_REASON"


@responses.activate
def test_diagnostic_body_read_stops_after_budget_expires(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ws.time, "monotonic", lambda: now[0])
    failed_task({"error": "Original failure", "error_code": 300})
    address()
    chunks = []

    class SlowResponse:
        status_code = 200
        headers = {}
        closed = False

        def iter_content(self, chunk_size):
            chunks.append(1)
            now[0] = 16.0
            yield b'[{"error":"too late"}]'
            chunks.append(2)
            yield b'[{"error":"must not read"}]'

        def close(self):
            self.closed = True

    slow = SlowResponse()
    request = ws.requests.request

    def transport(method, url, **kwargs):
        if method == "GET":
            assert 0 < kwargs["timeout"] <= 15
            return slow
        return request(method, url, **kwargs)

    monkeypatch.setattr(ws.requests, "request", transport)
    output = scrape()
    assert_failed(output)
    assert "Original failure" in output["error"]["message"]
    assert chunks == [1]
    assert slow.closed


@responses.activate
def test_diagnostic_download_timeout_shrinks_with_remaining_budget(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ws.time, "monotonic", lambda: now[0])
    failed_task()

    def get_address(request):
        now[0] = 14.0
        return 200, {}, json.dumps({"code": 200, "data": {"tasks_ids": TASK, "download": DOWNLOAD}})

    responses.add_callback(responses.POST, ADDRESS_URL, callback=get_address)
    responses.add(responses.GET, DOWNLOAD, json=[{"error": "Not collected", "error_code": 300}])
    request = ws.requests.request

    def transport(method, url, **kwargs):
        if url == DOWNLOAD:
            assert 0 < kwargs["timeout"] <= 1
        return request(method, url, **kwargs)

    monkeypatch.setattr(ws.requests, "request", transport)
    output = scrape()
    assert_failed(output)
    assert output["error"]["service_code"] == 300


@responses.activate
def test_blocked_diagnostic_transport_returns_failure_at_deadline(monkeypatch):
    failed_task({"error": "Original failure", "error_code": 300})
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    request = ws.requests.request
    paths = []

    class LateResponse:
        status_code = 200
        headers = {}

        def iter_content(self, chunk_size):
            yield json.dumps({"code": 200, "data": {"tasks_ids": TASK, "download": DOWNLOAD}}).encode()

        def close(self):
            closed.set()

    def transport(method, url, **kwargs):
        paths.append(url)
        if url == ADDRESS_URL:
            started.set()
            release.wait(4)
            return LateResponse()
        return request(method, url, **kwargs)

    monkeypatch.setattr(ws.requests, "request", transport)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(scrape, max_wait_seconds=1)
        try:
            assert started.wait(1)
            output = future.result(timeout=1.5)
            assert_failed(output)
            assert "Original failure" in output["error"]["message"]
        finally:
            release.set()
            future.result(timeout=2)
            assert closed.wait(2)
    assert DOWNLOAD not in paths


@responses.activate
def test_diagnostics_skip_when_all_workers_are_busy(monkeypatch):
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(ws, "_DIAGNOSTIC_SLOTS", slots, raising=False)
    failed_task({"error": "Original failure", "error_code": 300})
    try:
        output = scrape()
    finally:
        slots.release()
    assert_failed(output)
    assert output["error"]["service_code"] == 300
    assert len(responses.calls) == 2
