import json
from urllib.parse import quote_plus as standard_quote_plus, urlencode as standard_urlencode

import pytest
import requests
import responses

from llama_index.tools.thordata_serp.client import (
    DEFAULT_SCHEMA_ENDPOINT,
    DEFAULT_SERP_ENDPOINT,
    MAX_REQUEST_SIZE,
    MAX_RESPONSE_SIZE,
    SchemaHttpClient,
    SerpClient,
)
from llama_index.tools.thordata_serp.response import ResponseError


class FakeResponse:
    def __init__(self, status_code=200, chunks=(), headers=None):
        self.status_code = status_code
        self._chunks = tuple(chunks)
        self.headers = headers or {}
        self.closed = False
        self.chunk_size = None

    def iter_content(self, chunk_size):
        self.chunk_size = chunk_size
        yield from self._chunks

    def close(self):
        self.closed = True


@responses.activate
def test_explicit_bearer_auth_overrides_netrc_without_disabling_proxy_environment(tmp_path, monkeypatch):
    credentials = tmp_path / "netrc"
    credentials.write_text("machine example.test login other password wrong\n", encoding="utf-8")
    monkeypatch.setenv("NETRC", str(credentials))
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:8080")
    endpoint = "https://example.test/serp"
    responses.add(responses.POST, endpoint, json={})
    assert requests.Session().trust_env is True
    assert requests.utils.get_environ_proxies(endpoint)["https"] == "http://proxy.example.test:8080"

    SerpClient("test-token", endpoint=endpoint).request({"q": "coffee"})

    headers = responses.calls[0].request.headers
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["Origin"] == "Llamaindex"
    assert headers["api-source"] == "sdk"


@responses.activate
def test_schema_fetches_json_with_accept_header(schema_payload):
    endpoint = "https://example.test/schema"
    responses.add(responses.GET, endpoint, json=schema_payload, status=200)

    payload = SchemaHttpClient(endpoint=endpoint, timeout=17).fetch()

    assert payload == schema_payload
    assert responses.calls[0].request.headers["Accept"] == "application/json"


def test_schema_uses_configured_timeout_and_stream(monkeypatch, schema_payload):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(chunks=[json.dumps(schema_payload).encode("utf-8")])
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return response

    monkeypatch.setattr(client_module.requests, "get", fake_get)

    assert SchemaHttpClient(timeout=17).fetch() == schema_payload
    assert captured == {
        "url": DEFAULT_SCHEMA_ENDPOINT,
        "headers": {"Accept": "application/json"},
        "timeout": 17,
        "stream": True,
        "allow_redirects": False,
    }
    assert response.closed


def test_serp_posts_exact_form_headers_and_request_options(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(status_code=201, chunks=[b'{"ok":true}'])
    captured = {}
    form = {"engine": "google", "q": "coffee", "json": "1"}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return response

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    assert SerpClient(" secret ", timeout=37).request(form) == (201, b'{"ok":true}')
    assert form == {"engine": "google", "q": "coffee", "json": "1"}
    auth = captured.pop("auth")
    assert isinstance(auth, requests.auth.AuthBase)
    prepared = requests.Request("POST", DEFAULT_SERP_ENDPOINT).prepare()
    assert auth(prepared).headers["Authorization"] == "Bearer secret"
    assert captured == {
        "url": DEFAULT_SERP_ENDPOINT,
        "headers": {
            "Authorization": "Bearer secret",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "Llamaindex",
            "platform": "llamaindex",
            "api-source": "sdk",
        },
        "data": "engine=google&q=coffee&json=1",
        "timeout": 37,
        "stream": True,
        "allow_redirects": False,
    }
    assert response.closed


def test_serp_encodes_unicode_form_as_utf8(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(chunks=[b"{}"])
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    SerpClient("key").request({"engine": "google", "q": "咖啡", "json": "1"})

    assert captured["data"] == "engine=google&q=%E5%92%96%E5%95%A1&json=1"


def test_serp_rejects_oversized_encoded_request_before_network(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    def fail_post(*args, **kwargs):
        pytest.fail("oversized requests must not reach the network")

    monkeypatch.setattr(client_module.requests, "post", fail_post)

    with pytest.raises(ValueError, match="SERP request exceeds"):
        SerpClient("key").request({"q": "x" * MAX_REQUEST_SIZE})


def test_serp_rejects_raw_unicode_over_limit_before_quoting_or_network(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    quoted = []

    def spy_quote_plus(value, *args, **kwargs):
        quoted.append(value)
        return value

    def fail_post(*args, **kwargs):
        pytest.fail("oversized requests must not reach the network")

    monkeypatch.setattr(client_module, "quote_plus", spy_quote_plus, raising=False)
    monkeypatch.setattr(client_module.requests, "post", fail_post)
    raw_value = "咖" * (MAX_REQUEST_SIZE // len("咖".encode("utf-8")) + 1)

    with pytest.raises(ValueError, match="SERP request exceeds"):
        SerpClient("key").request({"q": raw_value})

    assert quoted == []


@pytest.mark.parametrize(
    "value_factory",
    [
        lambda: "x" * MAX_REQUEST_SIZE,
        lambda: "咖" * (MAX_REQUEST_SIZE // len("咖".encode("utf-8")) + 1),
    ],
    ids=["ascii", "unicode"],
)
def test_serp_rejects_raw_oversized_components_before_quote_plus(monkeypatch, value_factory):
    from llama_index.tools.thordata_serp import client as client_module

    component = value_factory()
    quoted_ids = []

    def spy_quote_plus(value, *args, **kwargs):
        quoted_ids.append(id(value))
        return standard_quote_plus(value, *args, **kwargs)

    def fail_post(*args, **kwargs):
        pytest.fail("oversized requests must not reach the network")

    monkeypatch.setattr(client_module, "quote_plus", spy_quote_plus, raising=False)
    monkeypatch.setattr(client_module.requests, "post", fail_post)

    with pytest.raises(ValueError, match="SERP request exceeds"):
        SerpClient("key").request({"q": component})

    assert id(component) not in quoted_ids


@pytest.mark.parametrize(
    "value_factory",
    [
        lambda: " " * 8 + "/" * (MAX_REQUEST_SIZE // 3),
        lambda: "咖" * (MAX_REQUEST_SIZE // 9 + 1),
        lambda: b"\x00" * (MAX_REQUEST_SIZE // 3 + 1),
    ],
    ids=["spaces-and-ascii-escapables", "unicode", "bytes"],
)
def test_serp_rejects_percent_expansion_before_quoting_oversized_component(
    monkeypatch, value_factory
):
    from llama_index.tools.thordata_serp import client as client_module

    component = value_factory()
    quoted_ids = []

    def spy_quote_plus(value, *args, **kwargs):
        quoted_ids.append(id(value))
        if value is component:
            pytest.fail("quote_plus must not receive an oversized component")
        return standard_quote_plus(value, *args, **kwargs)

    def fail_post(*args, **kwargs):
        pytest.fail("oversized requests must not reach the network")

    monkeypatch.setattr(client_module, "quote_plus", spy_quote_plus)
    monkeypatch.setattr(client_module.requests, "post", fail_post)

    with pytest.raises(ValueError, match="SERP request exceeds"):
        SerpClient("key").request({"q": component})

    assert id(component) not in quoted_ids


def test_serp_rejects_later_percent_expansion_before_quoting_value(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    escaped_value = "\x00" * 4
    quoted = []

    def spy_quote_plus(value, *args, **kwargs):
        quoted.append(value)
        if value is escaped_value:
            pytest.fail("quote_plus must not receive an oversized later value")
        return standard_quote_plus(value, *args, **kwargs)

    def fail_post(*args, **kwargs):
        pytest.fail("oversized requests must not reach the network")

    monkeypatch.setattr(client_module, "quote_plus", spy_quote_plus)
    monkeypatch.setattr(client_module.requests, "post", fail_post)
    form = {"q": "a" * (MAX_REQUEST_SIZE - 16), "x": escaped_value}

    with pytest.raises(ValueError, match="SERP request exceeds"):
        SerpClient("key").request(form)

    assert quoted == ["q", form["q"]]


def test_serp_accepts_encoded_body_exactly_at_request_limit(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(chunks=[b"{}"])
    captured = {}
    monkeypatch.setattr(
        client_module.requests,
        "post",
        lambda *args, **kwargs: (captured.update(kwargs), response)[1],
    )
    form = {"q": "a" * (MAX_REQUEST_SIZE - 2)}

    SerpClient("key").request(form)

    assert captured["data"] == standard_urlencode(form)
    assert len(captured["data"]) == MAX_REQUEST_SIZE


@pytest.mark.parametrize(
    "form",
    [
        {"engine": "google", "q": "coffee and tea", "json": "1"},
        {"q": "咖啡", "country": "CN"},
        {"q": b"a b/\x00", "page": 1},
    ],
)
def test_serp_preserves_urlencode_output_for_representative_forms(monkeypatch, form):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(chunks=[b"{}"])
    captured = {}
    monkeypatch.setattr(
        client_module.requests,
        "post",
        lambda *args, **kwargs: (captured.update(kwargs), response)[1],
    )

    SerpClient("key").request(form)

    assert captured["data"] == standard_urlencode(form)


@responses.activate
def test_schema_does_not_follow_redirects():
    source = "https://example.test/schema-source"
    target = "https://example.test/schema-target"
    responses.add(responses.GET, source, body=b"redirect", status=307, headers={"Location": target})
    responses.add(responses.GET, target, json={"unexpected": True}, status=200)

    with pytest.raises(ResponseError, match="HTTP 307"):
        SchemaHttpClient(endpoint=source).fetch()

    assert len(responses.calls) == 1


@responses.activate
def test_serp_does_not_follow_redirects():
    source = "https://example.test/serp-source"
    target = "https://example.test/serp-target"
    responses.add(responses.POST, source, body=b"redirect", status=307, headers={"Location": target})
    responses.add(responses.POST, target, body=b"unexpected", status=200)

    assert SerpClient("key", endpoint=source).request({"q": "coffee"}) == (307, b"redirect")

    assert len(responses.calls) == 1


def test_serp_accepts_response_exactly_at_limit(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(chunks=[b"x" * MAX_RESPONSE_SIZE])
    monkeypatch.setattr(client_module.requests, "post", lambda *args, **kwargs: response)

    status, body = SerpClient("key").request({"q": "coffee"})

    assert status == 200
    assert len(body) == MAX_RESPONSE_SIZE
    assert response.chunk_size == 64 * 1024
    assert response.closed


def test_serp_rejects_stream_that_exceeds_response_limit(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(chunks=[b"x" * MAX_RESPONSE_SIZE, b"x"])
    monkeypatch.setattr(client_module.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(ResponseError, match="response.*size"):
        SerpClient("key").request({"q": "coffee"})

    assert response.closed


def test_serp_rejects_oversized_content_length_and_closes_response(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(headers={"Content-Length": str(MAX_RESPONSE_SIZE + 1)})
    monkeypatch.setattr(client_module.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(ResponseError, match="response.*size"):
        SerpClient("key").request({"q": "coffee"})

    assert response.closed


def test_serp_enforces_stream_limit_after_small_content_length(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(
        chunks=[b"x" * MAX_RESPONSE_SIZE, b"x"],
        headers={"Content-Length": str(MAX_RESPONSE_SIZE)},
    )
    monkeypatch.setattr(client_module.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(ResponseError, match="response.*size"):
        SerpClient("key").request({"q": "coffee"})

    assert response.closed


def test_schema_rejects_error_status_and_closes_response(monkeypatch):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(status_code=503)
    monkeypatch.setattr(client_module.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(ResponseError, match="HTTP 503"):
        SchemaHttpClient().fetch()

    assert response.closed


@pytest.mark.parametrize(
    "body",
    [b"\xff", b"not-json", b"[]"],
    ids=["invalid-utf8", "invalid-json", "nonobject-json"],
)
def test_schema_rejects_invalid_response_bodies_and_closes(monkeypatch, body):
    from llama_index.tools.thordata_serp import client as client_module

    response = FakeResponse(chunks=[body])
    monkeypatch.setattr(client_module.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(ResponseError):
        SchemaHttpClient().fetch()

    assert response.closed


@pytest.mark.parametrize(
    ("client_factory", "method", "error"),
    [
        (lambda: SchemaHttpClient(), "get", requests.Timeout("late")),
        (lambda: SerpClient("key"), "post", requests.RequestException("offline")),
    ],
)
def test_transport_exceptions_propagate(monkeypatch, client_factory, method, error):
    from llama_index.tools.thordata_serp import client as client_module

    def raise_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(client_module.requests, method, raise_error)

    with pytest.raises(type(error)) as caught:
        client_factory().fetch() if method == "get" else client_factory().request({"q": "coffee"})

    assert caught.value is error


def test_clients_strip_valid_constructor_values():
    schema = SchemaHttpClient(endpoint=" https://example.test/schema ", timeout=3)
    serp = SerpClient(" secret ", endpoint=" http://example.test/request ", timeout=4)

    assert schema.endpoint == "https://example.test/schema"
    assert schema.timeout == 3
    assert serp.api_key == "secret"
    assert serp.endpoint == "http://example.test/request"
    assert serp.timeout == 4


@pytest.mark.parametrize("endpoint", ["", "   ", "ftp://example.test", "https:///path", "example.test"])
def test_clients_reject_invalid_endpoints(endpoint):
    with pytest.raises(ValueError, match="endpoint"):
        SchemaHttpClient(endpoint=endpoint)
    with pytest.raises(ValueError, match="endpoint"):
        SerpClient("key", endpoint=endpoint)


@pytest.mark.parametrize("timeout", [False, True, 0, -1, 1.5, "1"])
def test_clients_reject_nonpositive_noninteger_or_boolean_timeouts(timeout):
    with pytest.raises(ValueError, match="timeout"):
        SchemaHttpClient(timeout=timeout)
    with pytest.raises(ValueError, match="timeout"):
        SerpClient("key", timeout=timeout)


@pytest.mark.parametrize("api_key", ["", "   ", None, 7])
def test_serp_client_rejects_blank_or_nonstring_api_keys(api_key):
    with pytest.raises(ValueError, match="api key"):
        SerpClient(api_key)
