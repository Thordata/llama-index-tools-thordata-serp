import json

import pytest
import requests
import responses

from llama_index.tools.thordata_serp import base as tool_spec_module


ThordataSerpToolSpec = tool_spec_module.ThordataSerpToolSpec


SCHEMA_ENDPOINT = "https://example.test/schema"
SERP_ENDPOINT = "https://example.test/serp"
API_KEY = "secret-api-key"


def result(value):
    return json.loads(value)


def make_spec(**kwargs):
    return ThordataSerpToolSpec(
        API_KEY,
        schema_endpoint=SCHEMA_ENDPOINT,
        serp_endpoint=SERP_ENDPOINT,
        **kwargs,
    )


@responses.activate
@pytest.mark.parametrize("branch", ["http-json", "http-text", "code", "error", "data"])
def test_secret_is_redacted_before_error_truncation(schema_payload, branch):
    secret = "boundary-secret-" + "z" * 40
    message = "x" * 490 + secret
    status = 401 if branch.startswith("http") else 200
    body = {
        "http-json": json.dumps({"message": message}),
        "http-text": message,
        "code": json.dumps({"code": 401, "msg": message}),
        "error": json.dumps({"error": message}),
        "data": json.dumps({"data": "error " + message}),
    }[branch]
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload)
    responses.add(responses.POST, SERP_ENDPOINT, body=body, status=status)
    spec = ThordataSerpToolSpec(secret, schema_endpoint=SCHEMA_ENDPOINT, serp_endpoint=SERP_ENDPOINT)

    output = result(spec.search(query="coffee"))

    assert output["ok"] is False
    assert "boundary" not in output["error"]["message"]
    assert len(output["error"]["message"]) <= 512


@responses.activate
def test_list_engines_summarizes_remote_schema_and_then_uses_cache(schema_payload):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    spec = make_spec()

    first = result(spec.list_engines())
    second = result(spec.get_engine_schema("google"))

    assert first == {
        "ok": True,
        "schema_version": "1",
        "audience": "is_serp_old=0",
        "default_engine": "google",
        "schema_source": "remote",
        "engines": [{"key": "google", "name": "Search", "query_field": "q"}],
    }
    assert second == {
        "ok": True,
        "schema_source": "cache",
        "engine": schema_payload["data"]["categories"][0]["engines"][0],
    }
    assert len(responses.calls) == 1


@responses.activate
def test_search_posts_serialized_form_and_compacts_metadata(schema_payload):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    responses.add(
        responses.POST,
        SERP_ENDPOINT,
        json={
            "data": {
                "organic_results": [{"title": "Coffee"}],
                "search_metadata": {"id": "request"},
                "nested": {"request_metadata": {"trace": "drop"}, "keep": True},
            }
        },
        status=200,
    )
    spec = make_spec()

    complete = result(spec.search("google", "coffee", {"num": 5}, "1", "complete"))
    compact = result(spec.search("google", "coffee", {"num": 5}, "1", "compact"))

    assert complete == {
        "ok": True,
        "status": 200,
        "engine": "google",
        "data": {
            "organic_results": [{"title": "Coffee"}],
            "search_metadata": {"id": "request"},
            "nested": {"request_metadata": {"trace": "drop"}, "keep": True},
        },
    }
    assert compact["data"] == {"organic_results": [{"title": "Coffee"}], "nested": {"keep": True}}
    request = responses.calls[1].request
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert request.body == "num=5&q=coffee&json=1&engine=google"
    assert len(responses.calls) == 3


@responses.activate
def test_search_uses_default_engine_and_non_q_query_field(schema_payload):
    engine = schema_payload["data"]["categories"][0]["engines"][0]
    engine["key"] = "custom"
    engine["query_field"] = "term"
    engine["groups"][0]["fields"][0]["key"] = "term"
    schema_payload["data"]["default_engine"] = "custom"
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    responses.add(responses.POST, SERP_ENDPOINT, json={"data": {"ok": True}}, status=200)

    payload = result(make_spec().search(query="coffee"))

    assert payload["ok"] is True
    assert payload["engine"] == "custom"
    assert "term=coffee" in responses.calls[1].request.body
    assert "engine=custom" in responses.calls[1].request.body


@responses.activate
def test_search_rejects_unknown_engine_after_schema_fetch(schema_payload):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    spec = make_spec()

    payload = result(spec.search(engine="unknown", query="coffee"))

    assert payload["ok"] is False
    assert payload["error"]["type"] == "SchemaError"
    assert len(responses.calls) == 1


@responses.activate
@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "coffee", "params": ["wrong"]},
        {"query": "coffee", "params": "wrong"},
        {"query": "coffee", "response_format": "9"},
        {"query": "coffee", "response_mode": "brief"},
    ],
)
def test_search_rejects_schema_independent_invalid_input_without_requests(
    schema_payload, kwargs
):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    responses.add(responses.POST, SERP_ENDPOINT, json={"data": {}}, status=200)

    payload = result(make_spec().search(**kwargs))

    assert payload["ok"] is False
    assert payload["error"]["type"] == "ValueError"
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.parametrize(
    "kwargs",
    [
        {"engine": 42},
        {"engine": ["google"]},
        {"query": 42},
        {"query": {"text": "coffee"}},
    ],
)
def test_search_rejects_non_string_engine_or_query_without_requests(schema_payload, kwargs):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    responses.add(responses.POST, SERP_ENDPOINT, json={"data": {}}, status=200)

    payload = result(make_spec().search(**kwargs))

    assert payload["ok"] is False
    assert payload["error"]["type"] == "ValueError"
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.parametrize("engine", [42, None, "", "   "])
def test_get_engine_schema_rejects_invalid_engine_without_requests(schema_payload, engine):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)

    payload = result(make_spec().get_engine_schema(engine))

    assert payload["ok"] is False
    assert payload["error"]["type"] == "ValueError"
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "x" * (1 << 20)},
        {"params": {"q": "x" * (1 << 20)}},
        {"query": "\ud800"},
    ],
)
def test_search_normalizes_client_form_errors_without_serp_request(schema_payload, kwargs):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)

    output = make_spec().search(**kwargs)
    payload = result(output)

    assert payload["ok"] is False
    assert payload["error"]["type"] == "ValueError"
    assert API_KEY not in output
    assert len(responses.calls) == 1


@responses.activate
@pytest.mark.parametrize("body", [b"NaN", b"Infinity", b"-Infinity", b"1e999", b'"\\ud800"', b'{"data":{"\\udfff":"value"}}'])
def test_search_normalizes_non_finite_upstream_json(schema_payload, body):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    responses.add(responses.POST, SERP_ENDPOINT, body=body, status=200)

    payload = result(make_spec().search(query="coffee"))

    assert payload["ok"] is False
    assert payload["error"]["type"] == "ResponseError"


@responses.activate
@pytest.mark.parametrize("secret", ["true", "200", 'quote"slash\\', "ok", "data"])
@pytest.mark.parametrize("status", [200, 401])
def test_structured_redaction_preserves_json_syntax_and_public_contract(schema_payload, secret, status):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload)
    body = {"data": {"nested": [{secret: secret}], "boolean": True, "number": 200}}
    if status != 200:
        body = {"message": "failed " + secret}
    responses.add(responses.POST, SERP_ENDPOINT, json=body, status=status)
    spec = ThordataSerpToolSpec(secret, schema_endpoint=SCHEMA_ENDPOINT, serp_endpoint=SERP_ENDPOINT)

    raw = spec.search(query="coffee")
    raw.encode("utf-8")
    output = result(raw)

    assert output["ok"] is (status == 200)
    if status == 200:
        assert output["status"] == 200
        assert output["data"] == {"nested": [{"[REDACTED]": "[REDACTED]"}], "boolean": True, "number": 200}
    else:
        assert output["error"]["status_code"] == 401
        assert output["error"]["message"] == "failed [REDACTED]"


@responses.activate
@pytest.mark.parametrize(
    "body",
    [
        b"<html><body>" + API_KEY.encode() + b"</body></html>",
        b"bad gateway " + API_KEY.encode() + b" " + b"x" * 600,
    ],
)
def test_search_normalizes_non_json_upstream_error_without_leaking_key(schema_payload, body):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    responses.add(responses.POST, SERP_ENDPOINT, body=body, status=502)

    output = make_spec().search(query="coffee")
    payload = result(output)

    assert payload["ok"] is False
    assert payload["error"]["type"] == "SerpApiError"
    assert payload["error"]["status_code"] == 502
    assert API_KEY not in output
    assert len(payload["error"]["message"]) <= 512


@responses.activate
@pytest.mark.parametrize(
    ("status", "body", "error_type", "error_status"),
    [
        (503, {"message": "provider unavailable"}, "SerpApiError", 503),
        (200, {"code": 401, "msg": "bad key"}, "SerpApiError", 401),
        (200, b"not-json", "ResponseError", None),
    ],
)
def test_search_normalizes_upstream_and_response_errors(
    schema_payload, status, body, error_type, error_status
):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    if isinstance(body, dict):
        responses.add(responses.POST, SERP_ENDPOINT, json=body, status=status)
    else:
        responses.add(responses.POST, SERP_ENDPOINT, body=body, status=status)

    payload = result(make_spec().search(query="coffee"))

    assert payload["ok"] is False
    assert payload["error"]["type"] == error_type
    if error_status is not None:
        assert payload["error"]["status_code"] == error_status


@responses.activate
@pytest.mark.parametrize("error", [requests.Timeout("late"), requests.RequestException("offline")])
def test_public_methods_normalize_transport_errors(schema_payload, error):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    responses.add(responses.POST, SERP_ENDPOINT, body=error)

    payload = result(make_spec().search(query="coffee"))

    assert payload == {
        "ok": False,
        "error": {"type": "ResponseError", "message": "SERP request failed: " + str(error)},
    }


@responses.activate
def test_all_public_output_redacts_api_key_in_data_and_errors(schema_payload):
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    responses.add(
        responses.POST,
        SERP_ENDPOINT,
        json={"data": {"key": API_KEY, "nested": [f"value {API_KEY}"]}},
        status=200,
    )
    success = make_spec().search(query="coffee")

    assert API_KEY not in success
    responses.reset()
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload, status=200)
    responses.add(responses.POST, SERP_ENDPOINT, json={"message": API_KEY}, status=500)
    failure = make_spec().search(query="coffee")

    assert API_KEY not in failure
    assert "[REDACTED]" in failure


@responses.activate
def test_list_engines_uses_bundled_snapshot_after_cold_schema_failure():
    responses.add(responses.GET, SCHEMA_ENDPOINT, body=requests.Timeout("schema offline"))

    payload = result(make_spec().list_engines())

    assert payload["ok"] is True
    assert payload["schema_source"] == "snapshot"
    assert payload["engines"]


def test_input_schemas_and_tool_metadata_are_agent_usable():
    assert tool_spec_module.ListEnginesInput.model_fields == {}
    assert tool_spec_module.GetEngineSchemaInput.model_fields["engine"].is_required()
    fields = tool_spec_module.SearchInput.model_fields
    assert fields["params"].annotation == dict | None
    assert set(fields["response_format"].annotation.__args__) == {"1", "2", "3"}
    assert set(fields["response_mode"].annotation.__args__) == {"complete", "compact"}

    tools = make_spec().to_tool_list()

    assert [tool.metadata.name for tool in tools] == ["list_engines", "get_engine_schema", "search"]
    assert all(tool.metadata.description for tool in tools)
    assert [tool.metadata.fn_schema for tool in tools] == [
        tool_spec_module.ListEnginesInput,
        tool_spec_module.GetEngineSchemaInput,
        tool_spec_module.SearchInput,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": ""},
        {"api_key": None},
        {"api_key": API_KEY, "serp_endpoint": "not a URL"},
        {"api_key": API_KEY, "schema_endpoint": "ftp://example.test/schema"},
        {"api_key": API_KEY, "timeout": 0},
        {"api_key": API_KEY, "schema_timeout": False},
        {"api_key": API_KEY, "schema_cache_ttl": 0},
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ThordataSerpToolSpec(**kwargs)


@responses.activate
def test_malformed_remote_field_schema_falls_back_before_caching(schema_payload):
    schema_payload["data"]["categories"][0]["engines"][0]["groups"][0]["fields"][0]["type"] = []
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload)
    responses.add(responses.POST, SERP_ENDPOINT, json={"data": {}})
    spec = make_spec()
    assert result(spec.list_engines())["schema_source"] == "snapshot"
    assert result(spec.search(query="coffee"))["ok"] is True
    assert result(spec.list_engines())["schema_source"] == "cache"


@responses.activate
@pytest.mark.parametrize("kind", ["unknown-deep", "object-deep", "object-cycle"])
def test_public_search_handles_nested_parameters_safely(schema_payload, kind):
    fields = schema_payload["data"]["categories"][0]["engines"][0]["groups"][0]["fields"]
    fields.append({"key": "config", "type": "object"})
    nested = []
    for _ in range(700):
        nested = [nested]
    value = {"nested": nested}
    if kind == "object-cycle":
        value = {}
        value["self"] = value
    params = {"unknown" if kind == "unknown-deep" else "config": value}
    responses.add(responses.GET, SCHEMA_ENDPOINT, json=schema_payload)
    responses.add(responses.POST, SERP_ENDPOINT, json={"data": {}})

    output = result(make_spec().search(query="coffee", params=params))

    if kind == "unknown-deep":
        assert output["ok"] is True
        assert responses.calls[1].request.body == "q=coffee&json=1&engine=google"
    else:
        assert output["ok"] is False
        assert output["error"]["type"] == "ValueError"
        assert "JSON object" in output["error"]["message"]
        assert len(responses.calls) == 1
    assert next(iter(params.values())) is value
