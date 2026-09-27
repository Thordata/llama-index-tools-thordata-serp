from copy import deepcopy
import json
import sys

import pytest

from llama_index.tools.thordata_serp.response import (
    _MAX_ERROR_MESSAGE_CHARS,
    ResponseError,
    SerpApiError,
    compact_serp_response,
    decode_serp_body,
    redact_serp_response,
)


def test_decodes_success_envelope_data_json_object():
    body = json.dumps({"code": 0, "data": json.dumps({"organic": [{"title": "Coffee"}]})}).encode()

    assert decode_serp_body(200, body) == {"organic": [{"title": "Coffee"}]}


def test_non_success_http_status_raises_upstream_error():
    body = json.dumps({"message": "upstream unavailable", "detail": "maintenance"}).encode()

    with pytest.raises(SerpApiError) as raised:
        decode_serp_body(503, body)

    assert raised.value.status_code == 503
    assert "upstream unavailable" in str(raised.value)
    assert raised.value.payload == {"message": "upstream unavailable", "detail": "maintenance"}


def test_business_error_uses_status_and_data_message():
    body = json.dumps({"code": 401, "data": "invalid API key"}).encode()

    with pytest.raises(SerpApiError) as raised:
        decode_serp_body(200, body)

    assert raised.value.status_code == 401
    assert "invalid API key" in str(raised.value)


@pytest.mark.parametrize("key", ["code", "statusCode", "status_code", "error_code"])
@pytest.mark.parametrize(("value", "expected_status"), [(401, 401), ("429", 429), ("broken", 500)])
def test_each_business_error_code_key_raises(key, value, expected_status):
    with pytest.raises(SerpApiError) as raised:
        decode_serp_body(200, json.dumps({key: value, "message": "request denied"}).encode())

    assert raised.value.status_code == expected_status
    assert "request denied" in str(raised.value)


@pytest.mark.parametrize("key", ["code", "statusCode", "status_code", "error_code"])
@pytest.mark.parametrize("value", [None, "", 0, "0", 200, "200"])
def test_success_business_codes_do_not_raise(key, value):
    assert decode_serp_body(200, json.dumps({key: value, "data": {"organic": []}}).encode()) == {
        "organic": []
    }


@pytest.mark.parametrize("key", ["code", "statusCode", "status_code", "error_code"])
@pytest.mark.parametrize("value", [False, True, 0.0, 200.0])
def test_boolean_and_float_business_codes_are_internal_errors(key, value):
    body = json.dumps({key: value, "message": "unsupported business code type"}).encode()

    with pytest.raises(SerpApiError) as raised:
        decode_serp_body(200, body)

    assert raised.value.status_code == 500
    assert "unsupported business code type" in str(raised.value)


def test_absent_business_code_is_successful():
    assert decode_serp_body(200, b'{"data":{"organic":[]}}') == {"organic": []}


def test_unparseable_structured_business_code_uses_internal_error_status():
    body = json.dumps({"code": {"upstream": "failed"}, "message": "request denied"}).encode()

    with pytest.raises(SerpApiError) as raised:
        decode_serp_body(200, body)

    assert raised.value.status_code == 500


def test_failure_business_code_is_not_masked_by_another_success_code():
    body = json.dumps({"code": 0, "statusCode": "401", "message": "invalid API key"}).encode()

    with pytest.raises(SerpApiError) as raised:
        decode_serp_body(200, body)

    assert raised.value.status_code == 401


def test_nonblank_top_level_error_string_raises_without_failure_code():
    with pytest.raises(SerpApiError, match="quota exhausted") as raised:
        decode_serp_body(200, b'{"error":"quota exhausted","data":{"organic":[]}}')

    assert raised.value.status_code == 500


@pytest.mark.parametrize(
    ("field", "prefix"),
    [("error", "quota exhausted: "), ("data", "error upstream: ")],
)
def test_oversized_business_error_message_is_bounded_with_explicit_truncation(field, prefix):
    message = prefix + "x" * 4096

    with pytest.raises(SerpApiError) as raised:
        decode_serp_body(200, json.dumps({field: message}).encode())

    assert raised.value.status_code == 500
    assert str(raised.value).startswith(prefix)
    assert len(str(raised.value)) == _MAX_ERROR_MESSAGE_CHARS
    assert str(raised.value).endswith("...")


@pytest.mark.parametrize("text", ["Error: invalid API key", "FAILED to reach provider"])
def test_error_like_json_string_payload_raises(text):
    with pytest.raises(SerpApiError, match="(?i)error|failed"):
        decode_serp_body(200, json.dumps(text).encode())


def test_ordinary_json_string_payload_remains_data():
    assert decode_serp_body(200, json.dumps("result id: 123").encode()) == "result id: 123"


@pytest.mark.parametrize("body", [b"", b"not-json", b"\xff"])
def test_invalid_or_empty_response_body_raises_response_error(body):
    with pytest.raises(ResponseError, match="invalid JSON"):
        decode_serp_body(200, body)


@pytest.mark.parametrize("body", [b"NaN", b"Infinity", b"-Infinity", b'{"data":"NaN"}'])
def test_non_finite_json_constants_raise_response_error(body):
    with pytest.raises(ResponseError, match="invalid JSON"):
        decode_serp_body(200, body)


@pytest.mark.parametrize(
    "body",
    [b"<html><body>bad gateway</body></html>", b"bad gateway"],
)
def test_non_success_non_json_response_is_safe_upstream_error(body):
    with pytest.raises(SerpApiError) as raised:
        decode_serp_body(502, body)

    assert raised.value.status_code == 502
    assert len(str(raised.value)) <= 512


def test_excessively_nested_json_body_raises_response_error():
    body = b"[" * 3000 + b"0" + b"]" * 3000

    with pytest.raises(ResponseError, match="deep|nesting|invalid"):
        decode_serp_body(200, body)


def test_excessively_nested_json_string_data_raises_response_error():
    nested_json = "[" * 3000 + "0" + "]" * 3000
    body = json.dumps({"data": nested_json}).encode()

    with pytest.raises(ResponseError, match="deep|nesting|invalid"):
        decode_serp_body(200, body)


def test_plain_arrays_are_preserved():
    assert decode_serp_body(200, b'[{"title":"Coffee"}]') == [{"title": "Coffee"}]


def test_unwraps_at_most_two_nested_json_string_layers():
    wrapped_object = json.dumps(json.dumps({"organic": []}))
    too_deep = json.dumps(json.dumps(json.dumps(json.dumps({"organic": []}))))

    assert decode_serp_body(200, wrapped_object.encode()) == {"organic": []}
    assert decode_serp_body(200, too_deep.encode()) == json.dumps({"organic": []})


def test_compact_response_removes_metadata_recursively_without_mutating_input():
    response = {
        "organic": [
            {
                "title": "Coffee",
                "search_metadata": {"id": "top"},
                "nested": [{"request_metadata": {"trace": "x"}, "keep": 1}],
            }
        ],
        "search_parameters": {"q": "coffee"},
        "serpapi_pagination": {"current": 1},
    }
    original = deepcopy(response)

    assert compact_serp_response(response) == {
        "organic": [{"title": "Coffee", "nested": [{"keep": 1}]}]
    }
    assert response == original


def test_redact_response_replaces_secret_recursively_without_mutating_input():
    secret = "token-123"
    response = {
        "url": f"https://example.test/?key={secret}",
        "nested": [{"header": f"Bearer {secret}"}, secret],
    }
    original = deepcopy(response)

    assert redact_serp_response(response, secret) == {
        "url": "https://example.test/?key=[REDACTED]",
        "nested": [{"header": "Bearer [REDACTED]"}, "[REDACTED]"],
    }
    assert response == original


def test_empty_secret_does_not_modify_response_strings():
    response = {"message": "ordinary response", "values": ["keep"]}

    assert redact_serp_response(response, "") == response


def _deep_list():
    root = []
    current = root
    for _ in range(sys.getrecursionlimit() + 10):
        nested = []
        current.append(nested)
        current = nested
    return root


@pytest.mark.parametrize(
    "transform",
    [
        compact_serp_response,
        lambda value: redact_serp_response(value, "token-123"),
    ],
)
def test_excessively_nested_python_containers_raise_response_error(transform):
    with pytest.raises(ResponseError, match="nesting"):
        transform(_deep_list())


@pytest.mark.parametrize("container_type", ["dict", "list"])
@pytest.mark.parametrize(
    "transform",
    [
        compact_serp_response,
        lambda value: redact_serp_response(value, "token-123"),
    ],
)
def test_cyclic_python_containers_raise_response_error(transform, container_type):
    value = {} if container_type == "dict" else []
    if isinstance(value, dict):
        value["self"] = value
    else:
        value.append(value)

    with pytest.raises(ResponseError, match="cycle"):
        transform(value)


@pytest.mark.parametrize("body", [
    b'"\\ud800"', b'{"data":{"\\udfff":"value"}}',
    b'{"data":{"n":1e999}}', b'{"data":"{\\"n\\":1e999}"}',
    b'{"data":"\\"\\\\ud800\\""}',
])
def test_decoder_rejects_json_that_cannot_be_safely_serialized(body):
    with pytest.raises(ResponseError):
        decode_serp_body(200, body)


@pytest.mark.parametrize("value", [{1: "value"}, {"nested": float("inf")}, {"nested": "\ud800"}])
def test_response_transforms_reject_non_json_trees(value):
    with pytest.raises(ResponseError):
        redact_serp_response(value, "token")


def test_redaction_includes_keys_and_resolves_collisions_with_last_value():
    value = {"token": 1, "[REDACTED]": 2, "nested": [{"prefix-token": "token"}]}
    assert redact_serp_response(value, "token") == {
        "[REDACTED]": 2, "nested": [{"prefix-[REDACTED]": "[REDACTED]"}]
    }
    assert value["token"] == 1


@pytest.mark.parametrize("body", [b"NaN", b"Infinity", b"-Infinity", b"1e999", b'"\\ud800"'])
def test_invalid_json_values_in_http_error_are_response_errors(body):
    with pytest.raises(ResponseError):
        decode_serp_body(401, body)
