from copy import deepcopy
import math
import sys

import pytest

from llama_index.tools.thordata_serp.schema import SchemaError, validate_schema_payload
from llama_index.tools.thordata_serp.serializer import ParameterError, serialize_search


@pytest.fixture
def serializer_schema(schema_payload):
    payload = deepcopy(schema_payload)
    fields = payload["data"]["categories"][0]["engines"][0]["groups"][0]["fields"]
    fields.extend(
        [
            {"key": "safe", "type": "boolean"},
            {"key": "tags", "type": "array"},
            {"key": "cr", "type": "tags"},
            {"key": "config", "type": "object"},
            {"key": "dates", "type": "date_range"},
            {"key": "hours", "type": "time_range"},
            {"key": "place", "type": "cascader"},
            {"key": "json", "type": "options"},
            {"key": "engine", "type": "string"},
        ]
    )
    return validate_schema_payload(payload)


def test_serializes_google_fields_by_schema(serializer_schema):
    engine, form = serialize_search(
        serializer_schema,
        " google ",
        "coffee",
        {
            "q": "ignored",
            "num": 0,
            "safe": False,
            "tags": ["a", 2],
            "cr": ["US", "country gb"],
            "config": {"a": 1},
            "dates": {"start": "2026-01-01", "end": "2026-01-31"},
            "hours": [9, 5, 18, 30],
            "place": ["us", "ny", "nyc"],
            "unknown": "ignored",
            "json": "ignored",
            "engine": "ignored",
        },
        2,
    )

    assert engine["key"] == "google"
    assert form == {
        "q": "coffee",
        "num": "0",
        "safe": "false",
        "tags": "a,2",
        "cr": "countryUS|countryGB",
        "config": '{"a":1}',
        "dates_start": "2026-01-01",
        "dates_end": "2026-01-31",
        "hours": "0905,1830",
        "place": "nyc",
        "json": "2",
        "engine": "google",
    }


def test_country_aliases_normalize_to_serp_country_codes(serializer_schema):
    _, form = serialize_search(
        serializer_schema,
        "google",
        None,
        {
            "cr": [
                "country United States",
                "USA",
                "uk",
                "United Kingdom",
                "Canada",
                "COUNTRY australia",
                "China",
                "zz",
            ]
        },
        None,
    )

    assert form["cr"] == "countryUS|countryUS|countryGB|countryGB|countryCA|countryAU|countryCN|countryZZ"


def test_nonblank_top_values_override_nested_control_values(serializer_schema):
    engine, form = serialize_search(
        serializer_schema,
        "google",
        "coffee",
        {"q": "tea", "json": 1, "engine": "other"},
        "json",
    )

    assert engine["key"] == "google"
    assert form["q"] == "coffee"
    assert form["json"] == "json"
    assert form["engine"] == "google"


def test_blank_top_query_preserves_nested_query_field(serializer_schema):
    _, form = serialize_search(
        serializer_schema,
        "",
        "  ",
        {"q": "tea", "unknown_none": None, "unknown_empty": ""},
        None,
    )

    assert form["q"] == "tea"
    assert form["engine"] == "google"
    assert "json" not in form
    assert "unknown_none" not in form
    assert "unknown_empty" not in form


def test_nonblank_query_preserves_surrounding_spaces(serializer_schema):
    _, form = serialize_search(serializer_schema, "google", "  coffee  ", {}, None)

    assert form["q"] == "  coffee  "


@pytest.mark.parametrize("value", ["false", "0", 0, 1])
def test_rejects_invalid_boolean_values(serializer_schema, value):
    with pytest.raises(ParameterError, match="safe.*boolean"):
        serialize_search(serializer_schema, "google", None, {"safe": value}, None)


def test_unknown_engine_raises_schema_error(serializer_schema):
    with pytest.raises(SchemaError, match="unknown engine"):
        serialize_search(serializer_schema, "missing", None, {}, None)


@pytest.mark.parametrize("field", ["departure_id", "arrival_id"])
def test_google_flights_airport_ids_are_stripped_and_uppercased(schema_payload, field):
    payload = deepcopy(schema_payload)
    engine = payload["data"]["categories"][0]["engines"][0]
    engine["key"] = "google_flights"
    engine["query_field"] = "q"
    engine["groups"][0]["fields"].append({"key": field, "type": "string"})
    payload["data"]["default_engine"] = "google_flights"
    schema = validate_schema_payload(payload)

    _, form = serialize_search(schema, "google_flights", None, {field: " lax "}, None)

    assert form[field] == "LAX"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"tags": ["ok", {"bad": 1}]}, "tags.*scalar"),
        ({"place": {"bad": 1}}, "place.*array"),
        ({"dates": {"start": "2026-01-01"}}, "dates.*start.*end"),
        ({"hours": [9, 5, 18]}, "hours.*four"),
        ({"hours": [9, 5, 18, "30.5"]}, "hours.*integer"),
    ],
)
def test_rejects_malformed_complex_values(serializer_schema, params, message):
    with pytest.raises(ParameterError, match=message):
        serialize_search(serializer_schema, "google", None, params, None)


def test_empty_ordinary_array_and_object_fields_are_omitted(serializer_schema):
    _, form = serialize_search(serializer_schema, "google", None, {"tags": [], "config": {}}, None)

    assert "tags" not in form
    assert "config" not in form


def test_time_range_accepts_digit_shaped_values_outside_clock_ranges(serializer_schema):
    _, form = serialize_search(serializer_schema, "google", None, {"hours": [99, 99, 99, 99]}, None)

    assert form["hours"] == "9999,9999"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([100, 0, 1, 2], "0000,0102"),
        ([100, "12345", 1, 2], "0045,0102"),
    ],
)
def test_time_range_keeps_only_each_component_final_two_digits(serializer_schema, value, expected):
    _, form = serialize_search(
        serializer_schema,
        "google",
        None,
        {"hours": value},
        None,
    )

    assert form["hours"] == expected


def _integer_beyond_string_conversion_limit():
    get_limit = getattr(sys, "get_int_max_str_digits", None)
    set_limit = getattr(sys, "set_int_max_str_digits", None)
    if not callable(get_limit) or not callable(set_limit):
        pytest.skip("the interpreter does not support configurable integer string digit limits")
    limit = get_limit()
    if limit <= 0:
        pytest.skip("the interpreter has no integer string digit limit")
    value = 10**limit
    with pytest.raises(ValueError):
        str(value)
    return value


@pytest.mark.parametrize("field", ["num", "tags", "hours"])
def test_huge_integer_conversion_errors_are_parameter_errors(serializer_schema, field):
    value = _integer_beyond_string_conversion_limit()
    params = {
        "num": {"num": value},
        "tags": {"tags": [value]},
        "hours": {"hours": [value, 0, 1, 2]},
    }[field]

    with pytest.raises(ParameterError, match=rf"{field}.*conversion"):
        serialize_search(serializer_schema, "google", None, params, None)


def test_integer_limit_helper_skips_at_runtime_when_capabilities_are_missing(monkeypatch):
    monkeypatch.delattr(sys, "get_int_max_str_digits", raising=False)
    monkeypatch.delattr(sys, "set_int_max_str_digits", raising=False)

    with pytest.raises(pytest.skip.Exception, match="does not support configurable"):
        _integer_beyond_string_conversion_limit()


def test_rejects_recursive_or_non_json_object(serializer_schema):
    recursive = {}
    recursive["self"] = recursive

    with pytest.raises(ParameterError, match="config.*object"):
        serialize_search(serializer_schema, "google", None, {"config": recursive}, None)

    with pytest.raises(ParameterError, match="config.*object"):
        serialize_search(serializer_schema, "google", None, {"config": {"number": math.nan}}, None)


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf])
def test_numbers_reject_boolean_and_non_finite_floats(serializer_schema, value):
    with pytest.raises(ParameterError, match="num.*number"):
        serialize_search(serializer_schema, "google", None, {"num": value}, None)


def test_boolean_false_and_numeric_zero_are_preserved(serializer_schema):
    _, form = serialize_search(serializer_schema, "google", None, {"safe": False, "num": 0}, None)

    assert form["safe"] == "false"
    assert form["num"] == "0"


def test_arrays_accept_scalar_values_and_lowercase_booleans(serializer_schema):
    _, form = serialize_search(
        serializer_schema,
        "google",
        None,
        {"tags": ["text", 2, 1.5, True, False]},
        None,
    )

    assert form["tags"] == "text,2,1.5,true,false"


def test_rejects_unsupported_schema_field_type(serializer_schema):
    serializer_schema["categories"][0]["engines"][0]["groups"][0]["fields"].append(
        {"key": "unsupported", "type": "unknown"}
    )

    with pytest.raises(ParameterError, match="unsupported.*unsupported schema type"):
        serialize_search(serializer_schema, "google", None, {"unsupported": "value"}, None)


def test_public_inputs_are_not_mutated(serializer_schema):
    schema = deepcopy(serializer_schema)
    params = {"config": {"a": [1]}, "place": ["us", "ny", "nyc"]}
    original_schema = deepcopy(schema)
    original_params = deepcopy(params)

    engine, _ = serialize_search(schema, "google", "coffee", params, 2)
    engine["groups"][0]["fields"][0]["key"] = "changed"

    assert schema == original_schema
    assert params == original_params


def test_effective_query_overrides_invalid_nested_value(serializer_schema):
    params = {"q": ["invalid"]}
    _, form = serialize_search(serializer_schema, "google", " coffee ", params, "1")
    assert form == {"q": " coffee ", "json": "1", "engine": "google"}
    assert params == {"q": ["invalid"]}


def test_effective_flights_query_uses_field_normalization(schema_payload):
    engine = schema_payload["data"]["categories"][0]["engines"][0]
    engine["key"] = "google_flights"
    engine["query_field"] = "departure_id"
    engine["groups"][0]["fields"][0]["key"] = "departure_id"
    schema_payload["data"]["default_engine"] = "google_flights"
    schema = validate_schema_payload(schema_payload)
    _, form = serialize_search(schema, "google_flights", " lax ", {}, "1")
    assert form == {"departure_id": "LAX", "json": "1", "engine": "google_flights"}


@pytest.mark.parametrize("value", [None, "", "  ", [], {}])
def test_all_empty_field_values_are_omitted(serializer_schema, value):
    params = {key: value for key in ("q", "safe", "num", "tags", "config", "dates", "hours", "place")}
    _, form = serialize_search(serializer_schema, "google", "", params, "1")
    assert form == {"json": "1", "engine": "google"}
