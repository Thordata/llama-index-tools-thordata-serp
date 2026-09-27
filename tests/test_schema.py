from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from llama_index.tools.thordata_serp.schema import (
    SchemaError,
    SchemaRepository,
    find_engine,
    load_snapshot,
    validate_schema_payload,
)


class StubSchemaClient:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0
        self._lock = threading.Lock()

    def fetch(self):
        with self._lock:
            self.calls += 1
            error = self.error
            payload = deepcopy(self.payload)
        if error is not None:
            raise error
        return payload


class SlowSchemaClient(StubSchemaClient):
    def __init__(self, payload, error=None):
        super().__init__(payload, error)
        self.started = threading.Event()
        self.release = threading.Event()

    def fetch(self):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release schema fetch")
        return super().fetch()


class RecoverySchemaClient(StubSchemaClient):
    def __init__(self, payload):
        super().__init__(payload)
        self.first_started = threading.Event()
        self.first_release = threading.Event()

    def fetch(self):
        with self._lock:
            self.calls += 1
            call = self.calls
            payload = deepcopy(self.payload)
        if call == 1:
            self.first_started.set()
            if not self.first_release.wait(timeout=5):
                raise TimeoutError("test did not release first schema fetch")
            raise OSError("generation one failed")
        return payload


class TrackingCondition(threading.Condition):
    def __init__(self, expected_waiters, pause_after_wake=False):
        super().__init__()
        self.expected_waiters = expected_waiters
        self.pause_after_wake = pause_after_wake
        self.wait_calls = 0
        self.paused_calls = 0
        self.all_waiting = threading.Event()
        self.all_paused = threading.Event()
        self.resume = threading.Event()
        self._wait_calls_lock = threading.Lock()

    def wait(self, timeout=None):
        with self._wait_calls_lock:
            self.wait_calls += 1
            if self.wait_calls == self.expected_waiters:
                self.all_waiting.set()
        result = super().wait(timeout)
        if self.pause_after_wake:
            self.release()
            try:
                with self._wait_calls_lock:
                    self.paused_calls += 1
                    if self.paused_calls == self.expected_waiters:
                        self.all_paused.set()
                if not self.resume.wait(timeout=5):
                    raise TimeoutError("test did not resume schema waiters")
            finally:
                self.acquire()
        return result


def write_snapshot(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def tracked_repository(client, snapshot_path, pause_after_wake=False, **kwargs):
    repository = SchemaRepository(client, snapshot_path, **kwargs)
    condition = TrackingCondition(expected_waiters=7, pause_after_wake=pause_after_wake)
    repository._condition = condition
    return repository, condition


def test_repository_caches_valid_remote_schema_within_ttl(tmp_path, schema_payload):
    clock = [100.0]
    client = StubSchemaClient(schema_payload)
    repository = SchemaRepository(client, tmp_path / "schema.json", ttl=60, clock=lambda: clock[0])

    remote_schema, remote_source = repository.current()
    clock[0] += 10
    cached_schema, cached_source = repository.current()

    assert remote_source == "remote"
    assert cached_source == "cache"
    assert cached_schema == remote_schema
    assert client.calls == 1


def test_repository_returns_stale_cache_when_expired_remote_refresh_fails(tmp_path, schema_payload):
    clock = [100.0]
    client = StubSchemaClient(schema_payload)
    repository = SchemaRepository(client, tmp_path / "schema.json", ttl=60, clock=lambda: clock[0])

    fresh_schema, _ = repository.current()
    clock[0] += 60
    client.error = OSError("upstream unavailable")
    stale_schema, source = repository.current()

    assert source == "cache"
    assert stale_schema == fresh_schema
    assert client.calls == 2


def test_repository_uses_snapshot_when_cold_remote_fetch_fails(tmp_path, schema_payload):
    snapshot_path = tmp_path / "schema.json"
    write_snapshot(snapshot_path, schema_payload)
    repository = SchemaRepository(
        StubSchemaClient(schema_payload, error=OSError("upstream unavailable")),
        snapshot_path,
    )

    schema, source = repository.current()

    assert source == "snapshot"
    assert schema["default_engine"] == "google"


def test_repository_shares_one_cold_refresh_across_overlapping_callers(tmp_path, schema_payload):
    client = SlowSchemaClient(schema_payload)
    repository, condition = tracked_repository(client, tmp_path / "schema.json")

    with ThreadPoolExecutor(max_workers=8) as executor:
        leader = executor.submit(repository.current)
        assert client.started.wait(timeout=5)
        followers = [executor.submit(repository.current) for _ in range(7)]
        assert condition.all_waiting.wait(timeout=5)
        client.release.set()
        results = [leader.result(timeout=5), *(future.result(timeout=5) for future in followers)]

    assert client.calls == 1
    assert all(source == "remote" for _, source in results)
    assert all(schema["default_engine"] == "google" for schema, _ in results)
    _, source = repository.current()
    assert source == "cache"


def test_repository_shares_cold_failure_fallback_across_overlapping_callers(tmp_path, schema_payload):
    snapshot_path = tmp_path / "schema.json"
    write_snapshot(snapshot_path, schema_payload)
    client = SlowSchemaClient(schema_payload, error=OSError("upstream unavailable"))
    repository, condition = tracked_repository(client, snapshot_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        leader = executor.submit(repository.current)
        assert client.started.wait(timeout=5)
        followers = [executor.submit(repository.current) for _ in range(7)]
        assert condition.all_waiting.wait(timeout=5)
        client.release.set()
        results = [leader.result(timeout=5), *(future.result(timeout=5) for future in followers)]

    assert client.calls == 1
    assert all(source == "snapshot" for _, source in results)
    assert all(schema["default_engine"] == "google" for schema, _ in results)
    _, source = repository.current()
    assert source == "cache"


def test_repository_shares_cold_remote_and_snapshot_failure_across_overlapping_callers(
    tmp_path, schema_payload
):
    snapshot_path = tmp_path / "schema.json"
    snapshot_path.write_text("{", encoding="utf-8")
    client = SlowSchemaClient(schema_payload, error=OSError("upstream unavailable"))
    repository, condition = tracked_repository(client, snapshot_path)

    def current_error():
        with pytest.raises(SchemaError) as caught:
            repository.current()
        return str(caught.value)

    with ThreadPoolExecutor(max_workers=8) as executor:
        leader = executor.submit(current_error)
        assert client.started.wait(timeout=5)
        followers = [executor.submit(current_error) for _ in range(7)]
        assert condition.all_waiting.wait(timeout=5)
        client.release.set()
        errors = [leader.result(timeout=5), *(future.result(timeout=5) for future in followers)]

    assert client.calls == 1
    assert len(set(errors)) == 1
    assert "upstream unavailable" in errors[0]
    assert "cannot load schema snapshot" in errors[0]


def test_repository_retries_after_cold_remote_and_snapshot_failure(tmp_path, schema_payload):
    snapshot_path = tmp_path / "schema.json"
    snapshot_path.write_text("{", encoding="utf-8")
    client = StubSchemaClient(schema_payload, error=OSError("upstream unavailable"))
    repository = SchemaRepository(client, snapshot_path)

    with pytest.raises(SchemaError, match="upstream unavailable.*cannot load schema snapshot"):
        repository.current()

    client.error = None
    schema, source = repository.current()

    assert client.calls == 2
    assert source == "remote"
    assert schema["default_engine"] == "google"


def test_repository_keeps_waiters_bound_to_their_refresh_generation(tmp_path, schema_payload):
    snapshot_path = tmp_path / "schema.json"
    snapshot_path.write_text("{", encoding="utf-8")
    client = RecoverySchemaClient(schema_payload)
    repository, condition = tracked_repository(
        client,
        snapshot_path,
        pause_after_wake=True,
    )

    def current_error():
        with pytest.raises(SchemaError) as caught:
            repository.current()
        return str(caught.value)

    with ThreadPoolExecutor(max_workers=9) as executor:
        leader = executor.submit(current_error)
        assert client.first_started.wait(timeout=5)
        followers = [executor.submit(current_error) for _ in range(7)]
        assert condition.all_waiting.wait(timeout=5)
        client.first_release.set()
        assert condition.all_paused.wait(timeout=5)

        recovery_schema, recovery_source = executor.submit(repository.current).result(timeout=5)
        condition.resume.set()
        leader_error = leader.result(timeout=5)
        follower_errors = [future.result(timeout=5) for future in followers]

    assert client.calls == 2
    assert recovery_source == "remote"
    assert recovery_schema["default_engine"] == "google"
    assert "generation one failed" in leader_error
    assert len(set(follower_errors)) == 1
    assert "generation one failed" in follower_errors[0]
    assert "cannot load schema snapshot" in follower_errors[0]
    _, source = repository.current()
    assert source == "cache"


def test_repository_recovers_from_warm_cache_clock_failure(tmp_path, schema_payload):
    clock = [100.0]
    client = StubSchemaClient(schema_payload)

    def read_clock():
        if isinstance(clock[0], Exception):
            raise clock[0]
        return clock[0]

    repository = SchemaRepository(client, tmp_path / "schema.json", clock=read_clock)

    repository.current()
    clock[0] = RuntimeError("cache clock unavailable")
    with pytest.raises(SchemaError, match="cache clock.*unavailable"):
        repository.current()

    clock[0] = 100.0
    schema, source = repository.current()

    assert source == "cache"
    assert schema["default_engine"] == "google"
    assert client.calls == 1


def test_repository_clears_refresh_when_cache_clock_raises(tmp_path, schema_payload):
    def broken_clock():
        raise RuntimeError("clock unavailable")

    client = SlowSchemaClient(schema_payload)
    repository, condition = tracked_repository(
        client,
        tmp_path / "schema.json",
        clock=broken_clock,
    )

    def current_error():
        with pytest.raises(SchemaError) as caught:
            repository.current()
        return str(caught.value)

    with ThreadPoolExecutor(max_workers=8) as executor:
        leader = executor.submit(current_error)
        assert client.started.wait(timeout=5)
        followers = [executor.submit(current_error) for _ in range(7)]
        assert condition.all_waiting.wait(timeout=5)
        client.release.set()
        errors = [leader.result(timeout=5), *(future.result(timeout=5) for future in followers)]

    assert client.calls == 1
    assert all("clock unavailable" in error for error in errors)
    with ThreadPoolExecutor(max_workers=1) as executor:
        later = executor.submit(current_error)
        assert "clock unavailable" in later.result(timeout=5)


def test_repository_returns_deep_copy_isolated_schemas(tmp_path, schema_payload):
    repository = SchemaRepository(StubSchemaClient(schema_payload), tmp_path / "schema.json")

    first_schema, _ = repository.current()
    first_schema["categories"][0]["engines"][0]["groups"][0]["fields"][0]["key"] = "changed"
    second_schema, _ = repository.current()

    assert second_schema["categories"][0]["engines"][0]["groups"][0]["fields"][0]["key"] == "q"


@pytest.mark.parametrize("ttl", [0, -1, False, True])
def test_repository_rejects_non_positive_or_boolean_ttl(tmp_path, schema_payload, ttl):
    with pytest.raises(ValueError, match="ttl"):
        SchemaRepository(StubSchemaClient(schema_payload), tmp_path / "schema.json", ttl=ttl)


def test_valid_envelope_unwraps_and_finds_default_google_engine(schema_payload):
    schema = validate_schema_payload(schema_payload)

    assert schema["default_engine"] == "google"
    assert find_engine(schema, "google")["query_field"] == "q"


def test_direct_engines_work_without_categories(schema_payload):
    schema_payload["data"]["engines"] = schema_payload["data"].pop("categories")[0]["engines"]

    schema = validate_schema_payload(schema_payload)

    assert find_engine(schema, "google")["query_field"] == "q"


def test_query_field_must_reference_a_declared_field(schema_payload):
    schema_payload["data"]["categories"][0]["engines"][0]["query_field"] = "missing"

    with pytest.raises(SchemaError, match="query_field.*declared field"):
        validate_schema_payload(schema_payload)


def test_direct_schema_document_is_accepted(schema_payload):
    schema = validate_schema_payload(deepcopy(schema_payload["data"]))

    assert schema["default_engine"] == "google"


def test_validate_schema_payload_returns_an_independent_nested_copy(schema_payload):
    schema = validate_schema_payload(schema_payload)
    schema["categories"][0]["engines"][0]["groups"][0]["fields"][0]["key"] = "changed"

    assert schema_payload["data"]["categories"][0]["engines"][0]["groups"][0]["fields"][0]["key"] == "q"


def test_find_engine_returns_an_independent_nested_copy(schema_payload):
    schema = validate_schema_payload(schema_payload)
    engine = find_engine(schema, "google")
    engine["groups"][0]["fields"][0]["key"] = "changed"

    assert schema["categories"][0]["engines"][0]["groups"][0]["fields"][0]["key"] == "q"


def test_business_failure_includes_upstream_code_and_message():
    with pytest.raises(SchemaError, match="401.*denied"):
        validate_schema_payload({"code": 401, "msg": "denied"})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["data"].__setitem__("audience", "is_serp_old=1"),
            "unsupported audience",
        ),
        (
            lambda payload: payload["data"].__setitem__("default_engine", "missing"),
            "default engine",
        ),
        (
            lambda payload: payload["data"].__setitem__("schema_version", ""),
            "schema_version",
        ),
    ],
)
def test_invalid_schema_payload_is_rejected(schema_payload, mutate, message):
    mutate(schema_payload)

    with pytest.raises(SchemaError, match=message):
        validate_schema_payload(schema_payload)


def test_package_snapshot_loads_and_declares_its_default_engine():
    snapshot_path = Path(__file__).parents[1] / "llama_index/tools/thordata_serp/serp-schema.snapshot.json"

    schema = load_snapshot(snapshot_path)

    assert schema["audience"] == "is_serp_old=0"
    assert find_engine(schema, schema["default_engine"])["key"] == schema["default_engine"]


def test_load_snapshot_rejects_invalid_utf8(tmp_path):
    snapshot_path = tmp_path / "invalid.json"
    snapshot_path.write_bytes(b"\xff")

    with pytest.raises(SchemaError, match="cannot load schema snapshot"):
        load_snapshot(snapshot_path)
@pytest.mark.parametrize(
    "attribute,value",
    [("type", []), ("type", None), ("control", {}), ("visible", "true"),
     ("required", 1), ("options", {}), ("options", ["bad"]),
     ("options", [{"label": [], "value": "x"}]),
     ("options", [{"label": "x", "value": {}}]),
     ("default", ["bad"]), ("default_value", {"bad": 1}),
     ("name", []), ("group", {}),
     ("show_when", {"field": "num", "operator": []}),
     ("show_when", {"field": "num", "values": {}})],
)
def test_rejects_malformed_field_structure(schema_payload, attribute, value):
    field = schema_payload["data"]["categories"][0]["engines"][0]["groups"][0]["fields"][0]
    field[attribute] = value
    with pytest.raises(SchemaError):
        validate_schema_payload(schema_payload)
