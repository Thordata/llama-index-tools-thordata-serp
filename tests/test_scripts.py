import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import smoke_serp, update_schema_snapshot


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_ROOT_RUNNER = """
from pathlib import Path
import runpy
import sys

script = Path(sys.argv[1]).resolve()
source_root = script.parents[1]
sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != source_root]
sys.argv = [str(script), "--help"]
runpy.run_path(str(script), run_name="__main__")
"""

SOURCE_ROOT_IMPORT_RUNNER = """
from pathlib import Path
import json
import runpy
import sys
import tempfile

script = Path(sys.argv[1]).resolve()
source_root = script.parents[1]
sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != source_root]
namespace = runpy.run_path(str(script), run_name="isolated_script")

payload = {
    "code": 0,
    "msg": "ok",
    "data": {
        "schema_version": "1",
        "audience": "is_serp_old=0",
        "default_engine": "google",
        "categories": [{
            "key": "google",
            "name": "Google",
            "engines": [{
                "key": "google",
                "name": "Search",
                "query_field": "q",
                "groups": [{"key": "parameters", "name": "Parameters", "fields": [{
                    "key": "q", "type": "string", "required": True, "visible": True
                }]}],
            }],
        }],
    },
}

if script.name == "smoke_serp.py":
    class Tool:
        def list_engines(self):
            return json.dumps({"ok": True, "schema_source": "stub", "default_engine": "google"})

        def get_engine_schema(self, engine):
            return json.dumps({"ok": True, "engine": {"key": engine}})

        def search(self, engine, query, response_mode):
            return json.dumps({"ok": True, "data": []})

    status = namespace["main"](["--api-key", "stub-key"], tool_spec_factory=lambda key: Tool())
else:
    class Client:
        def fetch(self):
            return payload

    with tempfile.TemporaryDirectory() as directory:
        status = namespace["main"](
            ["--target", str(Path(directory) / "snapshot.json")],
            client_factory=lambda endpoint, timeout: Client(),
        )

import llama_index.tools.thordata_serp
assert status == 0
"""

BOOTSTRAP_REMOVAL_RUNNER = """
from pathlib import Path
import sys

script = Path(sys.argv[1]).resolve()
source_root = script.parents[1]
sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != source_root]
bootstrap = "SOURCE_ROOT = str(Path(__file__).resolve().parents[1])\\nif SOURCE_ROOT not in sys.path:\\n    sys.path.insert(0, SOURCE_ROOT)\\n\\n"
source = script.read_text(encoding="utf-8").replace(bootstrap, "")
namespace = {"__file__": str(script), "__name__": "isolated_script"}
exec(compile(source, str(script), "exec"), namespace)
import llama_index.tools.thordata_serp
"""


@pytest.mark.parametrize("script_name", ["smoke_serp.py", "update_schema_snapshot.py"])
def test_maintenance_scripts_show_help_from_isolated_source_tree(script_name):
    script = PROJECT_ROOT / "scripts" / script_name

    completed = subprocess.run(
        [sys.executable, "-I", "-c", SOURCE_ROOT_RUNNER, str(script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"usage: {script_name}" in completed.stdout


@pytest.mark.parametrize("script_name", ["smoke_serp.py", "update_schema_snapshot.py"])
def test_maintenance_scripts_bootstrap_source_imports_without_network(script_name):
    script = PROJECT_ROOT / "scripts" / script_name

    completed = subprocess.run(
        [sys.executable, "-I", "-c", SOURCE_ROOT_IMPORT_RUNNER, str(script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "stub" in completed.stdout or "version=1" in completed.stdout


@pytest.mark.parametrize("script_name", ["smoke_serp.py", "update_schema_snapshot.py"])
def test_maintenance_script_bootstrap_control_fails_without_source_root(script_name):
    script = PROJECT_ROOT / "scripts" / script_name

    completed = subprocess.run(
        [sys.executable, "-I", "-c", BOOTSTRAP_REMOVAL_RUNNER, str(script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ModuleNotFoundError: No module named 'llama_index.tools'" in completed.stderr


class StubSchemaClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def fetch(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


def engine_count(schema):
    return sum(len(category.get("engines", [])) for category in schema.get("categories", [])) + len(
        schema.get("engines", [])
    )


def test_update_snapshot_validates_and_writes_utf8_envelope(tmp_path, schema_payload):
    target = tmp_path / "snapshot.json"

    schema = update_schema_snapshot.update_snapshot(StubSchemaClient(schema_payload), target)

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["data"]["default_engine"] == "google"
    assert schema["default_engine"] == "google"
    assert target.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    "client",
    [
        StubSchemaClient({"code": 401, "msg": "denied"}),
        StubSchemaClient(error=OSError("offline")),
    ],
)
def test_update_snapshot_never_replaces_existing_target_on_failure(tmp_path, client):
    target = tmp_path / "snapshot.json"
    target.write_text("old snapshot", encoding="utf-8")

    with pytest.raises(Exception):
        update_schema_snapshot.update_snapshot(client, target)

    assert target.read_text(encoding="utf-8") == "old snapshot"
    assert not list(tmp_path.glob("snapshot.json.*.tmp"))


def test_update_snapshot_writes_temp_before_atomic_replace(tmp_path, schema_payload, monkeypatch):
    target = tmp_path / "snapshot.json"
    target.write_text("old snapshot", encoding="utf-8")
    observed = []
    original_replace = Path.replace

    def replace(source, destination):
        observed.append((source, destination, source.read_text(encoding="utf-8")))
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)

    update_schema_snapshot.update_snapshot(StubSchemaClient(schema_payload), target)

    assert observed[0][0].suffix == ".tmp"
    assert observed[0][1] == target
    assert json.loads(observed[0][2])["data"]["default_engine"] == "google"


def test_update_snapshot_preserves_existing_target_when_replace_fails(tmp_path, schema_payload, monkeypatch):
    target = tmp_path / "snapshot.json"
    target.write_text("old snapshot", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("disk failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="disk failure"):
        update_schema_snapshot.update_snapshot(StubSchemaClient(schema_payload), target)

    assert target.read_text(encoding="utf-8") == "old snapshot"
    assert not list(tmp_path.glob("snapshot.json.*.tmp"))


def test_update_main_uses_factory_target_override_and_prints_summary(tmp_path, schema_payload, capsys):
    target = tmp_path / "override.json"
    created = []

    def factory(endpoint, timeout):
        created.append((endpoint, timeout))
        return StubSchemaClient(schema_payload)

    result = update_schema_snapshot.main(
        ["--endpoint", "https://example.test/schema", "--timeout", "17", "--target", str(target)],
        client_factory=factory,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert created == [("https://example.test/schema", 17)]
    assert target.exists()
    assert "version=1" in output
    assert f"engines={engine_count(schema_payload['data'])}" in output


def test_update_main_reports_invalid_remote_without_writing_target(tmp_path, capsys):
    target = tmp_path / "snapshot.json"
    target.write_text("old snapshot", encoding="utf-8")

    result = update_schema_snapshot.main(
        ["--target", str(target)],
        client_factory=lambda endpoint, timeout: StubSchemaClient({"code": 401, "msg": "x" * 5000}),
    )

    output = capsys.readouterr().out
    assert result == 1
    assert target.read_text(encoding="utf-8") == "old snapshot"
    assert "SchemaError" in output
    assert len(output) < 200


def test_smoke_main_requires_nonblank_key_without_constructing_tool(monkeypatch, capsys):
    monkeypatch.delenv("THORDATA_SERP_API_KEY", raising=False)

    result = smoke_serp.main([], tool_spec_factory=lambda key: pytest.fail("network tool constructed"))

    assert result == 2
    assert "THORDATA_SERP_API_KEY is required" in capsys.readouterr().out


class StubToolSpec:
    def __init__(self, api_key, calls, fail_at=None):
        self.api_key = api_key
        self.calls = calls
        self.fail_at = fail_at

    def _result(self, stage, value):
        self.calls.append(stage)
        if self.fail_at == stage:
            return json.dumps({"ok": False, "error": {"type": "SchemaError", "message": self.api_key}})
        return json.dumps(value)

    def list_engines(self):
        return self._result("list", {"ok": True, "schema_source": "remote", "default_engine": "google"})

    def get_engine_schema(self, engine):
        return self._result("schema", {"ok": True, "engine": {"key": engine}})

    def search(self, engine, query, response_mode):
        return self._result("search", {"ok": True, "data": {"organic_results": [{"title": "Coffee"}]}})


def test_smoke_main_runs_default_engine_compact_search_without_payload_output(capsys):
    calls = []
    constructed = []

    def factory(api_key):
        constructed.append(api_key)
        return StubToolSpec(api_key, calls)

    result = smoke_serp.main(["--api-key", "test-key"], tool_spec_factory=factory)

    output = capsys.readouterr().out
    assert result == 0
    assert constructed == ["test-key"]
    assert calls == ["list", "schema", "search"]
    assert "ok=True" in output
    assert "schema_source=remote" in output
    assert "default_engine=google" in output
    assert "result_size=1" in output
    assert "test-key" not in output
    assert "Coffee" not in output


@pytest.mark.parametrize("fail_at", ["list", "schema", "search"])
def test_smoke_main_reports_only_error_type_for_each_failed_stage(capsys, fail_at):
    calls = []

    result = smoke_serp.main(
        ["--api-key", "test-key"],
        tool_spec_factory=lambda api_key: StubToolSpec(api_key, calls, fail_at=fail_at),
    )

    output = capsys.readouterr().out
    assert result == 1
    assert output.strip() == "error=SchemaError"
    assert "test-key" not in output


def test_smoke_main_honors_api_key_and_query(capsys):
    calls = []
    details = {}

    class RecordingTool(StubToolSpec):
        def search(self, engine, query, response_mode):
            details.update(engine=engine, query=query, response_mode=response_mode)
            return super().search(engine, query, response_mode)

    result = smoke_serp.main(
        ["--api-key", "override-key", "--query", "tea"],
        tool_spec_factory=lambda api_key: RecordingTool(api_key, calls),
    )

    assert result == 0
    assert details == {"engine": "google", "query": "tea", "response_mode": "compact"}
    assert "override-key" not in capsys.readouterr().out
