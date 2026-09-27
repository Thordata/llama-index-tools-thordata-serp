import inspect
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).parents[1]
README = PROJECT_ROOT / "README.md"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
LICENSE = PROJECT_ROOT / "LICENSE"


def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_documents_the_public_tools_and_safe_setup():
    document = readme()

    for tool_name in ("list_engines", "get_engine_schema", "search"):
        assert f"`{tool_name}`" in document
    assert "python -m pip install llama-index-tools-thordata-serp" in document
    assert 'os.environ["THORDATA_SERP_API_KEY"]' in document
    assert "os.environ.get" not in document
    assert "os.getenv" not in document
    assert "ThordataSerpToolSpec(api_key=" in document
    assert "tools = tool_spec.to_tool_list()" in document
    assert "FunctionAgent" in document
    assert "schema_source" in document
    assert "snapshot" in document
    assert "five-minute cache" in document
    assert "history" not in document.lower()
    assert "statistics" not in document.lower()


def test_readme_examples_match_the_public_api_and_response_contract():
    document = readme()

    assert 'search(query="latest AI search trends", params={"num": 5}, response_mode="compact")' in document
    assert 'response_format="1"' in document
    assert '"1", "2", "3"' in document
    assert "`complete`" in document
    assert "`compact`" in document
    assert '"ok": true, "status": 200, "engine": "google", "data": {}' in document
    assert '"ok": false' in document
    assert '"type": "SerpApiError"' in document
    assert '"status_code": 401' in document
    assert "serp_endpoint=" in document
    assert "schema_endpoint=" in document
    assert "timeout=" in document
    assert "python scripts/update_schema_snapshot.py" in document
    assert "python scripts/smoke_serp.py" in document
    assert "mock HTTP" in document
    assert "Only `list_engines` and `get_engine_schema` include `schema_source`." in document
    assert "`search` returns `ok`, `status`, `engine`, and `data` without `schema_source`." in document
    assert "schema_cache_ttl=300" in document
    assert "cache duration override" in document
    assert 'python -m pip install "llama-index-llms-openai>=0.5.0,<0.7.0"' in document
    assert "The ToolSpec methods and HTTP client are synchronous." in document
    assert "async ToolSpec" not in document
    assert 'api_key=os.environ["OPENAI_API_KEY"]' in document
    assert "`OPENAI_API_KEY` environment variable" in document
    assert "For POSIX shells:" in document
    assert 'python scripts/smoke_serp.py --api-key "$THORDATA_SERP_API_KEY"' in document
    assert "For PowerShell:" in document
    assert 'python scripts/smoke_serp.py --api-key "$env:THORDATA_SERP_API_KEY"' in document
    assert "Windows cmd.exe" in document
    assert "python scripts/smoke_serp.py --api-key %THORDATA_SERP_API_KEY%" in document

    from llama_index.tools.thordata_serp import ThordataSerpToolSpec

    search_signature = inspect.signature(ThordataSerpToolSpec.search)
    assert list(search_signature.parameters) == [
        "self",
        "engine",
        "query",
        "params",
        "response_format",
        "response_mode",
    ]
    assert search_signature.parameters["engine"].default == ""
    assert search_signature.parameters["query"].default == ""
    assert search_signature.parameters["params"].default is None
    assert search_signature.parameters["response_format"].default == "1"
    assert search_signature.parameters["response_mode"].default == "complete"
    assert "api_key: str" in str(inspect.signature(ThordataSerpToolSpec))

    snippets = re.findall(r"```python\n(.*?)```", document, flags=re.DOTALL)
    assert snippets
    for snippet in snippets:
        compile(snippet, "README.md", "exec")


def test_project_metadata_matches_the_documented_public_package():
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    document = readme()

    project = metadata["project"]
    assert project["name"] == "llama-index-tools-thordata-serp"
    assert project["version"] == "0.1.0"
    assert project["authors"] == [{"name": "Thordata"}]
    assert metadata["tool"]["llamahub"]["import_path"] == "llama_index.tools.thordata_serp"
    assert metadata["tool"]["llamahub"]["class_authors"] == {"ThordataSerpToolSpec": "Thordata"}
    assert set(project["dependencies"]) == {
        "llama-index-core>=0.13.0,<0.15",
        "pydantic>=2.0,<3.0",
        "requests>=2.32,<3.0",
    }

    for public_value in (
        project["name"],
        project["version"],
        "llama_index.tools.thordata_serp",
        "ThordataSerpToolSpec",
        "Thordata",
        "llama-index-core>=0.13.0,<0.15",
        "pydantic>=2.0,<3.0",
        "requests>=2.32,<3.0",
    ):
        assert public_value in document
    assert "tomli>=2,<3; python_version < '3.11'" in metadata["project"]["optional-dependencies"]["dev"]


def test_license_is_standard_mit_for_thordata_2026():
    license_text = LICENSE.read_text(encoding="utf-8")

    assert license_text.startswith("MIT License\n\nCopyright (c) 2026 Thordata\n")
    assert "Permission is hereby granted, free of charge" in license_text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in license_text
