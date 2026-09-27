"""Fetch, validate, and atomically update the bundled SERP schema snapshot."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable
from uuid import uuid4

SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

DEFAULT_SCHEMA_ENDPOINT = "https://api.thordata.com/serp/playground/schema?lang=en"

DEFAULT_TARGET = (
    Path(__file__).resolve().parents[1]
    / "llama_index"
    / "tools"
    / "thordata_serp"
    / "serp-schema.snapshot.json"
)


def _engine_count(schema: dict[str, Any]) -> int:
    return sum(len(category.get("engines", [])) for category in schema.get("categories", [])) + len(
        schema.get("engines", [])
    )


def update_snapshot(client: Any, target: Path) -> dict[str, Any]:
    """Fetch once, validate the response, then atomically replace ``target``."""
    from llama_index.tools.thordata_serp.schema import validate_schema_payload

    payload = client.fetch()
    schema = validate_schema_payload(payload)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return schema


def main(
    argv: list[str] | None = None,
    client_factory: Callable[[str, int], Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Update the bundled Thordata SERP schema snapshot.")
    parser.add_argument("--endpoint", default=DEFAULT_SCHEMA_ENDPOINT)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        if client_factory is None:
            from llama_index.tools.thordata_serp.client import SchemaHttpClient

            client_factory = SchemaHttpClient
        schema = update_snapshot(client_factory(args.endpoint, args.timeout), args.target)
    except Exception as error:
        print(f"error={type(error).__name__}")
        return 1
    print(f"version={schema['schema_version']} engines={_engine_count(schema)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
