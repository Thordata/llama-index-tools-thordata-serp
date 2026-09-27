"""Run a small opt-in Thordata SERP smoke request."""

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any, Callable

SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

def _result(payload: str) -> dict[str, Any]:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("tool response must be an object")
    if not decoded.get("ok"):
        error = decoded.get("error")
        if isinstance(error, dict) and isinstance(error.get("type"), str):
            raise RuntimeError(error["type"])
        raise RuntimeError("ToolError")
    return decoded


def _result_size(data: Any) -> int:
    if isinstance(data, (Mapping, list, tuple, str, bytes)):
        return len(data)
    return 0


def main(
    argv: list[str] | None = None,
    tool_spec_factory: Callable[[str], Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run an opt-in Thordata SERP smoke request.")
    parser.add_argument("--api-key", default=os.environ.get("THORDATA_SERP_API_KEY", ""))
    parser.add_argument("--query", default="coffee")
    args = parser.parse_args(argv)
    api_key = args.api_key.strip() if isinstance(args.api_key, str) else ""
    if not api_key:
        print("THORDATA_SERP_API_KEY is required")
        return 2
    try:
        if tool_spec_factory is None:
            from llama_index.tools.thordata_serp import ThordataSerpToolSpec

            tool_spec_factory = ThordataSerpToolSpec
        tool = tool_spec_factory(api_key)
        listed = _result(tool.list_engines())
        engine = listed["default_engine"]
        if not isinstance(engine, str) or not engine:
            raise ValueError("default engine missing")
        _result(tool.get_engine_schema(engine))
        searched = _result(tool.search(engine, args.query, response_mode="compact"))
        source = listed.get("schema_source")
        if not isinstance(source, str):
            raise ValueError("schema source missing")
        print(
            f"ok=True schema_source={source} default_engine={engine} "
            f"result_size={_result_size(searched.get('data'))}"
        )
        return 0
    except RuntimeError as error:
        print(f"error={error}")
        return 1
    except Exception as error:
        print(f"error={type(error).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
