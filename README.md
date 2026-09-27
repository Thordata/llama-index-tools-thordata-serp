## LlamaIndex integration for Thordata

### Hero

[**Thordata**](https://www.thordata.com/?ls=github01&lk=github01) gives your LlamaIndex Agent **real-time, structured web search** through a single schema-driven tool.

`llama-index-tools-thordata-serp` version `0.1.0` plugs Thordata's SERP API into your LlamaIndex workflow. The public import path is `llama_index.tools.thordata_serp`. `ThordataSerpToolSpec` exposes live search engines through predictable, agent-ready JSON, so a `FunctionAgent` can search the web, news, images, maps, shopping, flights, scholar, and local results through one interface.

**Why teams pick it:**

- **Schema-driven, no guessing** — every search parameter is validated against the selected engine's live schema; unsupported fields are ignored before the request is sent.
- **34 engines in the bundled snapshot** — web, news, images, maps, shopping, flights, scholar, and local engines are unified behind one `search` tool, while remote schema updates can add new engines without a package release.
- **Output built for agents** — a predictable `ok` / `status` / `engine` / `data` envelope; `compact` mode strips verbose metadata to reduce context usage.
- **Robust and reliable** — bounded requests and responses, normalized errors, API-key redaction, and a built-in schema snapshot fallback.
- **Works with existing Thordata access** — pass your SERP API key explicitly and keep the package configuration under your control.

### Install

Python `>=3.10` is required.

```bash
python -m pip install llama-index-tools-thordata-serp
```

The runtime dependencies are `llama-index-core>=0.13.0,<0.15`, `pydantic>=2.0,<3.0`, and `requests>=2.32,<3.0`.

Create a Thordata SERP API key in the [Thordata dashboard](https://www.thordata.com) and set it in the calling environment. The tool spec does not read environment variables itself; read and pass the key explicitly:

```python
import os

from llama_index.tools.thordata_serp import ThordataSerpToolSpec

tool_spec = ThordataSerpToolSpec(api_key=os.environ["THORDATA_SERP_API_KEY"])
tools = tool_spec.to_tool_list()
```

### Direct use

Each tool returns a JSON string. Decode it with `json.loads` before consuming its fields.

```python
import json
import os

from llama_index.tools.thordata_serp import ThordataSerpToolSpec

tool_spec = ThordataSerpToolSpec(api_key=os.environ["THORDATA_SERP_API_KEY"])
engines = json.loads(tool_spec.list_engines())
default_engine = engines["default_engine"]
google_schema = json.loads(tool_spec.get_engine_schema("google"))
result = json.loads(
    tool_spec.search(query="latest AI search trends", params={"num": 5}, response_mode="compact")
)
```

#### `list_engines`

`list_engines()` reports the current schema version, default engine, schema source, and concise engine records. Use its `default_engine` when the caller needs to select an engine explicitly.

#### `get_engine_schema`

`get_engine_schema(engine)` returns the selected engine's complete groups and fields. Search parameters are schema-scoped: only fields defined by that engine are serialized. Empty strings, empty lists, empty objects, and `None` parameters are omitted; numeric `0` and boolean `false` are preserved.

#### `search`

`search(engine="", query="", params=None, response_format="1", response_mode="complete")` submits the selected engine's query and accepted schema fields. An empty `engine` uses the schema default. `response_format` accepts `"1", "2", "3"` and selects the Thordata response format. `response_mode` is `complete` for the returned data or `compact` to remove verbose request metadata.

Only `list_engines` and `get_engine_schema` include `schema_source`. `search` returns `ok`, `status`, `engine`, and `data` without `schema_source`.

Successful responses have this shape:

```json
{"ok": true, "status": 200, "engine": "google", "data": {}}
```

Failures are normalized into this shape. The status code is present for upstream API errors:

```json
{"ok": false, "error": {"type": "SerpApiError", "status_code": 401, "message": "invalid API key"}}
```

### FunctionAgent

Install `llama-index-llms-openai` separately to use OpenAI with LlamaIndex:

```bash
python -m pip install "llama-index-llms-openai>=0.5.0,<0.7.0"
```

This example also requires an `OPENAI_API_KEY` environment variable. The ToolSpec methods and HTTP client are synchronous. `FunctionAgent` accepts the exported tools; the `async def` and `await agent.run` below are only the agent workflow.

```python
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.thordata_serp import ThordataSerpToolSpec


async def answer(question: str):
    tool_spec = ThordataSerpToolSpec(api_key=os.environ["THORDATA_SERP_API_KEY"])
    agent = FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=OpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
        system_prompt="Use the SERP tools to answer the question.",
    )
    return await agent.run(question)
```

### Schema behavior

The package fetches the remote engine schema on first use, keeps it in a five-minute cache, and falls back to the bundled snapshot if a remote fetch fails before a schema is available. Only `list_engines` and `get_engine_schema` include `schema_source`, whose values are `remote`, `cache`, or `snapshot`. `search` returns `ok`, `status`, `engine`, and `data` without `schema_source`.

### Configuration

Use endpoint, timeout, and cache duration override settings for controlled networks or tests. The API key is always provided explicitly.

```python
import os

from llama_index.tools.thordata_serp import ThordataSerpToolSpec

tool_spec = ThordataSerpToolSpec(
    api_key=os.environ["THORDATA_SERP_API_KEY"],
    serp_endpoint="https://proxy.example.test/serp",
    schema_endpoint="https://proxy.example.test/schema",
    timeout=60,
    schema_timeout=15,
    schema_cache_ttl=300,
)
```

### Maintenance

Refresh the bundled schema snapshot after reviewing the remote schema:

```bash
python scripts/update_schema_snapshot.py
```

Run an opt-in smoke request with an explicitly supplied API key. For POSIX shells:

```bash
python scripts/smoke_serp.py --api-key "$THORDATA_SERP_API_KEY"
```

For PowerShell:

```powershell
python scripts/smoke_serp.py --api-key "$env:THORDATA_SERP_API_KEY"
```

For Windows cmd.exe:

```bat
python scripts/smoke_serp.py --api-key %THORDATA_SERP_API_KEY%
```

The smoke request requires a valid key and can make a live request. The test suite uses mock HTTP and does not require a real key.

### Support and license

For package support, contact Thordata through the [Thordata website](https://www.thordata.com). This package is released under the MIT License; see `LICENSE`.
