## LlamaIndex integration for Thordata

### Hero

[**Thordata**](https://www.thordata.com/products/serp-api/llamaindex-integration?op=flora&from=github) gives your LlamaIndex Agent **real-time, structured web search and fixed Web Scraper tasks** through one ToolSpec.

`llama-index-tools-thordata-serp` version `0.2.0` plugs Thordata's SERP API and Web Scraper API into your LlamaIndex workflow. The public import path is `llama_index.tools.thordata_serp`. `ThordataSerpToolSpec` exposes live search engines and seven fixed Web Scraper spiders through predictable, agent-ready JSON.

**Why teams pick it:**

- **Schema-driven, no guessing** — every search parameter is validated against the selected engine's live schema; unsupported fields are ignored before the request is sent.
- **34 engines in the bundled snapshot** — web, news, images, maps, shopping, flights, scholar, and local engines are unified behind one `search` tool, while remote schema updates can add new engines without a package release.
- **Output built for agents** — a predictable `ok` / `status` / `engine` / `data` envelope; `compact` mode strips verbose metadata to reduce context usage.
- **Web Scraper tasks** — the `scrape` tool runs one of seven fixed spiders, waits for the task, and returns `records` with task metadata.
- **Robust and reliable** — bounded requests and responses, normalized errors, API-key redaction, and a built-in schema snapshot fallback.
- **Works with existing Thordata access** — pass your SERP API key explicitly and keep the package configuration under your control.

### Install

Python `>=3.10` is required.

```bash
python -m pip install llama-index-tools-thordata-serp
```

The runtime dependencies are `llama-index-core>=0.13.0,<0.15`, `pydantic>=2.0,<3.0`, and `requests>=2.32,<3.0`.

Create a Thordata SERP API key in the [Thordata dashboard](https://www.thordata.com/products/serp-api/llamaindex-integration?op=flora&from=github) and set it in the calling environment. The tool spec does not read environment variables itself; read and pass the key explicitly:

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

#### `scrape`

`scrape` submits one Web Scraper task and waits synchronously for its final JSON records. It uses the same explicitly supplied API key as the SERP tools. The maximum wait is 300 seconds and the status endpoint is polled every 5 seconds; a call never resubmits an unknown task.

The tool accepts only these seven spider IDs and their fixed fields:

| Spider | Required fields | Optional fields |
| --- | --- | --- |
| `linkedin_company_information_by-url` | `url` | — |
| `linkedin_job_listings_information_by-job-url` | `job_url` | — |
| `youtube_video-post_by-keyword` | `keyword` | `num_of_posts` |
| `facebook_profile_by-profiles-url` | `url` | — |
| `youtube_video-post_by-explore` | `url` | `all_tabs` |
| `twitter_profile_by-profileurl` | `url` | — |
| `amazon_product_by-category-url` | `url` | `page_turning`, `sort_by` |

URL fields must be complete HTTP/HTTPS URLs without embedded credentials. `sort_by` defaults to `Best Sellers` and accepts `Newest Arrivals`, `Avg. Customer Review`, `Price: High to Low`, `Price: Low to High`, or `Featured`. Unknown fields and invalid values are rejected before a request is sent.

For Amazon category scraping, `page_turning` is optional. When omitted or set to `None` (JSON `null`), the SDK leaves it out of the request and uses the server's default behavior; it does not assume one page. Pass it explicitly to control the number of pages. Supplied values must be positive integers (or digit strings) no greater than `9007199254740991`; empty strings, booleans, zero, negative numbers, and fractions are rejected.

For an initial Amazon check, request one page explicitly. Omitting the page count can request a larger server-default workload and reach the 300-second wait limit. A single page reduces the requested work but does not guarantee completion within that limit:

```python
import json
import os

from llama_index.tools.thordata_serp import ThordataSerpToolSpec

tool_spec = ThordataSerpToolSpec(api_key=os.environ["THORDATA_SERP_API_KEY"])
amazon_result = json.loads(tool_spec.scrape(
    spider_id="amazon_product_by-category-url",
    params={
        "url": "https://www.amazon.com/s?i=baby-products-intl-ship",
        "page_turning": 1,
        "sort_by": "Best Sellers",
    },
    max_wait_seconds=300,
))
if amazon_result["ok"]:
    print(amazon_result["records"])
else:
    print(amazon_result["error"])
```

`wait_timeout` stops local waiting; it does not cancel the remote task. Keep the returned `task_id` and check the task in the Thordata dashboard before starting another request. The SDK does not automatically resubmit timed-out tasks.

The sorting names used in the official documentation are also accepted as aliases. Existing sort values keep their behavior:

| Accepted alias | Value sent to Thordata |
| --- | --- |
| `Average Rating` | `Avg. Customer Review` |
| `Featured Recommendations` | `Featured` |

```python
import json

scrape_result = json.loads(
    tool_spec.scrape(
        spider_id="linkedin_company_information_by-url",
        params={"url": "https://www.linkedin.com/company/acme"},
        max_wait_seconds=300,
        include_errors=True,
        file_name="{{TasksID}}",
    )
)
print(scrape_result["status"], scrape_result["task_id"])
print(scrape_result["records"])
```

Successful results contain `ok`, `status`, `spider_id`, `task_id`, `record_count`, `error_record_count`, and `records`:

```json
{"ok": true, "status": "completed", "spider_id": "linkedin_company_information_by-url", "task_id": "task-123", "record_count": 1, "error_record_count": 0, "records": [{"name": "Acme"}]}
```

`status` is `completed` when all records succeed, `partial` when some records fail, and `empty` when no meaningful business record is returned. Failures use the same `ok=false` style as SERP errors and redact API keys, request signatures, passwords, and credentialed download URLs.

When a task reports `Failed`, `include_errors=True` enables one attempt to retrieve its error records. This attempt uses at most 15 seconds of the remaining wait budget. The error message includes the available server reason, and `error.service_code` contains its error code when available. If diagnostics fail, time out, or are disabled, the tool preserves `scrape_failed`, the task ID, and any reason already supplied by the status response. A failed task is never reported as a successful result.

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

For package support, contact Thordata through the [Thordata website](https://www.thordata.com/products/serp-api/llamaindex-integration?op=flora&from=github). This package is released under the MIT License; see `LICENSE`.
