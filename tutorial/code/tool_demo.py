"""Call the Thordata SERP tool directly - no agent, no LLM involved."""

import json
import os
import textwrap

from dotenv import load_dotenv
from llama_index.tools.thordata_serp import ThordataSerpToolSpec

load_dotenv()

tool_spec = ThordataSerpToolSpec(api_key=os.environ["THORDATA_SERP_API_TOKEN"])

# 1. Which engines can this tool reach?
engines = json.loads(tool_spec.list_engines())
keys = [engine["key"] for engine in engines["engines"]]
print(f"1) Engines - schema v{engines['schema_version']} from {engines['schema_source']}")
print(f"   {len(keys)} engines, default is {engines['default_engine']}")
print(textwrap.fill(", ".join(keys[:10]), width=96, initial_indent="   ", subsequent_indent="   "))

# 2. What parameters does the google engine accept?
schema = json.loads(tool_spec.get_engine_schema("google"))
fields = [field["key"] for group in schema["engine"]["groups"] for field in group["fields"]]
print(f"\n2) The google engine accepts {len(fields)} parameters")
print(textwrap.fill(", ".join(fields), width=96, initial_indent="   ", subsequent_indent="   "))

# 3. Run a search and print the organic results.
result = json.loads(
    tool_spec.search(
        engine="google",
        query="best coffee grinders 2026",
        params={"num": 3, "gl": "us", "hl": "en"},
        response_mode="compact",
    )
)
if not result["ok"]:
    print(f"\n3) Search failed - {result['error']['type']}: {result['error']['message']}")
    raise SystemExit(1)
print(f"\n3) Search - ok={result['ok']} status={result['status']} engine={result['engine']}")
for item in result["data"]["organic"]:
    print(f"   {item['position']}. {item['title']}")
    print(f"      {item['link']}")
