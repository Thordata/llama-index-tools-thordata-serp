"""A LlamaIndex agent that answers questions with live Thordata SERP data."""

import asyncio
import os

from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
from llama_index.llms.openai_like import OpenAILike
from llama_index.tools.thordata_serp import ThordataSerpToolSpec

load_dotenv()

SYSTEM_PROMPT = (
    "You are a research assistant with a live web search tool. "
    "Always call the search tool before you answer, keep the answer under 120 words, "
    "and cite the result URLs you used."
)


def build_agent() -> FunctionAgent:
    """Wire the Thordata SERP tools and an OpenAI-compatible LLM into one agent."""
    tool_spec = ThordataSerpToolSpec(api_key=os.environ["THORDATA_SERP_API_TOKEN"])
    llm = OpenAILike(
        model=os.environ["LLM_MODEL"],
        api_base=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_TOKEN"],
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=int(os.environ.get("LLM_CONTEXT_WINDOW", "32768")),
        timeout=300,
    )
    return FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )


async def ask(agent: FunctionAgent, question: str) -> str:
    """Run one turn and report every tool call while the agent works."""
    handler = agent.run(question)
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            print(f"  [tool] {event.tool_name}({event.tool_kwargs})")
    return str(await handler)


async def main() -> None:
    agent = build_agent()
    print("Thordata SERP agent ready - ask a question, or type 'exit'.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print(f"\nAgent: {await ask(agent, question)}")


if __name__ == "__main__":
    asyncio.run(main())
