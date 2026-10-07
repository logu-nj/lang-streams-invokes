"""
=============================================================
 05_astream_events.py  —  ASYNCHRONOUS  astream_events()
=============================================================

WHAT IS astream_events()?
--------------------------
- The MOST POWERFUL and DETAILED streaming method.
- Yields structured EVENT dictionaries instead of raw chunks.
- Gives you visibility into EVERY step: when a chain starts,
  when the LLM starts, each token, when things end, etc.
- Essential for: debugging complex chains, building UIs that
  show intermediate steps, progress indicators, tool call tracking.

EVENT STRUCTURE:
  Each event is a dict:
  {
    "event":      str,   # e.g. "on_chat_model_stream", "on_chain_start"
    "name":       str,   # name of the component
    "run_id":     str,   # unique ID for this run
    "tags":       list,  # tags from config
    "metadata":   dict,  # metadata from config
    "data":       dict,  # event-specific payload
    "parent_ids": list,  # IDs of parent runs
  }

COMMON EVENT TYPES:
  on_chain_start          → a chain/runnable started
  on_chain_end            → a chain/runnable finished
  on_chat_model_start     → LLM started generating
  on_chat_model_stream    → a token chunk arrived  ← most frequent
  on_chat_model_end       → LLM finished
  on_prompt_start         → prompt template started
  on_prompt_end           → prompt template finished
  on_tool_start           → a tool was called
  on_tool_end             → a tool call finished
  on_retriever_start      → a retriever started
  on_retriever_end        → a retriever finished

VERSION NOTE:
  astream_events() takes a `version` param: "v1" or "v2" (default v2)
  v2 is recommended for newer LangChain versions.
"""

import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────
llm = ChatOllama(
    model="llama3",
    temperature=0.7,
    num_predict=200,
)


# ─────────────────────────────────────────────
# EXAMPLE 1: See ALL events from a simple LLM call
# ─────────────────────────────────────────────
async def example_all_events():
    print("=" * 60)
    print("EXAMPLE 1: All events from a simple LLM call")
    print("=" * 60)

    event_count = 0
    async for event in llm.astream_events(
        "What is 1+1?",
        version="v2",   # always specify version
    ):
        event_count += 1
        event_type = event["event"]
        name = event["name"]
        data = event["data"]
        print(f"  [{event_count:02d}] event={event_type:<30} name={name}")

    print(f"\nTotal events: {event_count}\n")


# ─────────────────────────────────────────────
# EXAMPLE 2: Filter ONLY token chunks — build a streamer
# ─────────────────────────────────────────────
async def example_filter_tokens():
    print("=" * 60)
    print("EXAMPLE 2: Filter only token chunks (on_chat_model_stream)")
    print("=" * 60)

    print("Streamed output: ", end="", flush=True)

    async for event in llm.astream_events(
        "Tell me a fun fact about space.",
        version="v2",
    ):
        # We only care about token-level events
        if event["event"] == "on_chat_model_stream":
            # data["chunk"] is an AIMessageChunk
            token = event["data"]["chunk"].content
            print(token, end="", flush=True)

    print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 3: Full chain — see ALL stages: prompt → llm → parser
# ─────────────────────────────────────────────
async def example_chain_events():
    print("=" * 60)
    print("EXAMPLE 3: Events from a full chain (prompt | llm | parser)")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ])
    parser = StrOutputParser()

    # Full chain: prompt → llm → string parser
    chain = prompt | llm | parser

    # include_names lets you filter by component name
    print("--- All Events from Chain ---")
    async for event in chain.astream_events(
        {"question": "What is an API?"},
        version="v2",
    ):
        event_type = event["event"]
        name = event["name"]

        if event_type == "on_chain_start":
            print(f"🟢 CHAIN START  : {name}")

        elif event_type == "on_prompt_start":
            print(f"📝 PROMPT START : {name}")

        elif event_type == "on_prompt_end":
            print(f"📝 PROMPT END   : {name}")

        elif event_type == "on_chat_model_start":
            print(f"🤖 LLM START    : {name}")

        elif event_type == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            print(token, end="", flush=True)

        elif event_type == "on_chat_model_end":
            print(f"\n🤖 LLM END      : {name}")

        elif event_type == "on_chain_end":
            print(f"🔴 CHAIN END    : {name}")

    print()


# ─────────────────────────────────────────────
# EXAMPLE 4: Using include_names to watch specific components
# ─────────────────────────────────────────────
async def example_include_names():
    print("=" * 60)
    print("EXAMPLE 4: include_names — watch only specific components")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
    ])

    # Give the LLM a name so we can filter by it
    named_llm = llm.with_config(run_name="MyOllamaLLM")

    chain = prompt | named_llm

    async for event in chain.astream_events(
        {"question": "What is machine learning? One sentence."},
        version="v2",
        include_names=["MyOllamaLLM"],   # only events from this component
    ):
        print(f"  event={event['event']:<30} | data keys={list(event['data'].keys())}")


# ─────────────────────────────────────────────
# EXAMPLE 5: include_types — filter by event type
# ─────────────────────────────────────────────
async def example_include_types():
    print("=" * 60)
    print("EXAMPLE 5: include_types — filter by component type")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
    ])
    chain = prompt | llm

    print("Only chat_model events:\n")
    async for event in chain.astream_events(
        {"question": "Name 3 planets."},
        version="v2",
        include_types=["chat_model"],  # only events where type == "chat_model"
    ):
        event_type = event["event"]
        if event_type == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        else:
            print(f"\n[{event_type}]", end="")

    print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 6: Full event inspection — see every field
# ─────────────────────────────────────────────
async def example_full_event_inspection():
    print("=" * 60)
    print("EXAMPLE 6: Full event inspection — every field explained")
    print("=" * 60)

    event_count = 0
    async for event in llm.astream_events(
        "Say hello.",
        version="v2",
        include_names=["ChatOllama"],
    ):
        event_count += 1
        if event_count <= 3:  # just show first 3 events in detail
            print(f"\n--- Event #{event_count} ---")
            print(f"  event     : {event['event']}")
            print(f"  name      : {event['name']}")
            print(f"  run_id    : {event['run_id']}")
            print(f"  tags      : {event['tags']}")
            print(f"  metadata  : {event['metadata']}")
            print(f"  data keys : {list(event['data'].keys())}")
            if event['event'] == 'on_chat_model_stream':
                print(f"  token     : {repr(event['data']['chunk'].content)}")

    print(f"\n...and {event_count - 3} more events total.\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
async def main():
    await example_all_events()
    await example_filter_tokens()
    await example_chain_events()
    await example_include_names()
    await example_include_types()
    await example_full_event_inspection()

    print("✅ astream_events() demo complete.")
    print("   KEY TAKEAWAY: astream_events() = full observability of every step in a chain.")
    print("   Use it when you need to know WHAT is happening, not just the final output.")


if __name__ == "__main__":
    asyncio.run(main())
