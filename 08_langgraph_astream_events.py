"""
=============================================================
 08_langgraph_astream_events.py  —  LangGraph astream_events()
=============================================================

WHAT IS graph.astream_events()?
---------------------------------
- The DEEPEST level of observability in LangGraph.
- Combines LangGraph's node-level events WITH LangChain's
  token-level events — you see EVERYTHING.
- Yields typed event dicts for every action at every level:
  graph start/end, node start/end, LLM start/end, each token.
- Best for: production monitoring, complex debugging, building
  rich UIs that show intermediate thinking steps.

EVENT HIERARCHY for a LangGraph call:
  on_chain_start      ← The graph itself starts
    on_chain_start    ← A node starts
      on_chat_model_start  ← LLM inside a node starts
        on_chat_model_stream  ← token arrives (many times)
      on_chat_model_end    ← LLM done
    on_chain_end      ← Node finishes
  on_chain_end        ← Graph finishes

ADDITIONAL LANGGRAPH-SPECIFIC EVENTS:
  on_chain_stream     → intermediate state updates from nodes
"""

import asyncio
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────
# Model + Graph setup
# ─────────────────────────────────────────────
llm = ChatOllama(model="llama3", temperature=0.7, num_predict=150)

class ResearchState(TypedDict):
    topic: str
    outline: str
    draft: str

def outline_node(state: ResearchState) -> dict:
    r = llm.invoke(f"Create a 3-point outline for a short article about: {state['topic']}")
    return {"outline": r.content}

def draft_node(state: ResearchState) -> dict:
    r = llm.invoke(f"Write a short article using this outline:\n{state['outline']}")
    return {"draft": r.content}

builder = StateGraph(ResearchState)
builder.add_node("outline", outline_node)
builder.add_node("draft", draft_node)
builder.add_edge(START, "outline")
builder.add_edge("outline", "draft")
builder.add_edge("draft", END)
graph = builder.compile()


# ─────────────────────────────────────────────
# EXAMPLE 1: See ALL events from a LangGraph run
# ─────────────────────────────────────────────
async def example_all_events():
    print("=" * 60)
    print("EXAMPLE 1: All events from a LangGraph run")
    print("=" * 60)

    event_count = 0
    async for event in graph.astream_events(
        {"topic": "quantum computing"},
        version="v2",
    ):
        event_count += 1
        print(f"  [{event_count:03d}] {event['event']:<35} | name={event['name']}")

    print(f"\nTotal events: {event_count}\n")


# ─────────────────────────────────────────────
# EXAMPLE 2: Build a rich progress UI
#   Show nodes starting/ending + stream tokens
# ─────────────────────────────────────────────
async def example_rich_progress_ui():
    print("=" * 60)
    print("EXAMPLE 2: Rich progress UI with events")
    print("=" * 60)

    current_node = None

    async for event in graph.astream_events(
        {"topic": "machine learning"},
        version="v2",
    ):
        evt = event["event"]
        name = event["name"]

        # ── Node lifecycle ────────────────────────────────────
        if evt == "on_chain_start" and name in ("outline", "draft"):
            current_node = name
            print(f"\n{'─'*50}")
            print(f"🟢 Starting node: [{name.upper()}]")

        elif evt == "on_chain_end" and name in ("outline", "draft"):
            print(f"\n✅ Finished node: [{name.upper()}]")

        # ── Token streaming (inside any node's LLM call) ──────
        elif evt == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)

    print(f"\n{'─'*50}\n")


# ─────────────────────────────────────────────
# EXAMPLE 3: Filter events by tags
#   Tag specific nodes to filter their events
# ─────────────────────────────────────────────
async def example_filter_by_tags():
    print("=" * 60)
    print("EXAMPLE 3: Filter events by tags")
    print("=" * 60)

    # Tag the LLM so we can filter events
    tagged_llm = llm.with_config(tags=["ollama-call"])

    class TaggedState(TypedDict):
        question: str
        answer: str

    def tagged_node(state: TaggedState) -> dict:
        r = tagged_llm.invoke(state["question"])
        return {"answer": r.content}

    tagged_builder = StateGraph(TaggedState)
    tagged_builder.add_node("qa", tagged_node)
    tagged_builder.add_edge(START, "qa")
    tagged_builder.add_edge("qa", END)
    tagged_graph = tagged_builder.compile()

    print("Only events tagged with 'ollama-call':\n")
    print("Answer: ", end="", flush=True)
    async for event in tagged_graph.astream_events(
        {"question": "What is a REST API?"},
        version="v2",
        include_tags=["ollama-call"],   # only events with this tag
    ):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)

    print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 4: Capture node inputs AND outputs
# ─────────────────────────────────────────────
async def example_capture_node_io():
    print("=" * 60)
    print("EXAMPLE 4: Capture node inputs and outputs")
    print("=" * 60)

    async for event in graph.astream_events(
        {"topic": "blockchain"},
        version="v2",
    ):
        evt = event["event"]
        name = event["name"]
        data = event["data"]

        # When a node starts, data["input"] has the state it received
        if evt == "on_chain_start" and name in ("outline", "draft"):
            print(f"\n[{name}] INPUT state:")
            if isinstance(data.get("input"), dict):
                for k, v in data["input"].items():
                    print(f"    {k} = {str(v)[:60]}...")

        # When a node ends, data["output"] has what it returned
        elif evt == "on_chain_end" and name in ("outline", "draft"):
            print(f"[{name}] OUTPUT:")
            output = data.get("output", {})
            if isinstance(output, dict):
                for k, v in output.items():
                    print(f"    {k} = {str(v)[:60]}...")

    print()


# ─────────────────────────────────────────────
# EXAMPLE 5: Token streaming on a message-based graph
#   Most realistic chat app pattern
# ─────────────────────────────────────────────
async def example_chat_token_streaming():
    print("=" * 60)
    print("EXAMPLE 5: Token streaming on a chat graph (realistic pattern)")
    print("=" * 60)

    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def chat_node(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    chat_builder = StateGraph(ChatState)
    chat_builder.add_node("assistant", chat_node)
    chat_builder.add_edge(START, "assistant")
    chat_builder.add_edge("assistant", END)
    chat_graph = chat_builder.compile()

    user_input = [HumanMessage(content="Explain recursion with a simple example.")]

    print("Assistant: ", end="", flush=True)

    full_response = ""
    async for event in chat_graph.astream_events(
        {"messages": user_input},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            print(token, end="", flush=True)
            full_response += token

    print(f"\n\nFull response length: {len(full_response)} chars\n")


# ─────────────────────────────────────────────
# EXAMPLE 6: Complete event inspection reference
#   Shows every possible field of an event
# ─────────────────────────────────────────────
async def example_event_field_reference():
    print("=" * 60)
    print("EXAMPLE 6: Complete event structure reference")
    print("=" * 60)

    shown_events = set()
    async for event in graph.astream_events(
        {"topic": "AI"},
        version="v2",
    ):
        evt_type = event["event"]
        if evt_type not in shown_events:
            shown_events.add(evt_type)
            print(f"\n📌 Event Type: {evt_type}")
            print(f"   event      : {event['event']}")
            print(f"   name       : {event['name']}")
            print(f"   run_id     : {event['run_id'][:8]}...")
            print(f"   parent_ids : {event['parent_ids'][:1]}...")
            print(f"   tags       : {event['tags']}")
            print(f"   metadata   : {event['metadata']}")
            print(f"   data keys  : {list(event['data'].keys())}")

    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
async def main():
    await example_all_events()
    await example_rich_progress_ui()
    await example_filter_by_tags()
    await example_capture_node_io()
    await example_chat_token_streaming()
    await example_event_field_reference()

    print("✅ LangGraph astream_events() demo complete.")
    print("   KEY TAKEAWAY: astream_events() gives you FULL observability —")
    print("   every node, every token, every step of the graph is visible.")


if __name__ == "__main__":
    asyncio.run(main())
