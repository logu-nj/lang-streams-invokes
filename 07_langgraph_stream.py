"""
=============================================================
 07_langgraph_stream.py  —  LangGraph  graph.stream()
=============================================================

WHAT IS graph.stream()?
------------------------
- Runs the graph but yields state snapshots after EACH NODE completes.
- You see the state EVOLVE step-by-step as nodes execute.
- Unlike LLM stream() which gives tokens, graph.stream() gives
  you STATE UPDATES at the graph/node level.
- Best for: debugging multi-step workflows, showing progress
  in a UI ("Step 1 done, Step 2 running..."), tracing execution.

STREAM MODES:
  graph.stream() has a `stream_mode` parameter:
  ┌─────────────────┬───────────────────────────────────────────────────┐
  │  stream_mode    │  What each yielded item contains                  │
  ├─────────────────┼───────────────────────────────────────────────────┤
  │  "values"       │ Full state dict after each node                   │
  │  "updates"      │ Only the CHANGES a node made to state (default)   │
  │  "messages"     │ Token-level streaming of LLM messages             │
  │  "debug"        │ Very verbose debug info for every event           │
  └─────────────────┴───────────────────────────────────────────────────┘

FLOW:
  graph.stream(input, stream_mode="updates")
  → yields {"node_name": {state_update}}  after node1
  → yields {"node_name": {state_update}}  after node2
  → ...
"""

import asyncio
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────
# Model + shared graph setup
# ─────────────────────────────────────────────
llm = ChatOllama(model="llama3", temperature=0.7, num_predict=200)

class PipelineState(TypedDict):
    question: str
    simplified: str
    answer: str
    summary: str

def simplify_node(state: PipelineState) -> dict:
    """Rephrase the question more clearly."""
    r = llm.invoke(f"Rephrase this question more clearly in one sentence: {state['question']}")
    return {"simplified": r.content}

def answer_node(state: PipelineState) -> dict:
    """Answer the simplified question."""
    r = llm.invoke(f"Answer this: {state['simplified']}")
    return {"answer": r.content}

def summarize_node(state: PipelineState) -> dict:
    """Summarize the answer to one line."""
    r = llm.invoke(f"Summarize this in one line: {state['answer']}")
    return {"summary": r.content}

# Build the graph
builder = StateGraph(PipelineState)
builder.add_node("simplify", simplify_node)
builder.add_node("answer", answer_node)
builder.add_node("summarize", summarize_node)
builder.add_edge(START, "simplify")
builder.add_edge("simplify", "answer")
builder.add_edge("answer", "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()


# ─────────────────────────────────────────────
# EXAMPLE 1: stream_mode="updates"  (DEFAULT)
#   → yields only what each node CHANGED
# ─────────────────────────────────────────────
print("=" * 60)
print('EXAMPLE 1: stream_mode="updates" — see each node\'s output')
print("=" * 60)

input_data = {"question": "What is deep learning?"}

for step in graph.stream(input_data, stream_mode="updates"):
    # step is a dict: {"node_name": {key: value, ...}}
    for node_name, node_output in step.items():
        print(f"\n📦 Node '{node_name}' produced:")
        for key, value in node_output.items():
            print(f"   {key}: {str(value)[:80]}...")

print()


# ─────────────────────────────────────────────
# EXAMPLE 2: stream_mode="values"
#   → yields the FULL state after each node
# ─────────────────────────────────────────────
print("=" * 60)
print('EXAMPLE 2: stream_mode="values" — full state after each node')
print("=" * 60)

for i, state_snapshot in enumerate(
    graph.stream({"question": "What is a neural network?"}, stream_mode="values")
):
    print(f"\n🔵 State after step {i + 1}:")
    for key, value in state_snapshot.items():
        val_str = str(value)[:70] + "..." if value else "(empty)"
        print(f"   {key}: {val_str}")

print()


# ─────────────────────────────────────────────
# EXAMPLE 3: stream_mode="messages"
#   → token-level streaming for message-based graphs
# ─────────────────────────────────────────────
print("=" * 60)
print('EXAMPLE 3: stream_mode="messages" — token-by-token output')
print("=" * 60)

# For this mode we need a message-based state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

chat_builder = StateGraph(ChatState)
chat_builder.add_node("chat", chat_node)
chat_builder.add_edge(START, "chat")
chat_builder.add_edge("chat", END)
chat_graph = chat_builder.compile()

print("Streaming tokens: ", end="", flush=True)
for chunk, metadata in chat_graph.stream(
    {"messages": [HumanMessage("Write a haiku about AI.")]},
    stream_mode="messages",
):
    # chunk is a BaseMessage or AIMessageChunk
    if hasattr(chunk, "content") and chunk.content:
        print(chunk.content, end="", flush=True)

print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 4: stream_mode="debug"
#   → very verbose, great for debugging
# ─────────────────────────────────────────────
print("=" * 60)
print('EXAMPLE 4: stream_mode="debug" — verbose debug events')
print("=" * 60)

debug_count = 0
for event in graph.stream(
    {"question": "What is Python?"},
    stream_mode="debug",
):
    debug_count += 1
    print(f"  Debug event #{debug_count}: type={event.get('type', 'N/A')}, "
          f"step={event.get('step', 'N/A')}")
    if debug_count >= 6:  # limit output for readability
        print("  ... (truncated for brevity)")
        break

print()


# ─────────────────────────────────────────────
# EXAMPLE 5: Async version — graph.astream()
#   Same stream_mode options, but async
# ─────────────────────────────────────────────
async def example_astream():
    print("=" * 60)
    print("EXAMPLE 5: graph.astream() — async version of stream()")
    print("=" * 60)

    print("Node-by-node progress:\n")
    async for step in graph.astream(
        {"question": "What is reinforcement learning?"},
        stream_mode="updates",
    ):
        for node_name, output in step.items():
            print(f"  ✅ '{node_name}' completed")
            for key, value in output.items():
                print(f"     {key} = {str(value)[:70]}...")

    print()


asyncio.run(example_astream())


# ─────────────────────────────────────────────
# EXAMPLE 6: Combining multiple stream modes
#   You can pass a LIST to stream_mode!
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 6: Multiple stream modes at once")
print("=" * 60)

for item in graph.stream(
    {"question": "What is an API?"},
    stream_mode=["updates", "values"],   # get BOTH
):
    # When using list, each item is a tuple: (mode_name, data)
    mode, data = item
    print(f"  Mode: {mode:<10} | Keys in data: {list(data.keys())}")

print("\n✅ LangGraph stream() demo complete.")
print("   KEY TAKEAWAY: stream_mode controls the GRANULARITY of what you see.")
print("   'updates' = per-node changes | 'values' = full state | 'messages' = tokens")
