"""
=============================================================
 09_comparison_summary.py  —  Side-by-side comparison of all methods
=============================================================

This file runs a SINGLE question through ALL 5 invocation methods
so you can directly compare their behavior and output format.

Run this file after understanding the individual examples.
"""

import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3", temperature=0.0, num_predict=80)

QUESTION = "What is Python? Answer in exactly one sentence."

# ════════════════════════════════════════════════════════════
# 1. invoke() — blocking, returns complete AIMessage
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("METHOD 1: invoke()  [SYNC | BLOCKING | COMPLETE RESPONSE]")
print("═" * 60)
print("→ Blocks thread. Returns ONLY after full response is ready.")
print("─" * 60)

result = llm.invoke(QUESTION)

print(f"Return type : {type(result).__name__}")   # AIMessage
print(f"Content     : {result.content}")
print(f"Blocking?   : YES — your code waited for the full response")


# ════════════════════════════════════════════════════════════
# 2. ainvoke() — non-blocking, returns complete AIMessage
# ════════════════════════════════════════════════════════════
async def run_ainvoke():
    print("\n" + "═" * 60)
    print("METHOD 2: ainvoke()  [ASYNC | NON-BLOCKING | COMPLETE RESPONSE]")
    print("═" * 60)
    print("→ Releases event loop. Returns ONLY after full response is ready.")
    print("─" * 60)

    result = await llm.ainvoke(QUESTION)

    print(f"Return type : {type(result).__name__}")   # AIMessage
    print(f"Content     : {result.content}")
    print(f"Blocking?   : NO — event loop was free while waiting")


# ════════════════════════════════════════════════════════════
# 3. stream() — blocking, yields chunks in real-time
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("METHOD 3: stream()  [SYNC | BLOCKING | STREAMING CHUNKS]")
print("═" * 60)
print("→ Blocks thread. Yields AIMessageChunks as they arrive.")
print("─" * 60)

chunk_count = 0
print("Chunks → ", end="")
for chunk in llm.stream(QUESTION):
    chunk_count += 1
    print(f"[{chunk.content}]", end="", flush=True)

print(f"\nReturn type per iter : AIMessageChunk")
print(f"Total chunks         : {chunk_count}")
print(f"Blocking?            : YES — thread waited, but output appeared live")


# ════════════════════════════════════════════════════════════
# 4. astream() — non-blocking, yields chunks in real-time
# ════════════════════════════════════════════════════════════
async def run_astream():
    print("\n" + "═" * 60)
    print("METHOD 4: astream()  [ASYNC | NON-BLOCKING | STREAMING CHUNKS]")
    print("═" * 60)
    print("→ Releases event loop. Yields AIMessageChunks as they arrive.")
    print("─" * 60)

    chunk_count = 0
    print("Chunks → ", end="")
    async for chunk in llm.astream(QUESTION):
        chunk_count += 1
        print(f"[{chunk.content}]", end="", flush=True)

    print(f"\nReturn type per iter : AIMessageChunk")
    print(f"Total chunks         : {chunk_count}")
    print(f"Blocking?            : NO — event loop free between each chunk")


# ════════════════════════════════════════════════════════════
# 5. astream_events() — async, yields structured event dicts
# ════════════════════════════════════════════════════════════
async def run_astream_events():
    print("\n" + "═" * 60)
    print("METHOD 5: astream_events()  [ASYNC | NON-BLOCKING | EVENTS]")
    print("═" * 60)
    print("→ Yields structured event dicts for EVERY step (start/stream/end).")
    print("─" * 60)

    events_by_type = {}
    async for event in llm.astream_events(QUESTION, version="v2"):
        etype = event["event"]
        events_by_type[etype] = events_by_type.get(etype, 0) + 1

    print(f"Return type per iter : dict (event)")
    print(f"Events received:")
    for etype, count in events_by_type.items():
        print(f"   {etype:<35} × {count}")
    print(f"Blocking?            : NO — fully async and non-blocking")


# ════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ════════════════════════════════════════════════════════════
def print_comparison_table():
    print("\n" + "═" * 60)
    print("FINAL COMPARISON TABLE")
    print("═" * 60)
    print(f"{'Method':<20} {'Sync/Async':<12} {'Blocks?':<10} {'Returns':<25} {'Best For'}")
    print("─" * 100)
    rows = [
        ("invoke()",         "Sync",  "YES", "Complete AIMessage",   "Scripts, batch, tests"),
        ("ainvoke()",        "Async", "NO",  "Complete AIMessage",   "Concurrent LLM calls"),
        ("stream()",         "Sync",  "YES", "AIMessageChunk iter",  "CLI tools, notebooks"),
        ("astream()",        "Async", "NO",  "AIMessageChunk iter",  "Web APIs, WebSockets"),
        ("astream_events()", "Async", "NO",  "Event dict iter",      "Debugging, observability"),
    ]
    for row in rows:
        print(f"  {row[0]:<18} {row[1]:<12} {row[2]:<10} {row[3]:<25} {row[4]}")

    print("\n" + "═" * 60)
    print("LANGGRAPH ADDITIONS")
    print("═" * 60)
    print(f"{'Method':<25} {'Returns':<30} {'Best For'}")
    print("─" * 90)
    lg_rows = [
        ("graph.invoke()",          "Final state dict",            "Simple workflows"),
        ("graph.ainvoke()",         "Final state dict (async)",    "Async workflows"),
        ('graph.stream("updates")', "Per-node state changes",      "Step-by-step progress"),
        ('graph.stream("values")',  "Full state after each node",  "Full state tracking"),
        ('graph.stream("messages")',"Tokens from LLM nodes",       "Chat token streaming"),
        ("graph.astream()",         "Same as stream() but async",  "Async graph execution"),
        ("graph.astream_events()",  "Full event hierarchy",        "Full observability"),
    ]
    for row in lg_rows:
        print(f"  {row[0]:<25} {row[1]:<30} {row[2]}")


async def main():
    await run_ainvoke()
    await run_astream()
    await run_astream_events()
    print_comparison_table()
    print("\n✅ Comparison complete! Study 01_*.py → 08_*.py for detailed examples of each.")


if __name__ == "__main__":
    run_ainvoke_task = asyncio.run(main())
