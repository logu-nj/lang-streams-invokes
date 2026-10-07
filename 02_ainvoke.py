"""
=============================================================
 02_ainvoke.py  —  ASYNCHRONOUS  ainvoke()
=============================================================

WHAT IS ainvoke()?
------------------
- The ASYNC version of invoke().
- Must be called with `await` inside an `async` function.
- Does NOT block the event loop while waiting for the LLM.
- Other async tasks can run concurrently while waiting.
- Still returns a COMPLETE AIMessage (not streamed).
- Best for: FastAPI endpoints, async web apps, running
  multiple LLM calls CONCURRENTLY with asyncio.gather().

FLOW:
  async fn  →  await ainvoke()  →  [event loop free to do other work]
            →  returns full AIMessage when ready

COMPARISON vs invoke():
  invoke()   → blocks the thread until LLM finishes
  ainvoke()  → releases the event loop, returns when LLM finishes
"""

import asyncio
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# Model setup (same options as invoke)
# ─────────────────────────────────────────────
llm = ChatOllama(
    model="llama3",
    temperature=0.7,
    num_predict=256,
)


# ─────────────────────────────────────────────
# EXAMPLE 1: Basic ainvoke
# ─────────────────────────────────────────────
async def example_basic_ainvoke():
    print("=" * 60)
    print("EXAMPLE 1: Basic ainvoke()")
    print("=" * 60)

    # await is REQUIRED — this is a coroutine
    response = await llm.ainvoke("What is the capital of Japan? One sentence.")

    print(f"Type    : {type(response)}")
    print(f"Content : {response.content}")
    print()


# ─────────────────────────────────────────────
# EXAMPLE 2: ainvoke with messages list
# ─────────────────────────────────────────────
async def example_ainvoke_with_messages():
    print("=" * 60)
    print("EXAMPLE 2: ainvoke() with messages list")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a Python tutor. Be concise."),
        HumanMessage(content="What is a list comprehension?"),
    ]

    response = await llm.ainvoke(messages)
    print(f"Content : {response.content}")
    print()


# ─────────────────────────────────────────────
# EXAMPLE 3: THE REAL POWER — concurrent ainvoke calls
# This is WHY you use ainvoke() over invoke()!
# Run 3 LLM calls at the SAME TIME instead of one by one.
# ─────────────────────────────────────────────
async def example_concurrent_ainvoke():
    print("=" * 60)
    print("EXAMPLE 3: Concurrent ainvoke() with asyncio.gather()")
    print("=" * 60)
    print("Running 3 LLM calls SIMULTANEOUSLY...\n")

    questions = [
        "What is photosynthesis? One sentence.",
        "What is gravity? One sentence.",
        "What is democracy? One sentence.",
    ]

    # ── SEQUENTIAL (slow) — for comparison ───────────────────
    start = time.time()
    sequential_results = []
    for q in questions:
        r = await llm.ainvoke(q)
        sequential_results.append(r.content)
    sequential_time = time.time() - start

    print(f"Sequential time: {sequential_time:.2f}s")
    for i, r in enumerate(sequential_results, 1):
        print(f"  Q{i}: {r[:60]}...")
    print()

    # ── CONCURRENT (fast) — all 3 run at the same time ───────
    start = time.time()
    concurrent_results = await asyncio.gather(
        llm.ainvoke(questions[0]),
        llm.ainvoke(questions[1]),
        llm.ainvoke(questions[2]),
    )
    concurrent_time = time.time() - start

    print(f"Concurrent time: {concurrent_time:.2f}s  ← much faster!")
    for i, r in enumerate(concurrent_results, 1):
        print(f"  Q{i}: {r.content[:60]}...")
    print()


# ─────────────────────────────────────────────
# EXAMPLE 4: ainvoke with LCEL chain + config
# ─────────────────────────────────────────────
async def example_ainvoke_chain():
    print("=" * 60)
    print("EXAMPLE 4: ainvoke() on an LCEL chain with config")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {topic}."),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    response = await chain.ainvoke(
        {"topic": "astronomy", "question": "What is a black hole? 2 sentences."},
        config={
            "run_name": "blackhole-query",
            "tags": ["science", "async-demo"],
            "metadata": {"source": "02_ainvoke.py"},
        }
    )

    print(f"Content : {response.content}")
    print()


# ─────────────────────────────────────────────
# Main runner — runs all examples
# ─────────────────────────────────────────────
async def main():
    await example_basic_ainvoke()
    await example_ainvoke_with_messages()
    await example_concurrent_ainvoke()
    await example_ainvoke_chain()
    print("✅ ainvoke() demo complete.")
    print("   KEY TAKEAWAY: ainvoke() shines when running MULTIPLE calls concurrently.")


if __name__ == "__main__":
    asyncio.run(main())
