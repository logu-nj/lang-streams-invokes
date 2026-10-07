"""
=============================================================
 04_astream.py  —  ASYNCHRONOUS  astream()
=============================================================

WHAT IS astream()?
------------------
- The ASYNC version of stream().
- An async generator — you iterate it with `async for`.
- Yields AIMessageChunk objects as tokens arrive, WITHOUT
  blocking the event loop.
- Best for: FastAPI with Server-Sent Events (SSE), WebSocket
  streaming, async web frameworks, anywhere you want to stream
  to a client without blocking other requests.

FLOW:
  async for chunk in astream():
      → yields chunk1  (event loop free between chunks)
      → yields chunk2
      → ...

COMPARISON:
  stream()   → sync generator, blocks thread between chunks
  astream()  → async generator, releases event loop between chunks

USE CASE MATRIX:
  ┌─────────────┬────────────┬────────────┐
  │             │  Blocking  │  Non-Block │
  ├─────────────┼────────────┼────────────┤
  │  Full resp  │  invoke()  │ ainvoke()  │
  │  Streaming  │  stream()  │ astream()  │
  └─────────────┴────────────┴────────────┘
"""

import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────
llm = ChatOllama(
    model="llama3",
    temperature=0.7,
    num_predict=300,
)


# ─────────────────────────────────────────────
# EXAMPLE 1: Basic astream — async token streaming
# ─────────────────────────────────────────────
async def example_basic_astream():
    print("=" * 60)
    print("EXAMPLE 1: Basic astream() — async token streaming")
    print("=" * 60)

    print("Response: ", end="", flush=True)

    # `async for` is the KEY difference from stream()
    async for chunk in llm.astream("Write a haiku about coding."):
        print(chunk.content, end="", flush=True)

    print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 2: astream with message list
# ─────────────────────────────────────────────
async def example_astream_messages():
    print("=" * 60)
    print("EXAMPLE 2: astream() with messages list")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a motivational coach. Be brief."),
        HumanMessage(content="Give me one powerful piece of advice for beginners."),
    ]

    print("Coach says: ", end="", flush=True)
    async for chunk in llm.astream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 3: astream on an LCEL chain
# ─────────────────────────────────────────────
async def example_astream_chain():
    print("=" * 60)
    print("EXAMPLE 3: astream() on prompt | llm chain")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}."),
        ("human", "{question}"),
    ])
    chain = prompt | llm

    print("Answer: ", end="", flush=True)
    async for chunk in chain.astream({
        "role": "cybersecurity expert",
        "question": "What is a SQL injection attack? 2 sentences.",
    }):
        print(chunk.content, end="", flush=True)
    print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 4: THE REAL POWER — streaming to multiple "clients"
# concurrently (simulates a real web server scenario)
# ─────────────────────────────────────────────
async def stream_to_client(client_id: str, question: str):
    """Simulates streaming a response to a client (e.g. WebSocket)."""
    print(f"\n[Client {client_id}] Streaming: ", end="", flush=True)
    async for chunk in llm.astream(question):
        print(chunk.content, end="", flush=True)
    print(f"  ← [Client {client_id} done]")


async def example_concurrent_streaming():
    print("=" * 60)
    print("EXAMPLE 4: Streaming to MULTIPLE clients concurrently")
    print("=" * 60)
    print("All 3 streams run simultaneously:\n")

    # asyncio.gather runs all 3 streaming calls at the same time
    # This is impossible with synchronous stream()!
    await asyncio.gather(
        stream_to_client("A", "What is DNA? One sentence."),
        stream_to_client("B", "What is WiFi? One sentence."),
        stream_to_client("C", "What is SQL? One sentence."),
    )
    print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 5: astream with config + chunk accumulation
# ─────────────────────────────────────────────
async def example_astream_with_config():
    print("=" * 60)
    print("EXAMPLE 5: astream() with config + accumulating chunks")
    print("=" * 60)

    full_content = ""
    chunk_count = 0

    print("Streaming: ", end="", flush=True)
    async for chunk in llm.astream(
        "Explain async programming in Python in 3 bullet points.",
        config={
            "run_name": "async-explanation",
            "tags": ["python", "async", "astream-demo"],
            "metadata": {"file": "04_astream.py"},
        }
    ):
        print(chunk.content, end="", flush=True)
        full_content += chunk.content
        chunk_count += 1

    print(f"\n\nTotal chunks: {chunk_count}")
    print(f"Total chars : {len(full_content)}")
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
async def main():
    await example_basic_astream()
    await example_astream_messages()
    await example_astream_chain()
    await example_concurrent_streaming()
    await example_astream_with_config()

    print("✅ astream() demo complete.")
    print("   KEY TAKEAWAY: astream() = stream() + async — perfect for web servers.")


if __name__ == "__main__":
    asyncio.run(main())
