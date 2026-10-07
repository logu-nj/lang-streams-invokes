"""
=============================================================
 03_stream.py  —  SYNCHRONOUS  stream()
=============================================================

WHAT IS stream()?
-----------------
- A SYNCHRONOUS generator that yields chunks as they arrive.
- Instead of waiting for the FULL response, you get pieces
  (tokens/chunks) one at a time, in real-time.
- Your program STILL blocks the thread (sync), BUT the user
  sees text appearing word-by-word — like ChatGPT's typewriter.
- Best for: CLI tools, terminal apps, Jupyter notebooks,
  anywhere you want real-time feedback without async complexity.

FLOW:
  stream() → yields chunk1 → yields chunk2 → ... → done
  (each chunk is an AIMessageChunk with partial .content)

COMPARISON:
  invoke()  → returns ONE complete response after waiting
  stream()  → yields MANY partial chunks as they stream in
"""

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
# EXAMPLE 1: Basic stream — see tokens arrive live
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 1: Basic stream() — tokens appear as they arrive")
print("=" * 60)

print("LLM Response: ", end="", flush=True)

full_response = ""  # We can reconstruct the full response manually

for chunk in llm.stream("Tell me a short joke about programmers."):
    # chunk is an AIMessageChunk — it has .content with partial text
    print(chunk.content, end="", flush=True)   # print without newline, flush immediately
    full_response += chunk.content             # accumulate

print("\n")  # newline after streaming finishes
print(f"Full reconstructed response:\n{full_response}")
print()


# ─────────────────────────────────────────────
# EXAMPLE 2: Inspecting chunks — what's inside each chunk?
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 2: Inspecting individual chunks")
print("=" * 60)

chunk_count = 0
for chunk in llm.stream("What is 2+2? Answer only with the number."):
    chunk_count += 1
    print(f"  Chunk #{chunk_count:02d} | type={type(chunk).__name__} | content={repr(chunk.content)}")

print(f"\nTotal chunks received: {chunk_count}")
print()


# ─────────────────────────────────────────────
# EXAMPLE 3: stream() with message list
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 3: stream() with messages list (chat-style)")
print("=" * 60)

messages = [
    SystemMessage(content="You are a creative storyteller. Be vivid but brief."),
    HumanMessage(content="Start a story about a robot who discovers music. 3 sentences."),
]

print("Story: ", end="", flush=True)
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 4: stream() on an LCEL chain
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 4: stream() on a prompt | llm chain")
print("=" * 60)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Answer concisely."),
    ("human", "{question}"),
])

chain = prompt | llm

print("Answer: ", end="", flush=True)
for chunk in chain.stream({
    "domain": "cooking",
    "question": "What are the 5 French mother sauces?",
}):
    print(chunk.content, end="", flush=True)
print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 5: stream() with config
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 5: stream() with runtime config")
print("=" * 60)

print("Answer: ", end="", flush=True)
for chunk in llm.stream(
    "List 3 famous scientists and one discovery each.",
    config={
        "run_name": "scientists-query",
        "tags": ["education", "stream-demo"],
        "metadata": {"file": "03_stream.py"},
    }
):
    print(chunk.content, end="", flush=True)
print("\n")


# ─────────────────────────────────────────────
# EXAMPLE 6: Collecting usage/metadata from LAST chunk
# The last chunk often carries usage metadata
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 6: Usage stats from the last chunk")
print("=" * 60)

last_chunk = None
print("Response: ", end="", flush=True)
for chunk in llm.stream("What is the speed of light?"):
    print(chunk.content, end="", flush=True)
    last_chunk = chunk  # keep updating to get the very last one

print("\n")
if last_chunk:
    print(f"Last chunk metadata : {last_chunk.response_metadata}")
    print(f"Usage metadata      : {last_chunk.usage_metadata}")

print()
print("✅ stream() demo complete.")
print("   KEY TAKEAWAY: stream() gives real-time output but still blocks the thread.")
