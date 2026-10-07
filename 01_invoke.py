"""
=============================================================
 01_invoke.py  —  SYNCHRONOUS  invoke()
=============================================================

WHAT IS invoke()?
-----------------
- The most basic, BLOCKING call.
- You call it, the LLM processes the ENTIRE response internally,
  and ONLY THEN does it return a complete AIMessage object.
- Your program is FROZEN / BLOCKED until the full response arrives.
- Best for: scripts, batch jobs, unit tests, simple one-shot calls.

FLOW:
  Your code  →  invoke()  →  [waits...]  →  returns full AIMessage
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# 1. Basic Model Configuration
# ─────────────────────────────────────────────
# model       : which Ollama model to use (must be pulled via `ollama pull <name>`)
# temperature : 0.0 = deterministic, 1.0 = very creative
# num_predict : max tokens to generate (-1 = unlimited)
# top_p       : nucleus sampling (0.9 = consider top 90% probability mass)
# top_k       : only sample from top-k tokens (40 is a common default)

llm = ChatOllama(
    model="llama3",        # Change to your pulled model name
    temperature=0.7,
    num_predict=256,
    top_p=0.9,
    top_k=40,
    # base_url="http://localhost:11434",  # Default Ollama URL (optional)
)

print("=" * 60)
print("EXAMPLE 1: Simple string invoke")
print("=" * 60)

# ── Simplest call: pass a plain string ──────────────────────
response = llm.invoke("What is the capital of France? Answer in one sentence.")

# response is an AIMessage object
print(f"Type    : {type(response)}")           # <class 'langchain_core.messages.ai.AIMessage'>
print(f"Content : {response.content}")         # The actual text
print(f"Metadata: {response.response_metadata}")  # token counts, model name, etc.
print()


print("=" * 60)
print("EXAMPLE 2: invoke() with a list of messages (conversation)")
print("=" * 60)

# ── Pass a list of messages (chat-style) ─────────────────────
messages = [
    SystemMessage(content="You are a helpful assistant that answers concisely."),
    HumanMessage(content="Explain what Python is in 2 sentences."),
]

response = llm.invoke(messages)
print(f"Content : {response.content}")
print()


print("=" * 60)
print("EXAMPLE 3: invoke() with a PromptTemplate (chain)")
print("=" * 60)

# ── Using a prompt template + pipe (LCEL chain) ──────────────
# LCEL = LangChain Expression Language
# The pipe | creates a chain: prompt → llm
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "{question}"),
])

# Chain = prompt | llm
chain = prompt | llm

# invoke() on a chain takes a dict matching the template variables
response = chain.invoke({
    "domain": "machine learning",
    "question": "What is overfitting? Keep it under 3 sentences.",
})

print(f"Content : {response.content}")
print()


print("=" * 60)
print("EXAMPLE 4: invoke() with additional runtime config")
print("=" * 60)

# ── Runtime configuration via config dict ────────────────────
# config allows you to pass:
#   - "run_name"   : label this run for tracing
#   - "tags"       : list of strings for filtering in LangSmith
#   - "metadata"   : arbitrary key/value pairs for logging
#   - "callbacks"  : custom callback handlers
#   - "max_concurrency": for batch operations

response = llm.invoke(
    "List 3 benefits of exercise.",
    config={
        "run_name": "exercise-query",
        "tags": ["health", "demo"],
        "metadata": {"user_id": "user_001", "session": "demo"},
    }
)

print(f"Content : {response.content}")
print()

print("✅ invoke() demo complete — notice how ALL responses printed AFTER the full LLM call finished.")
