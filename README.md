# 🦙 LangChain + LangGraph Invocation Methods — Complete Guide

> **Local LLM:** Ollama (`llama3`)  
> **Focus:** Understanding every way to call an LLM — when, why, and how.

---

## 📁 File Structure

| File | Topic | Sync/Async |
|------|-------|-----------|
| [`01_invoke.py`](./01_invoke.py) | `invoke()` — blocking full response | Sync |
| [`02_ainvoke.py`](./02_ainvoke.py) | `ainvoke()` — async full response + concurrent calls | Async |
| [`03_stream.py`](./03_stream.py) | `stream()` — blocking token-by-token | Sync |
| [`04_astream.py`](./04_astream.py) | `astream()` — async token-by-token | Async |
| [`05_astream_events.py`](./05_astream_events.py) | `astream_events()` — full event observability | Async |
| [`06_langgraph_invoke.py`](./06_langgraph_invoke.py) | LangGraph `graph.invoke()` — single/multi-node, branching | Sync |
| [`07_langgraph_stream.py`](./07_langgraph_stream.py) | LangGraph `graph.stream()` — all stream modes | Sync/Async |
| [`08_langgraph_astream_events.py`](./08_langgraph_astream_events.py) | LangGraph `graph.astream_events()` — full observability | Async |
| [`09_comparison_summary.py`](./09_comparison_summary.py) | Side-by-side comparison of all methods | Both |

---

## ⚡ Quick Setup

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3

# 3. Install Python packages
py -m pip install -r requirements.txt

# 4. Run any file
py 01_invoke.py
py 02_ainvoke.py
# ...
py 09_comparison_summary.py   ← start here for the overview
```

---

## 🧠 Decision Guide — Which Method Should I Use?

```
Do you need async (web server, FastAPI)?
├── YES → Do you need token streaming?
│         ├── YES → Do you need step visibility / events?
│         │         ├── YES → astream_events()
│         │         └── NO  → astream()
│         └── NO  → Do you need multiple concurrent calls?
│                   ├── YES → ainvoke() + asyncio.gather()
│                   └── NO  → ainvoke()
└── NO  → Do you need token streaming?
          ├── YES → stream()
          └── NO  → invoke()
```

---

## 📊 Complete Comparison Table

| Method | Sync/Async | Blocks Thread? | Returns | Best For |
|--------|-----------|---------------|---------|----------|
| `invoke()` | Sync | ✅ YES | Complete `AIMessage` | Scripts, batch, tests |
| `ainvoke()` | Async | ❌ NO | Complete `AIMessage` | Concurrent LLM calls |
| `stream()` | Sync | ✅ YES | `AIMessageChunk` iterator | CLI tools, notebooks |
| `astream()` | Async | ❌ NO | `AIMessageChunk` iterator | Web APIs, WebSockets |
| `astream_events()` | Async | ❌ NO | Event `dict` iterator | Debugging, observability |

### LangGraph Graph Methods

| Method | Returns | Best For |
|--------|---------|----------|
| `graph.invoke()` | Final state dict | Simple workflows |
| `graph.ainvoke()` | Final state dict (async) | Async workflows |
| `graph.stream("updates")` | Per-node state changes | Step-by-step progress |
| `graph.stream("values")` | Full state after each node | Full state tracking |
| `graph.stream("messages")` | Tokens from LLM nodes | Chat token streaming |
| `graph.astream()` | Same as stream() but async | Async graph execution |
| `graph.astream_events()` | Full event hierarchy | Full observability |

---

## 🔑 Key Concepts

### `invoke()` — The Simplest
```python
response = llm.invoke("Your question")  # blocks until done
print(response.content)                 # AIMessage
```

### `ainvoke()` — Async Power: Run Multiple at Once
```python
# 3x faster than sequential!
r1, r2, r3 = await asyncio.gather(
    llm.ainvoke("Q1"),
    llm.ainvoke("Q2"),
    llm.ainvoke("Q3"),
)
```

### `stream()` — Live Output in Terminal
```python
for chunk in llm.stream("Your question"):
    print(chunk.content, end="", flush=True)  # words appear live
```

### `astream()` — Live Output in Web App
```python
async for chunk in llm.astream("Your question"):
    await websocket.send(chunk.content)  # stream to browser
```

### `astream_events()` — See Everything
```python
async for event in llm.astream_events("Your question", version="v2"):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
    # Also catches: on_chat_model_start, on_chat_model_end, etc.
```

### LangGraph `stream_mode`
```python
# 4 modes you need to know:
graph.stream(input, stream_mode="updates")   # what each node changed
graph.stream(input, stream_mode="values")    # full state after each node
graph.stream(input, stream_mode="messages")  # token-by-token
graph.stream(input, stream_mode="debug")     # verbose debug info
```

---

## ⚙️ Configuration Options

Every method accepts a `config` dict:

```python
llm.invoke("...", config={
    "run_name": "my-run",           # label in LangSmith traces
    "tags": ["prod", "v2"],         # filter tags
    "metadata": {"user": "alice"},  # arbitrary key-value pairs
    "callbacks": [...],             # custom callback handlers
})

graph.invoke(state, config={
    "recursion_limit": 25,          # max nodes executed (prevents infinite loops)
    "run_name": "my-graph-run",
    "tags": ["langgraph"],
})
```

### Model Configuration (Ollama)
```python
llm = ChatOllama(
    model="llama3",          # model name (must be pulled)
    temperature=0.7,           # 0.0=deterministic, 1.0=creative
    num_predict=256,           # max tokens to generate
    top_p=0.9,                 # nucleus sampling
    top_k=40,                  # top-k sampling
    base_url="http://localhost:11434",  # Ollama server URL
)
```

---

## 🎯 astream_events — Event Types Reference

| Event Type | When it fires |
|-----------|--------------|
| `on_chain_start` | A chain/graph/node starts |
| `on_chain_stream` | Intermediate output from a node |
| `on_chain_end` | A chain/graph/node finishes |
| `on_chat_model_start` | LLM starts generating |
| `on_chat_model_stream` | A token arrives ← **most used** |
| `on_chat_model_end` | LLM finishes |
| `on_prompt_start` | Prompt template starts |
| `on_prompt_end` | Prompt template finishes |
| `on_tool_start` | A tool call begins |
| `on_tool_end` | A tool call finishes |

### Filtering Events
```python
# By event name (component name)
graph.astream_events(..., include_names=["MyLLM"])

# By component type
graph.astream_events(..., include_types=["chat_model"])

# By tags
graph.astream_events(..., include_tags=["my-tag"])
```
