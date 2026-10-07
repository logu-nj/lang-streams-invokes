"""
=============================================================
 06_langgraph_invoke.py  —  LangGraph  graph.invoke()
=============================================================

WHAT IS LangGraph?
------------------
LangGraph lets you build STATEFUL, MULTI-STEP workflows as a
directed graph (nodes + edges). Each node is a function that
reads/writes a shared state dict.

WHAT IS graph.invoke()?
-----------------------
- Runs the ENTIRE graph synchronously from START to END.
- Blocks until all nodes have completed.
- Returns the FINAL state dict after all nodes ran.
- Best for: simple workflows, testing, batch processing.

GRAPH CONCEPTS:
  StateGraph   → the graph builder (holds nodes + edges + state schema)
  State        → a TypedDict shared across all nodes
  Node         → a Python function: (state) → dict of updates
  Edge         → a connection between nodes
  START        → the entry point of the graph
  END          → the terminal point of the graph
  Conditional  → a function that decides WHICH node to go to next

FLOW:
  graph.invoke(initial_state)
  → runs node1 → node2 → ... → END
  → returns final state
"""

from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────
llm = ChatOllama(model="llama3", temperature=0.7, num_predict=256)


# ─────────────────────────────────────────────
# EXAMPLE 1: Simplest possible graph
#   START → chat_node → END
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 1: Simple single-node graph")
print("=" * 60)

# ── Define the State ─────────────────────────────────────────
# State is a TypedDict — all nodes read from and write to this.
# `add_messages` is a special reducer: instead of overwriting the
# messages list, it APPENDS new messages to it.
class SimpleChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ── Define a Node ────────────────────────────────────────────
# A node is just a function: takes state, returns dict of updates.
def chat_node(state: SimpleChatState) -> dict:
    """Call the LLM with the current message history."""
    response = llm.invoke(state["messages"])
    # Return a dict — LangGraph merges it into the state
    return {"messages": [response]}

# ── Build the Graph ──────────────────────────────────────────
builder = StateGraph(SimpleChatState)
builder.add_node("chat", chat_node)           # register the node
builder.add_edge(START, "chat")               # START → chat
builder.add_edge("chat", END)                 # chat → END
graph = builder.compile()                     # compile to a runnable

# ── Run with invoke() ────────────────────────────────────────
initial_state = {
    "messages": [HumanMessage(content="What is LangGraph in one sentence?")]
}

final_state = graph.invoke(initial_state)

# final_state is the complete state dict after the graph ran
print(f"State keys     : {list(final_state.keys())}")
print(f"Messages count : {len(final_state['messages'])}")
print(f"Last message   : {final_state['messages'][-1].content}")
print()


# ─────────────────────────────────────────────
# EXAMPLE 2: Multi-node graph
#   START → preprocess → llm_call → postprocess → END
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 2: Multi-node graph with preprocessing")
print("=" * 60)

class PipelineState(TypedDict):
    user_input: str
    processed_input: str
    response: str
    word_count: int

def preprocess_node(state: PipelineState) -> dict:
    """Clean and prepare the user input."""
    cleaned = state["user_input"].strip().lower()
    processed = f"Please answer this question clearly: {cleaned}"
    print(f"  [preprocess_node] Input  : '{state['user_input']}'")
    print(f"  [preprocess_node] Output : '{processed}'")
    return {"processed_input": processed}

def llm_node(state: PipelineState) -> dict:
    """Call the LLM."""
    print(f"  [llm_node] Calling LLM...")
    response = llm.invoke(state["processed_input"])
    return {"response": response.content}

def postprocess_node(state: PipelineState) -> dict:
    """Count words in the response."""
    word_count = len(state["response"].split())
    print(f"  [postprocess_node] Word count: {word_count}")
    return {"word_count": word_count}

builder2 = StateGraph(PipelineState)
builder2.add_node("preprocess", preprocess_node)
builder2.add_node("llm_call", llm_node)
builder2.add_node("postprocess", postprocess_node)

builder2.add_edge(START, "preprocess")
builder2.add_edge("preprocess", "llm_call")
builder2.add_edge("llm_call", "postprocess")
builder2.add_edge("postprocess", END)

graph2 = builder2.compile()

result = graph2.invoke({"user_input": "  What is Python?  "})
print(f"\nFinal response : {result['response'][:100]}...")
print(f"Word count     : {result['word_count']}")
print()


# ─────────────────────────────────────────────
# EXAMPLE 3: Conditional edges (branching)
#   START → classify → [short_answer | detailed_answer] → END
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 3: Conditional edges (branching logic)")
print("=" * 60)

class BranchState(TypedDict):
    question: str
    complexity: str   # "simple" or "complex"
    answer: str

def classify_node(state: BranchState) -> dict:
    """Classify the question as simple or complex."""
    prompt = f"Is this question simple (factual, one-word) or complex (needs explanation)? Reply ONLY with 'simple' or 'complex'.\nQuestion: {state['question']}"
    result = llm.invoke(prompt)
    complexity = "simple" if "simple" in result.content.lower() else "complex"
    print(f"  [classify_node] Classified as: {complexity}")
    return {"complexity": complexity}

def simple_answer_node(state: BranchState) -> dict:
    """Give a brief answer for simple questions."""
    print("  [simple_answer_node] Generating brief answer...")
    response = llm.invoke(f"Answer this in ONE sentence: {state['question']}")
    return {"answer": response.content}

def detailed_answer_node(state: BranchState) -> dict:
    """Give a detailed answer for complex questions."""
    print("  [detailed_answer_node] Generating detailed answer...")
    response = llm.invoke(f"Give a thorough explanation (3-4 sentences): {state['question']}")
    return {"answer": response.content}

# ── Router function — returns the name of the next node ──────
def route_by_complexity(state: BranchState) -> str:
    """Decide which node to go to based on state."""
    if state["complexity"] == "simple":
        return "simple_answer"
    return "detailed_answer"

builder3 = StateGraph(BranchState)
builder3.add_node("classify", classify_node)
builder3.add_node("simple_answer", simple_answer_node)
builder3.add_node("detailed_answer", detailed_answer_node)

builder3.add_edge(START, "classify")
# add_conditional_edges: from "classify", call router, route to different nodes
builder3.add_conditional_edges(
    "classify",
    route_by_complexity,
    {
        "simple_answer": "simple_answer",
        "detailed_answer": "detailed_answer",
    }
)
builder3.add_edge("simple_answer", END)
builder3.add_edge("detailed_answer", END)

graph3 = builder3.compile()

# Test with a simple question
print("Testing with simple question:")
r1 = graph3.invoke({"question": "What is the capital of France?"})
print(f"Answer: {r1['answer']}\n")

# Test with a complex question
print("Testing with complex question:")
r2 = graph3.invoke({"question": "How does transformer architecture work in LLMs?"})
print(f"Answer: {r2['answer'][:150]}...\n")


# ─────────────────────────────────────────────
# EXAMPLE 4: invoke() with config (run_name, tags, etc.)
# ─────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 4: graph.invoke() with config")
print("=" * 60)

result = graph.invoke(
    {"messages": [HumanMessage(content="What is a graph database?")]},
    config={
        "run_name": "graph-db-query",
        "tags": ["langgraph", "demo"],
        "metadata": {"file": "06_langgraph_invoke.py"},
        "recursion_limit": 10,   # max number of node steps (prevents infinite loops)
    }
)

print(f"Answer: {result['messages'][-1].content}")

print("\n✅ LangGraph invoke() demo complete.")
print("   KEY TAKEAWAY: graph.invoke() runs the whole graph and returns the final state.")
