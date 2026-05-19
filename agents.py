"""
agents.py
---------
Builds a 3-node LangGraph pipeline:

    [search_agent] --> [rank_agent] --> [recommend_agent] --> END

- search_agent  : Retrieves top stories from FAISS
- rank_agent    : Re-ranks by duration / language fit using Gemma
- recommend_agent: Generates a warm, personalised recommendation
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

from ingest import search_stories

load_dotenv()

MODEL = "llama-3.1-8b-instant"


# ── State shared across all nodes ─────────────────────────────────────────────

class StoryState(TypedDict):
    query:          str
    genre:          Optional[str]
    language:       Optional[str]
    max_duration:   Optional[int]
    search_results: list[dict]
    ranked_results: list[dict]
    recommendation: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

def search_agent(state: StoryState) -> StoryState:
    """Node 1 — Semantic search over FAISS vector store."""
    print("[search_agent] Searching FAISS ...")
    results = search_stories(
        query           = state["query"],
        genre_filter    = state.get("genre"),
        language_filter = state.get("language"),
        max_duration    = state.get("max_duration"),
        k               = 6,
    )
    print(f"[search_agent] Found {len(results)} stories")
    return {**state, "search_results": results}


def rank_agent(state: StoryState) -> StoryState:
    """
    Node 2 — Uses Gemma to re-rank results by relevance to the user query.
    Returns the top 3 story IDs in order.
    """
    print("[rank_agent] Re-ranking with Gemma ...")
    llm = ChatGroq(model=MODEL, temperature=0)

    stories_text = "\n".join([
        f"ID {s['id']}: {s['title']} | {s['genre']} | {s['language']} | "
        f"{s['duration_mins']} mins"
        for s in state["search_results"]
    ])

    messages = [
        SystemMessage(content=(
            "You are a story ranking engine. "
            "Reply with ONLY a comma-separated list of story IDs (e.g. 3,7,12) "
            "ranked from most to least relevant. No extra text."
        )),
        HumanMessage(content=(
            f"User request: {state['query']}\n\n"
            f"Stories to rank:\n{stories_text}"
        )),
    ]

    response  = llm.invoke(messages)
    id_string = response.content.strip()

    try:
        ranked_ids = [int(x.strip()) for x in id_string.split(",")]
    except ValueError:
        # Gemma gave unexpected output — keep original order
        ranked_ids = [s["id"] for s in state["search_results"]]

    id_to_story = {s["id"]: s for s in state["search_results"]}
    ranked = [id_to_story[i] for i in ranked_ids if i in id_to_story][:3]

    print(f"[rank_agent] Top 3 IDs: {[s['id'] for s in ranked]}")
    return {**state, "ranked_results": ranked}


def recommend_agent(state: StoryState) -> StoryState:
    """Node 3 — Generates a warm, conversational recommendation using Gemma."""
    print("[recommend_agent] Generating recommendation ...")
    llm = ChatGroq(model=MODEL, temperature=0.7)

    stories_text = "\n".join([
        f"- '{s['title']}' ({s['genre']}, {s['language']}, {s['duration_mins']} mins)"
        for s in state["ranked_results"]
    ])

    messages = [
        SystemMessage(content=(
            "You are Kuku's AI story curator. You recommend Indian audio stories "
            "with warmth and personality. Keep responses under 120 words. "
            "Mention each story title and one sentence on why it fits the user's mood."
        )),
        HumanMessage(content=(
            f"User asked for: {state['query']}\n\n"
            f"Top matching stories:\n{stories_text}\n\n"
            "Write a short, friendly recommendation."
        )),
    ]

    response = llm.invoke(messages)
    print("[recommend_agent] Done.")
    return {**state, "recommendation": response.content}


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(StoryState)

    graph.add_node("search",    search_agent)
    graph.add_node("rank",      rank_agent)
    graph.add_node("recommend", recommend_agent)

    graph.set_entry_point("search")
    graph.add_edge("search",    "rank")
    graph.add_edge("rank",      "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


# ── Quick local test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "query":          "I want a short horror story in Tamil",
        "genre":          None,
        "language":       "Tamil",
        "max_duration":   10,
        "search_results": [],
        "ranked_results": [],
        "recommendation": "",
    })
    print("\n── RECOMMENDATION ──")
    print(result["recommendation"])
    print("\n── TOP STORIES ──")
    for s in result["ranked_results"]:
        print(f"  {s['title']} ({s['language']}, {s['duration_mins']} mins)")
