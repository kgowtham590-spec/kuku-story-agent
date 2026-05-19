"""
main.py
-------
FastAPI app exposing the Kuku Story Agent as REST endpoints.
Logs every query to a local SQLite database (mirrors MySQL schema).

Start with:
    uvicorn main:app --reload
"""

import sqlite3
import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import build_graph

load_dotenv()

app   = FastAPI(title="Kuku Story Agent", version="1.0.0")
graph = build_graph()

DB_PATH = "queries.db"


# ── Database setup ────────────────────────────────────────────────────────────

def get_db():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create query log table if it doesn't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                query        TEXT    NOT NULL,
                genre        TEXT,
                language     TEXT,
                max_duration INTEGER,
                stories_json TEXT,
                recommendation TEXT,
                created_at   TEXT    NOT NULL
            )
        """)
        conn.commit()


init_db()


# ── Request / Response schemas ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query:        str
    genre:        Optional[str] = None
    language:     Optional[str] = None
    max_duration: Optional[int] = None

class StoryOut(BaseModel):
    id:           int
    title:        str
    genre:        str
    language:     str
    duration_mins: int

class RecommendResponse(BaseModel):
    recommendation: str
    top_stories:    list[StoryOut]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick liveness check."""
    return {"status": "ok", "agent": "Kuku Story Agent v1.0"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Main endpoint — runs the 3-agent LangGraph pipeline and returns
    a personalised story recommendation.

    Example request:
        {
          "query": "I want a short horror story in Tamil under 10 minutes",
          "language": "Tamil",
          "max_duration": 10
        }
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = graph.invoke({
            "query":          req.query,
            "genre":          req.genre,
            "language":       req.language,
            "max_duration":   req.max_duration,
            "search_results": [],
            "ranked_results": [],
            "recommendation": "",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    top_stories = [StoryOut(**s) for s in result["ranked_results"]]

    # Log to SQLite
    import json
    with get_db() as conn:
        conn.execute(
            """INSERT INTO query_log
               (query, genre, language, max_duration, stories_json, recommendation, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                req.query,
                req.genre,
                req.language,
                req.max_duration,
                json.dumps([s.model_dump() for s in top_stories]),
                result["recommendation"],
                datetime.datetime.now().isoformat(),
            ),
        )
        conn.commit()

    return RecommendResponse(
        recommendation = result["recommendation"],
        top_stories    = top_stories,
    )


@app.get("/logs")
def get_logs(limit: int = 10):
    """Returns the last N queries logged to the database."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, query, language, genre, recommendation, created_at "
            "FROM query_log ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()

    return [
        {
            "id":             r[0],
            "query":          r[1],
            "language":       r[2],
            "genre":          r[3],
            "recommendation": r[4],
            "created_at":     r[5],
        }
        for r in rows
    ]
