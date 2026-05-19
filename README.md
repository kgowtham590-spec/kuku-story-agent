# Kuku Story Agent

A story recommendation system built using LangChain, LangGraph and FAISS. You describe what kind of story you want and it finds the best matching Kuku-style stories and gives you a recommendation.

I built this because I wanted to learn how multi-agent systems actually work beyond just reading about them. The Kuku platform was a good use case since it has stories in multiple languages and genres which makes the search problem more interesting.

---

## How it works

Three agents run in sequence using LangGraph:

1. **Search** — takes your query, converts it to an embedding using a HuggingFace model, and finds the closest matching stories from a FAISS vector store
2. **Rank** — sends those results to an LLM and asks it to re-rank them based on what the user actually asked for
3. **Recommend** — takes the top 3 and generates a friendly recommendation message

The whole thing is exposed as a REST API using FastAPI.

---

## Stack

- LangGraph / LangChain for the agent pipeline
- FAISS for vector similarity search
- HuggingFace `all-MiniLM-L6-v2` for embeddings
- LLaMA 3.1 8B via Groq API
- FastAPI for the backend
- SQLite for logging queries
- Dataset: 25 stories covering Hindi, Tamil and English genres

---

## Running it locally

You'll need a free Groq API key from console.groq.com

```bash
git clone https://github.com/YOUR_USERNAME/kuku-story-agent
cd kuku-story-agent

python -m venv venv
venv\Scripts\activate  # on Mac: source venv/bin/activate

pip install -r requirements.txt

# copy .env.example to .env and add your Groq key
cp .env.example .env

# build the FAISS index first (only need to do this once)
python ingest.py

# start the server
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs to test it.

---

## Example request

```json
POST /recommend

{
  "query": "something scary in Tamil, short",
  "language": "Tamil",
  "max_duration": 10
}
```

Response:

```json
{
  "recommendation": "Kaala Saya is perfect for what you're looking for — a 6 minute Tamil horror set in a village near Madurai...",
  "top_stories": [
    {
      "id": 3,
      "title": "Kaala Saya",
      "genre": "horror",
      "language": "Tamil",
      "duration_mins": 6
    }
  ]
}
```

---

## Project structure

```
kuku-story-agent/
├── ingest.py        # reads CSV, embeds stories, saves FAISS index
├── agents.py        # LangGraph pipeline with 3 agents
├── main.py          # FastAPI server
├── data/
│   └── stories.csv  # mock story dataset
└── .env.example
```

---

## What I want to improve

- the mock dataset is only 25 stories, would be better with real data
- no evaluation yet — want to add test cases that check for hallucination and wrong language outputs
- the ranking agent sometimes returns unexpected formatting so there's a fallback but it's not great
- would be interesting to try fine-tuning a smaller model specifically for the recommendation step

---

## Notes

The dataset is fake — I couldn't get access to real Kuku story metadata so I wrote descriptions myself trying to match their content style. The architecture would work the same with real data.

Model used is llama-3.1-8b-instant on Groq. Originally tried gemma2-9b-it but Groq deprecated it.
