# Hiring Agent Platform

An AI-powered job matching agent that automatically matches candidates with relevant job listings using vector similarity search and LLM re-ranking.

Built as a portfolio project demonstrating production-grade AI agent engineering skills.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Agent workflow orchestration |
| **PostgreSQL + pgvector** | Vector database for similarity search |
| **Google Gemini** | Embedding generation & LLM re-ranking |
| **FastAPI** | REST API server |
| **LangSmith** | Agent monitoring & tracing |
| **Docker** | Database containerization |
| **asyncpg** | Async PostgreSQL driver |

---

## How It Works

```
Candidate resume (text or PDF)
        ↓
Node 1: Generate 768-dimension embedding via Gemini
        ↓
Node 2: Vector cosine similarity search in pgvector
        ↓
Node 3: LLM re-ranking + reasoning via Gemini
        ↓
JSON response with top matches + explanations
```

### Why Two-Stage Ranking?

Pure vector similarity finds jobs with similar text — but it cannot reason about context. For example, a "Full Stack Developer" role might score high for a backend specialist because of keyword overlap, but an LLM understands that the candidate lacks frontend skills.

By combining vector search (fast, scalable retrieval) with LLM re-ranking (contextual reasoning), the agent produces more accurate and explainable results.

---

## Project Structure

```
hiring_agent/
├── main.py              ← FastAPI server + LangGraph agent
├── agent.py             ← Standalone agent (for testing)
├── test_db.py           ← Database connection test
├── test_embedding.py    ← Embedding generation test
├── test_search.py       ← Vector search test
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Docker Desktop
- Google AI API Key (from https://aistudio.google.com/apikey)
- LangSmith API Key (from https://smith.langchain.com) — optional, for monitoring

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/hiring-agent.git
cd hiring-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:
```
DATABASE_URL=postgresql://hiring_user:hiring_pass@localhost:5432/hiring_agent
GOOGLE_API_KEY=your_google_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=hiring-agent
```

### 5. Start PostgreSQL + pgvector with Docker
```bash
docker run -d --name hiring_agent_db \
  -e POSTGRES_USER=hiring_user \
  -e POSTGRES_PASSWORD=hiring_pass \
  -e POSTGRES_DB=hiring_agent \
  -p 5432:5432 \
  pgvector/pgvector:0.8.2-pg18-trixie
```

### 6. Initialize the database
Run the SQL schema in DBeaver or psql:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE job_listings (
    job_id      SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    company     TEXT NOT NULL,
    location    TEXT,
    job_type    TEXT,
    description TEXT NOT NULL,
    requirements TEXT[],
    embedding   vector(768)
);

CREATE TABLE candidates (
    candidate_id    SERIAL PRIMARY KEY,
    name            TEXT,
    email           TEXT UNIQUE,
    resume_text     TEXT,
    parsed_profile  JSONB,
    embedding       vector(768),
    status          TEXT DEFAULT 'new'
);
```

### 7. Run the server
```bash
python main.py
```

### 8. Open API documentation
```
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/match-jobs` | Match jobs from plain text resume |
| POST | `/match-jobs-pdf` | Match jobs from PDF file upload |

### Example Request — Plain Text
```bash
curl -X POST http://localhost:8000/match-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_name": "John Doe",
    "resume_text": "Python developer with 5 years experience in FastAPI and PostgreSQL."
  }'
```

### Example Request — PDF Upload
```bash
curl -X POST "http://localhost:8000/match-jobs-pdf?candidate_name=John%20Doe" \
  -F "file=@resume.pdf"
```

### Example Response
```json
{
  "candidate_name": "John Doe",
  "matches": [
    {
      "title": "Senior Python Engineer",
      "company": "TechCorp Indonesia",
      "location": "Jakarta",
      "job_type": "full-time",
      "similarity": 0.7314
    }
  ],
  "llm_analysis": "RANKING:\n1. Senior Python Engineer — Direct match with candidate's Python and FastAPI expertise...\n\nREKOMENDASI:\nPrioritize Senior Python Engineer role."
}
```

---

## Agent Architecture

```
FastAPI Gateway
      │
      ▼
LangGraph Agent (AgentState)
      │
      ├── Node 1: embed_resume
      │     └── Gemini text-embedding-001 (768 dimensions)
      │
      ├── Node 2: search_jobs
      │     └── pgvector cosine similarity (<=> operator)
      │           └── ivfflat index (lists=10)
      │
      └── Node 3: llm_rerank
            └── Gemini LLM
                  ├── Contextual re-ranking
                  ├── Match reasoning
                  └── Final recommendation
```

---

## Monitoring

This project integrates with **LangSmith** for full agent observability:
- Trace every node execution
- Monitor latency per node
- Track token usage and API costs
- Debug agent failures

Access your traces at: https://smith.langchain.com

---

## Skills Demonstrated

- ✅ LangGraph agent workflow with typed state
- ✅ Vector database design and indexing (pgvector + ivfflat)
- ✅ Embedding generation and semantic search
- ✅ LLM integration with structured prompting
- ✅ Async Python patterns (asyncpg, FastAPI)
- ✅ REST API design with automatic documentation
- ✅ PDF parsing and text extraction
- ✅ Production monitoring with LangSmith
- ✅ Docker containerization

---

## License

MIT License — feel free to use this project as a reference for your own AI agent portfolio.
