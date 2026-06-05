import asyncpg
import json
import time
from datetime import datetime, timezone, timedelta
from google import genai
from dotenv import load_dotenv
from typing import Optional, TypedDict
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader
import uvicorn
import io
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ── FastAPI app ────────────────────────────────────
app = FastAPI(
    title="Hiring Agent API",
    description="AI-powered job matching agent",
    version="1.0.0"
)

# ── Request & Response schema ──────────────────────
class MatchRequest(BaseModel):
    candidate_name: str
    resume_text: str

class JobMatch(BaseModel):
    title: str
    company: str
    location: str
    job_type: str
    similarity: float

class MatchResponse(BaseModel):
    candidate_name: str
    matches: list[JobMatch]
    llm_analysis: str


# ── STATE ──────────────────────────────────────────
class AgentState(TypedDict):
    resume_text: str
    candidate_name: str
    candidate_id: Optional[int]
    embedding: Optional[str]
    matches: Optional[list]
    llm_analysis: Optional[str]
    error: Optional[str]


# ── DB HELPERS ─────────────────────────────────────
async def upsert_candidate(conn, name: str, resume_text: str, embedding_str: str) -> int:
    existing = await conn.fetchrow(
        "SELECT candidate_id FROM candidates WHERE name = $1", name
    )
    if existing:
        await conn.execute("""
            UPDATE candidates
            SET resume_text = $1, embedding = $2::vector
            WHERE candidate_id = $3
        """, resume_text, embedding_str, existing['candidate_id'])
        return existing['candidate_id']
    else:
        row = await conn.fetchrow("""
            INSERT INTO candidates (name, resume_text, embedding, status)
            VALUES ($1, $2, $3::vector, 'new')
            RETURNING candidate_id
        """, name, resume_text, embedding_str)
        return row['candidate_id']


async def save_run_and_matches(
    candidate_id: int,
    matches: list,
    llm_analysis: str,
    resume_text: str,
    duration_ms: int
) -> int:
    conn = await asyncpg.connect(DATABASE_URL)

    now = datetime.now(timezone.utc)
    started_at = now - timedelta(milliseconds=duration_ms)

    run = await conn.fetchrow("""
        INSERT INTO agent_runs
            (agent_name, candidate_id, status, input_data, output_data, duration_ms, started_at, completed_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING run_id
    """,
        'hiring_agent',
        candidate_id,
        'completed',
        json.dumps({"resume_preview": resume_text[:300]}),
        json.dumps({"match_count": len(matches), "llm_analysis": llm_analysis}),
        duration_ms,
        started_at,
        now
    )
    run_id = run['run_id']

    for rank, job in enumerate(matches, 1):
        await conn.execute("""
            INSERT INTO matches (candidate_id, job_id, run_id, vector_score, llm_rank, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, candidate_id, job.get('job_id'), run_id, float(job['similarity']), rank, now)

    await conn.close()
    return run_id


# ── NODES ──────────────────────────────────────────
async def embed_resume(state: AgentState) -> AgentState:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=state['resume_text'],
        config={"output_dimensionality": 768}
    )
    embedding = response.embeddings[0].values
    embedding_str = str(list(embedding)).replace(" ", "")

    conn = await asyncpg.connect(DATABASE_URL)
    candidate_id = await upsert_candidate(
        conn, state['candidate_name'], state['resume_text'], embedding_str
    )
    await conn.close()

    return {**state, "embedding": embedding_str, "candidate_id": candidate_id}


async def search_jobs(state: AgentState) -> AgentState:
    conn = await asyncpg.connect(DATABASE_URL)
    embedding_str = state['embedding']

    results = await conn.fetch(f"""
        SELECT job_id, title, company, location, job_type,
               1 - (embedding <=> '{embedding_str}'::vector) AS similarity
        FROM job_listings
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> '{embedding_str}'::vector
        LIMIT 3
    """)

    matches = [dict(r) for r in results]
    await conn.close()
    return {**state, "matches": matches}


async def llm_rerank(state: AgentState) -> AgentState:
    jobs_text = ""
    for i, job in enumerate(state['matches'], 1):
        jobs_text += f"""
Job {i}: {job['title']} — {job['company']}
Lokasi: {job['location']} ({job['job_type']})
Similarity score: {job['similarity']:.4f}
"""

    prompt = f"""Kamu adalah AI recruiter yang ahli.

RESUME KANDIDAT:
{state['resume_text']}

CANDIDATE JOBS:
{jobs_text}

Tugasmu:
1. Re-rank job mana yang PALING cocok untuk kandidat ini
2. Berikan alasan konkret (1-2 kalimat) kenapa tiap job cocok atau tidak
3. Berikan final recommendation

Balas dalam format:
RANKING:
1. [nama job] — [alasan konkret]
2. [nama job] — [alasan konkret]
3. [nama job] — [alasan konkret]

REKOMENDASI:
[rekomendasi singkat]"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return {**state, "llm_analysis": response.text}


# ── GRAPH ──────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("embed_resume", embed_resume)
    graph.add_node("search_jobs", search_jobs)
    graph.add_node("llm_rerank", llm_rerank)

    graph.set_entry_point("embed_resume")
    graph.add_edge("embed_resume", "search_jobs")
    graph.add_edge("search_jobs", "llm_rerank")
    graph.add_edge("llm_rerank", END)

    return graph.compile()

agent = build_agent()


# ── ENDPOINTS ──────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Hiring Agent API is running!"}

@app.post("/match-jobs", response_model=MatchResponse)
async def match_jobs(request: MatchRequest):
    start = time.time()
    try:
        result = await agent.ainvoke({
            "candidate_name": request.candidate_name,
            "resume_text": request.resume_text,
            "candidate_id": None,
            "embedding": None,
            "matches": None,
            "llm_analysis": None,
            "error": None
        })

        duration_ms = int((time.time() - start) * 1000)
        await save_run_and_matches(
            result['candidate_id'],
            result['matches'],
            result['llm_analysis'],
            request.resume_text,
            duration_ms
        )

        return MatchResponse(
            candidate_name=result['candidate_name'],
            matches=[JobMatch(**{k: v for k, v in m.items() if k != 'job_id'}) for m in result['matches']],
            llm_analysis=result['llm_analysis']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match-jobs-pdf", response_model=MatchResponse)
async def match_jobs_pdf(
    candidate_name: str,
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF!")

    pdf_bytes = await file.read()
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))

    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="PDF tidak bisa dibaca atau kosong!")

    start = time.time()
    result = await agent.ainvoke({
        "candidate_name": candidate_name,
        "resume_text": resume_text,
        "candidate_id": None,
        "embedding": None,
        "matches": None,
        "llm_analysis": None,
        "error": None
    })

    duration_ms = int((time.time() - start) * 1000)
    await save_run_and_matches(
        result['candidate_id'],
        result['matches'],
        result['llm_analysis'],
        resume_text,
        duration_ms
    )

    return MatchResponse(
        candidate_name=result['candidate_name'],
        matches=[JobMatch(**{k: v for k, v in m.items() if k != 'job_id'}) for m in result['matches']],
        llm_analysis=result['llm_analysis']
    )


# ── RUN ────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
