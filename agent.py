import asyncio
import asyncpg
from google import genai
from dotenv import load_dotenv
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ── STATE ─────────────────────────────────────────
class AgentState(TypedDict):
    resume_text: str
    candidate_name: str
    embedding: Optional[str]
    matches: Optional[list]
    llm_analysis: Optional[str]
    error: Optional[str]


# ── NODE 1: Embed Resume ───────────────────────────
async def embed_resume(state: AgentState) -> AgentState:
    print(f"\n[Node 1] Embed resume milik {state['candidate_name']}...")
    
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=state['resume_text'],
        config={"output_dimensionality": 768}
    )
    embedding = response.embeddings[0].values
    embedding_str = str(list(embedding)).replace(" ", "")
    
    print(f"  ✅ Embedding didapat: {len(embedding)} dimensi")
    return {**state, "embedding": embedding_str}


# ── NODE 2: Search Jobs ────────────────────────────
async def search_jobs(state: AgentState) -> AgentState:
    print(f"\n[Node 2] Mencari job yang cocok...")
    
    conn = await asyncpg.connect(DATABASE_URL)
    embedding_str = state['embedding']
    
    results = await conn.fetch(f"""
        SELECT 
            title,
            company,
            location,
            job_type,
            1 - (embedding <=> '{embedding_str}'::vector) AS similarity
        FROM job_listings
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> '{embedding_str}'::vector
        LIMIT 3
    """)
    
    matches = [dict(r) for r in results]
    await conn.close()
    
    print(f"  ✅ Ditemukan {len(matches)} matches")
    return {**state, "matches": matches}


# ── NODE 3: Display Results ────────────────────────
async def display_results(state: AgentState) -> AgentState:
    print(f"\n[Node 4] Hasil untuk {state['candidate_name']}:")
    print("=" * 45)

    print("\n📊 Vector Similarity Matches:")
    for i, job in enumerate(state['matches'], 1):
        print(f"  {i}. {job['title']} — {job['company']}")
        print(f"     📍 {job['location']} ({job['job_type']})")
        print(f"     🎯 Similarity: {job['similarity']:.4f}")

    print("\n🤖 LLM Analysis:")
    print(state['llm_analysis'])

    return state

# ── NODE 4: LLM Re-rank & Explain ─────────────────
async def llm_rerank(state: AgentState) -> AgentState:
    print(f"\n[Node 4] LLM sedang analisis dan re-rank jobs...")

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

CANDIDATE JOBS (sudah diurutkan by vector similarity):
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

    llm_analysis = response.text
    print(f"  ✅ LLM analisis selesai")

    return {**state, "llm_analysis": llm_analysis}
# ── GRAPH ──────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("embed_resume", embed_resume)
    graph.add_node("search_jobs", search_jobs)
    graph.add_node("llm_rerank", llm_rerank)      
    graph.add_node("display_results", display_results)

    graph.set_entry_point("embed_resume")
    graph.add_edge("embed_resume", "search_jobs")
    graph.add_edge("search_jobs", "llm_rerank")    
    graph.add_edge("llm_rerank", "display_results")  
    graph.add_edge("display_results", END)

    return graph.compile()


# ── RUN ────────────────────────────────────────────
async def main():
    agent = build_agent()
    
    # Test dengan resume Budi
    result = await agent.ainvoke({
        "candidate_name": "Budi Santoso",
        "resume_text": "Python developer with 5 years experience. Skilled in FastAPI, PostgreSQL, Docker. Worked at fintech startup building REST APIs.",
        "embedding": None,
        "matches": None,
        "llm_analysis": None,  
        "error": None
    })

asyncio.run(main())