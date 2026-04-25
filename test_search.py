import asyncio
import asyncpg
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embedding(text: str) -> str:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 768}
    )
    embedding = response.embeddings[0].values
    return str(list(embedding)).replace(" ", "")

async def search_jobs_for_candidate(candidate_id: int):
    conn = await asyncpg.connect(DATABASE_URL)

    # Ambil resume kandidat
    candidate = await conn.fetchrow(
        "SELECT name, resume_text FROM candidates WHERE candidate_id = $1",
        candidate_id
    )
    print(f"Mencari job untuk: {candidate['name']}")
    print(f"Resume: {candidate['resume_text'][:80]}...")

    # Generate embedding dari resume
    embedding_str = get_embedding(candidate['resume_text'])

    # Vector similarity search!
    results = await conn.fetch(f"""
        SELECT 
            title,
            company,
            location,
            1 - (embedding <=> '{embedding_str}'::vector) AS similarity
        FROM job_listings
        ORDER BY embedding <=> '{embedding_str}'::vector
        LIMIT 3
    """)

    print(f"\n🎯 Top 3 job matches:")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['title']} — {r['company']}")
        print(f"     📍 {r['location']}")
        print(f"     🎯 Similarity: {r['similarity']:.4f}")

    await conn.close()

# Test untuk semua kandidat
async def main():
    for candidate_id in [1, 2, 3]:
        print("\n" + "="*50)
        await search_jobs_for_candidate(candidate_id)

asyncio.run(main())