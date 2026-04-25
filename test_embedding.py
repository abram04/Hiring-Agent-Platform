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

async def embed_all():
    conn = await asyncpg.connect(DATABASE_URL)

    # ── Kandidat — hanya yang belum punya embedding ────
    candidates = await conn.fetch("""
        SELECT candidate_id, name, resume_text 
        FROM candidates 
        WHERE embedding IS NULL
    """)
    
    if len(candidates) == 0:
        print("✅ Semua kandidat sudah punya embedding, skip!")
    else:
        print(f"Generate embedding untuk {len(candidates)} kandidat...")
        for c in candidates:
            embedding_str = get_embedding(c['resume_text'])
            await conn.execute(
                "UPDATE candidates SET embedding = $1::vector WHERE candidate_id = $2",
                embedding_str, c['candidate_id']
            )
            print(f"  ✅ {c['name']}")

    # ── Job Listings — hanya yang belum punya embedding ─
    jobs = await conn.fetch("""
        SELECT job_id, title, description 
        FROM job_listings 
        WHERE embedding IS NULL
    """)
    
    if len(jobs) == 0:
        print("✅ Semua job listings sudah punya embedding, skip!")
    else:
        print(f"\nGenerate embedding untuk {len(jobs)} job listings...")
        for j in jobs:
            embedding_str = get_embedding(j['description'])
            await conn.execute(
                "UPDATE job_listings SET embedding = $1::vector WHERE job_id = $2",
                embedding_str, j['job_id']
            )
            print(f"  ✅ {j['title']}")

    await conn.close()
    print("\n🎉 Selesai!")

asyncio.run(embed_all())